from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleGenerator,
    SeededLifecycleSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def test_seeded_lifecycle_schedule_round_trips_exactly() -> None:
    schedule = SeededLifecycleSchedule(
        seed=17,
        actions=(
            NodeLifecycleAction(
                action_id="crash-n2",
                node_id="n2",
                kind=NodeLifecycleKind.CRASH,
                before_action_index=0,
            ),
            NodeLifecycleAction(
                action_id="restart-n2",
                node_id="n2",
                kind=NodeLifecycleKind.RESTART,
                before_action_index=1,
            ),
        ),
    )

    assert SeededLifecycleSchedule.from_json(schedule.to_json()) == schedule


def test_seeded_lifecycle_generator_is_reproducible_and_state_valid() -> None:
    generator = SeededLifecycleGenerator(
        nodes=("n3", "n1", "n2"),
        crash_rate=0.45,
        restart_rate=0.45,
    )

    schedule = generator.compile(seed=29, boundary_count=24)

    assert schedule == generator.compile(seed=29, boundary_count=24)
    assert schedule.actions
    alive = {"n1", "n2", "n3"}
    crashed: set[str] = set()
    for action in schedule.actions:
        if action.kind is NodeLifecycleKind.CRASH:
            assert action.node_id in alive
            alive.remove(action.node_id)
            crashed.add(action.node_id)
        else:
            assert action.node_id in crashed
            crashed.remove(action.node_id)
            alive.add(action.node_id)


def test_follower_crash_restart_replays_and_catches_up_deterministically() -> None:
    workload = SeededClientWorkloadSchedule(
        seed=41,
        actions=(
            ClientWorkloadAction(
                operation_id="write-one",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="write-two",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="two",
                request_id=2,
            ),
        ),
    )
    lifecycle = SeededLifecycleSchedule(
        seed=41,
        actions=(
            NodeLifecycleAction(
                action_id="crash-n2",
                node_id="n2",
                kind=NodeLifecycleKind.CRASH,
                before_action_index=0,
            ),
            NodeLifecycleAction(
                action_id="restart-n2",
                node_id="n2",
                kind=NodeLifecycleKind.RESTART,
                before_action_index=1,
            ),
        ),
    )
    faults = SeededFaultSchedule(seed=41, rules=())

    first = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=lifecycle,
    ).run()
    second = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=SeededLifecycleSchedule.from_json(lifecycle.to_json()),
    ).run()

    assert first.linearizability.linearizable
    assert first.trace == second.trace
    assert first.snapshots == second.snapshots == {
        "n1": {"x": "two"},
        "n2": {"x": "two"},
        "n3": {"x": "two"},
    }
    lifecycle_records = [
        record for record in first.trace if record.kind == "scenario-lifecycle"
    ]
    assert [record.details["action"] for record in lifecycle_records] == [
        "crash",
        "restart",
    ]
    n2_applies = [
        record
        for record in first.trace
        if record.kind == "kv-apply" and record.details["node"] == "n2"
    ]
    assert [record.details["request_id"] for record in n2_applies] == [1, 2]


def test_lifecycle_schedule_rejects_invalid_runtime_transition() -> None:
    workload = SeededClientWorkloadSchedule(seed=5, actions=())
    lifecycle = SeededLifecycleSchedule(
        seed=5,
        actions=(
            NodeLifecycleAction(
                action_id="restart-live",
                node_id="n2",
                kind=NodeLifecycleKind.RESTART,
                before_action_index=0,
            ),
        ),
    )

    runner = ReplicatedKVScenarioRunner(
        workload,
        SeededFaultSchedule(seed=5, rules=()),
        lifecycle=lifecycle,
    )

    try:
        runner.run()
    except RuntimeError as exc:
        assert "cannot restart live node" in str(exc)
    else:
        raise AssertionError("invalid lifecycle transition should fail replay")
