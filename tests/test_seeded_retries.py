import json

from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadGenerator,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner
from distlab.simulator import FaultAction, FaultRule


def test_retry_action_round_trips_and_old_workload_json_remains_readable() -> None:
    schedule = SeededClientWorkloadSchedule(
        seed=7,
        actions=(
            ClientWorkloadAction(
                operation_id="write",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="retry-attempt",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.RETRY,
                key="x",
                retry_of="write",
            ),
        ),
    )

    assert SeededClientWorkloadSchedule.from_json(schedule.to_json()) == schedule

    old_payload = json.loads(schedule.to_json())
    old_payload["actions"] = old_payload["actions"][:1]
    old_payload["actions"][0].pop("retry_of")
    restored = SeededClientWorkloadSchedule.from_json(json.dumps(old_payload))
    assert restored.actions[0].retry_of is None


def test_seeded_generator_can_emit_replayable_retry_attempts() -> None:
    generator = SeededClientWorkloadGenerator(
        clients=("client",),
        nodes=("n1", "n2", "n3"),
        keys=("x",),
        values=("one",),
        put_rate=0.45,
        delete_rate=0.0,
        retry_rate=0.45,
    )

    schedules = tuple(generator.compile(seed, 16) for seed in range(32))
    schedule = next(
        item
        for item in schedules
        if any(action.kind is ClientOperationKind.RETRY for action in item.actions)
    )

    assert schedule == generator.compile(schedule.seed, 16)
    writes = {
        action.operation_id: action
        for action in schedule.actions
        if action.kind in (ClientOperationKind.PUT, ClientOperationKind.DELETE)
    }
    for retry in (action for action in schedule.actions if action.kind is ClientOperationKind.RETRY):
        assert retry.retry_of in writes
        assert writes[retry.retry_of].client_id == retry.client_id
        assert writes[retry.retry_of].key == retry.key
        assert retry.request_id is None


def test_pending_write_retry_reuses_one_logical_history_operation() -> None:
    workload = SeededClientWorkloadSchedule(
        seed=23,
        actions=(
            ClientWorkloadAction(
                operation_id="write",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="retry-attempt",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.RETRY,
                key="x",
                retry_of="write",
            ),
        ),
    )
    faults = SeededFaultSchedule(
        seed=23,
        rules=(
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=2),
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=3),
        ),
    )

    result = ReplicatedKVScenarioRunner(workload, faults).run()

    assert result.linearizability.linearizable
    assert tuple(item.operation_id for item in result.history.invocations()) == ("write",)
    assert tuple(item.operation_id for item in result.history.completed()) == ("write",)
    assert result.history.pending() == ()
    retry_records = [record for record in result.trace if record.kind == "client-retry"]
    assert len(retry_records) == 1
    assert retry_records[0].details["operation_id"] == "write"
    assert result.snapshots == {
        "n1": {"x": "one"},
        "n2": {"x": "one"},
        "n3": {"x": "one"},
    }


def test_retry_after_response_is_deterministically_suppressed() -> None:
    workload = SeededClientWorkloadSchedule(
        seed=11,
        actions=(
            ClientWorkloadAction(
                operation_id="write",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="late-timeout",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.RETRY,
                key="x",
                retry_of="write",
            ),
        ),
    )

    result = ReplicatedKVScenarioRunner(
        workload,
        SeededFaultSchedule(seed=11, rules=()),
    ).run()

    assert result.linearizability.linearizable
    assert tuple(item.operation_id for item in result.history.invocations()) == ("write",)
    assert not any(record.kind == "client-retry" for record in result.trace)
    suppressed = [
        record for record in result.trace if record.kind == "client-retry-suppressed"
    ]
    assert len(suppressed) == 1
    assert suppressed[0].details["retry_of"] == "write"
