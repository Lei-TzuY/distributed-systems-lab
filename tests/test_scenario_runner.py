from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner
from distlab.simulator import FaultAction, FaultRule


def _workload(*actions: ClientWorkloadAction) -> SeededClientWorkloadSchedule:
    return SeededClientWorkloadSchedule(seed=7, actions=actions)


def test_explicit_schedule_replays_to_identical_trace_and_history() -> None:
    workload = _workload(
        ClientWorkloadAction(
            operation_id="write",
            client_id="writer",
            node_id="n1",
            kind=ClientOperationKind.PUT,
            key="x",
            value="one",
            request_id=1,
        ),
        ClientWorkloadAction(
            operation_id="read",
            client_id="reader",
            node_id="n1",
            kind=ClientOperationKind.GET,
            key="x",
        ),
    )
    faults = SeededFaultSchedule(seed=11, rules=())

    first = ReplicatedKVScenarioRunner(workload, faults).run()
    second = ReplicatedKVScenarioRunner(
        SeededClientWorkloadSchedule.from_json(workload.to_json()),
        SeededFaultSchedule.from_json(faults.to_json()),
    ).run()

    assert first.linearizability.linearizable
    assert first.linearizability.order == ("write", "read")
    assert first.history.events == second.history.events
    assert first.trace == second.trace
    assert first.snapshots == second.snapshots == {
        "n1": {"x": "one"},
        "n2": {"x": "one"},
        "n3": {"x": "one"},
    }


def test_faulted_follower_produces_replayable_non_linearizable_stale_read() -> None:
    workload = _workload(
        ClientWorkloadAction(
            operation_id="write",
            client_id="writer",
            node_id="n1",
            kind=ClientOperationKind.PUT,
            key="x",
            value="one",
            request_id=1,
        ),
        ClientWorkloadAction(
            operation_id="stale-read",
            client_id="reader",
            node_id="n2",
            kind=ClientOperationKind.GET,
            key="x",
        ),
    )
    faults = SeededFaultSchedule(
        seed=19,
        rules=(
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=2),
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=3),
        ),
    )

    result = ReplicatedKVScenarioRunner(workload, faults).run()

    assert result.snapshots["n1"] == {"x": "one"}
    assert result.snapshots["n2"] == {}
    assert not result.linearizability.linearizable
    assert [record.kind for record in result.trace].count("drop") == 2


def test_write_stays_pending_when_response_replica_never_applies_request() -> None:
    workload = _workload(
        ClientWorkloadAction(
            operation_id="write",
            client_id="writer",
            node_id="n2",
            kind=ClientOperationKind.PUT,
            key="x",
            value="one",
            request_id=1,
        )
    )
    faults = SeededFaultSchedule(
        seed=23,
        rules=(
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=2),
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=3),
        ),
    )

    result = ReplicatedKVScenarioRunner(workload, faults).run()

    assert tuple(item.operation_id for item in result.history.pending()) == ("write",)
    assert result.linearizability.linearizable


def test_runner_rejects_multi_key_workload_until_checker_scope_expands() -> None:
    workload = _workload(
        ClientWorkloadAction(
            operation_id="a",
            client_id="reader",
            node_id="n1",
            kind=ClientOperationKind.GET,
            key="x",
        ),
        ClientWorkloadAction(
            operation_id="b",
            client_id="reader",
            node_id="n1",
            kind=ClientOperationKind.GET,
            key="y",
        ),
    )

    try:
        ReplicatedKVScenarioRunner(workload, SeededFaultSchedule(seed=1, rules=()))
    except ValueError as exc:
        assert "single KV key" in str(exc)
    else:
        raise AssertionError("multi-key workload must be rejected")
