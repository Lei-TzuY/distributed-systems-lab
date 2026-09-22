import pytest

from distlab.client_history import KVClientHistory
from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner, ScenarioExecutionError


def _single_write() -> SeededClientWorkloadSchedule:
    return SeededClientWorkloadSchedule(
        seed=31,
        actions=(
            ClientWorkloadAction(
                operation_id="write",
                client_id="writer",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
        ),
    )


def test_scenario_rejects_crashed_configured_leader_before_history_mutation(
    monkeypatch,
) -> None:
    invoked = False
    original = KVClientHistory.invoke_write

    def observe_invoke(self, *args, **kwargs):
        nonlocal invoked
        invoked = True
        return original(self, *args, **kwargs)

    monkeypatch.setattr(KVClientHistory, "invoke_write", observe_invoke)
    lifecycle = SeededLifecycleSchedule(
        seed=31,
        actions=(
            NodeLifecycleAction(
                action_id="crash-leader",
                node_id="n1",
                kind=NodeLifecycleKind.CRASH,
                before_action_index=0,
            ),
        ),
    )

    with pytest.raises(
        ScenarioExecutionError,
        match="active configured leader generation",
    ):
        ReplicatedKVScenarioRunner(
            _single_write(),
            SeededFaultSchedule(seed=31, rules=()),
            lifecycle=lifecycle,
        ).run()

    assert invoked is False


def test_scenario_fences_generation_loss_after_first_replication_round(
    monkeypatch,
) -> None:
    completed = False
    rounds = 0
    original_round = ReplicatedKVScenarioRunner._replicate_round
    original_complete = KVClientHistory.complete_write

    def retire_after_round(self, replicator):
        nonlocal rounds
        original_round(self, replicator)
        rounds += 1
        if rounds == 1:
            assert replicator.leader.step_down(reason="test-mid-write-retire")

    def observe_complete(self, *args, **kwargs):
        nonlocal completed
        completed = True
        return original_complete(self, *args, **kwargs)

    monkeypatch.setattr(
        ReplicatedKVScenarioRunner,
        "_replicate_round",
        retire_after_round,
    )
    monkeypatch.setattr(KVClientHistory, "complete_write", observe_complete)

    with pytest.raises(
        ScenarioExecutionError,
        match="first replication round",
    ):
        ReplicatedKVScenarioRunner(
            _single_write(),
            SeededFaultSchedule(seed=31, rules=()),
        ).run()

    assert rounds == 1
    assert completed is False


def test_generation_guard_does_not_add_runtime_supervisor_trace_events() -> None:
    result = ReplicatedKVScenarioRunner(
        _single_write(),
        SeededFaultSchedule(seed=31, rules=()),
    ).run()

    assert result.linearizability.linearizable is True
    assert not [
        record for record in result.trace if record.kind == "raft-leader-runtime-start"
    ]
