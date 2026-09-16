import json

from distlab.combined_fault_artifact import CombinedFaultFailureArtifact
from distlab.lifecycle import NodeLifecycleAction, NodeLifecycleKind, SeededLifecycleSchedule
from distlab.link_fault_schedule import LinkFaultAction, LinkFaultKind, SeededLinkFaultSchedule
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def _inputs():
    workload = SeededClientWorkloadSchedule(
        seed=41,
        actions=(
            ClientWorkloadAction(
                operation_id="put",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="get",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.GET,
                key="x",
            ),
        ),
    )
    faults = SeededFaultSchedule(seed=41, rules=())
    lifecycle = SeededLifecycleSchedule(
        seed=41,
        actions=(
            NodeLifecycleAction(
                action_id="stop-n3",
                node_id="n3",
                kind=next(iter(NodeLifecycleKind)),
                before_action_index=2,
            ),
        ),
    )
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="block",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="heal",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=2,
            ),
        ),
    )
    return workload, faults, lifecycle, link_faults


def test_combined_failure_artifact_round_trips_and_replays_exact_lifecycle() -> None:
    workload, faults, lifecycle, link_faults = _inputs()
    result = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=lifecycle,
        link_faults=link_faults,
    ).run()
    assert not result.linearizability.linearizable

    artifact = CombinedFaultFailureArtifact.capture(
        workload,
        faults,
        lifecycle,
        link_faults,
        result,
    )
    encoded = artifact.to_json()
    restored = CombinedFaultFailureArtifact.from_json(encoded)

    assert restored.to_json() == encoded
    assert restored.lifecycle == lifecycle
    assert restored.kept_link_fault_action_indices == (0,)
    assert restored.removed_link_fault_action_indices == (1,)
    replay = restored.replay()
    lifecycle_records = [record for record in replay.trace if record.kind == "scenario-lifecycle"]
    assert [record.details["action_id"] for record in lifecycle_records] == ["stop-n3"]


def test_combined_failure_artifact_rejects_lifecycle_seed_drift() -> None:
    workload, faults, lifecycle, link_faults = _inputs()
    result = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=lifecycle,
        link_faults=link_faults,
    ).run()
    artifact = CombinedFaultFailureArtifact.capture(
        workload,
        faults,
        lifecycle,
        link_faults,
        result,
    )
    raw = json.loads(artifact.to_json())
    raw["lifecycle"]["seed"] = 99

    try:
        CombinedFaultFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("lifecycle seed drift must be rejected")
