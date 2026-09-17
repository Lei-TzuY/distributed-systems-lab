import json

import pytest

from distlab.joint_combined_fault_artifact import JointCombinedFaultFailureArtifact
from distlab.lifecycle import NodeLifecycleAction, NodeLifecycleKind, SeededLifecycleSchedule
from distlab.link_fault_schedule import LinkFaultAction, LinkFaultKind, SeededLinkFaultSchedule
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def _failure_inputs():
    workload = SeededClientWorkloadSchedule(
        seed=43,
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
    faults = SeededFaultSchedule(seed=43, rules=())
    lifecycle = SeededLifecycleSchedule(
        seed=43,
        actions=(
            NodeLifecycleAction(
                action_id="redundant-stop",
                node_id="n3",
                kind=next(iter(NodeLifecycleKind)),
                before_action_index=2,
            ),
        ),
    )
    link_faults = SeededLinkFaultSchedule(
        seed=43,
        actions=(
            LinkFaultAction(
                action_id="block",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="redundant-heal",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=2,
            ),
        ),
    )
    return workload, faults, lifecycle, link_faults


def _artifact():
    workload, faults, lifecycle, link_faults = _failure_inputs()
    result = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=lifecycle,
        link_faults=link_faults,
    ).run()
    assert not result.linearizability.linearizable
    return JointCombinedFaultFailureArtifact.capture(
        workload,
        faults,
        lifecycle,
        link_faults,
        result,
    )


def test_joint_combined_failure_artifact_round_trips_and_replays() -> None:
    artifact = _artifact()
    encoded = artifact.to_json()
    restored = JointCombinedFaultFailureArtifact.from_json(encoded)

    assert restored.to_json() == encoded
    assert restored.minimized_workload == restored.failure.workload
    assert restored.kept_workload_action_indices == (0, 1)
    assert restored.removed_workload_action_indices == ()
    assert restored.minimized_lifecycle.actions == ()
    assert restored.kept_lifecycle_action_indices == ()
    assert restored.removed_lifecycle_action_indices == (0,)
    assert restored.kept_link_fault_action_indices == (0,)
    assert restored.removed_link_fault_action_indices == (1,)
    restored.replay()


def test_joint_combined_failure_artifact_rejects_workload_projection_drift() -> None:
    raw = json.loads(_artifact().to_json())
    raw["kept_workload_action_indices"] = [1]
    raw["removed_workload_action_indices"] = [0]

    with pytest.raises(ValueError, match="workload"):
        JointCombinedFaultFailureArtifact.from_json(json.dumps(raw))


def test_joint_combined_failure_artifact_rejects_projection_drift() -> None:
    raw = json.loads(_artifact().to_json())
    raw["kept_link_fault_action_indices"] = [1]
    raw["removed_link_fault_action_indices"] = [0]

    with pytest.raises(ValueError, match="link fault"):
        JointCombinedFaultFailureArtifact.from_json(json.dumps(raw))


def test_joint_combined_failure_artifact_rejects_seed_drift() -> None:
    raw = json.loads(_artifact().to_json())
    raw["minimized_lifecycle"]["seed"] = 99

    with pytest.raises(ValueError, match="seed"):
        JointCombinedFaultFailureArtifact.from_json(json.dumps(raw))
