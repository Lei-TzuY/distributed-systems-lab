from __future__ import annotations

import json
from dataclasses import dataclass

from .campaign import FailureArtifactReplayMismatch
from .combined_fault_artifact import CombinedFaultFailureArtifact
from .joint_scenario_minimizer import NonLinearizableJointScenarioMinimizer
from .lifecycle import SeededLifecycleSchedule
from .link_fault_schedule import SeededLinkFaultSchedule
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioResult, ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class JointCombinedFaultFailureArtifact:
    """Persisted exact-replay evidence for a jointly minimized combined failure."""

    failure: CombinedFaultFailureArtifact
    minimized_workload: SeededClientWorkloadSchedule
    minimized_faults: SeededFaultSchedule
    minimized_lifecycle: SeededLifecycleSchedule
    minimized_link_faults: SeededLinkFaultSchedule
    kept_workload_action_indices: tuple[int, ...]
    removed_workload_action_indices: tuple[int, ...]
    kept_fault_rule_indices: tuple[int, ...]
    removed_fault_rule_indices: tuple[int, ...]
    kept_lifecycle_action_indices: tuple[int, ...]
    removed_lifecycle_action_indices: tuple[int, ...]
    kept_link_fault_action_indices: tuple[int, ...]
    removed_link_fault_action_indices: tuple[int, ...]

    @classmethod
    def capture(
        cls,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        lifecycle: SeededLifecycleSchedule,
        link_faults: SeededLinkFaultSchedule,
        result: ReplicatedKVScenarioResult,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> JointCombinedFaultFailureArtifact:
        failure = CombinedFaultFailureArtifact.capture(
            workload,
            faults,
            lifecycle,
            link_faults,
            result,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        reduction = NonLinearizableJointScenarioMinimizer().minimize(
            workload,
            faults,
            lifecycle,
            link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        return cls(
            failure=failure,
            minimized_workload=reduction.workload,
            minimized_faults=reduction.faults,
            minimized_lifecycle=reduction.lifecycle,
            minimized_link_faults=reduction.link_faults,
            kept_workload_action_indices=reduction.kept_workload_original_indices,
            removed_workload_action_indices=reduction.removed_workload_original_indices,
            kept_fault_rule_indices=reduction.kept_fault_original_indices,
            removed_fault_rule_indices=reduction.removed_fault_original_indices,
            kept_lifecycle_action_indices=reduction.kept_lifecycle_original_indices,
            removed_lifecycle_action_indices=reduction.removed_lifecycle_original_indices,
            kept_link_fault_action_indices=reduction.kept_link_fault_original_indices,
            removed_link_fault_action_indices=reduction.removed_link_fault_original_indices,
        )

    def to_json(self) -> str:
        payload = {
            "version": 3,
            "failure": json.loads(self.failure.to_json()),
            "minimized_workload": json.loads(self.minimized_workload.to_json()),
            "minimized_faults": json.loads(self.minimized_faults.to_json()),
            "minimized_lifecycle": json.loads(self.minimized_lifecycle.to_json()),
            "minimized_link_faults": json.loads(self.minimized_link_faults.to_json()),
            "kept_workload_action_indices": list(self.kept_workload_action_indices),
            "removed_workload_action_indices": list(self.removed_workload_action_indices),
            "kept_fault_rule_indices": list(self.kept_fault_rule_indices),
            "removed_fault_rule_indices": list(self.removed_fault_rule_indices),
            "kept_lifecycle_action_indices": list(self.kept_lifecycle_action_indices),
            "removed_lifecycle_action_indices": list(
                self.removed_lifecycle_action_indices
            ),
            "kept_link_fault_action_indices": list(
                self.kept_link_fault_action_indices
            ),
            "removed_link_fault_action_indices": list(
                self.removed_link_fault_action_indices
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> JointCombinedFaultFailureArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 3:
            raise ValueError("unsupported joint combined fault failure artifact format")
        try:
            failure = CombinedFaultFailureArtifact.from_json(json.dumps(raw["failure"]))
            workload = SeededClientWorkloadSchedule.from_json(
                json.dumps(raw["minimized_workload"])
            )
            faults = SeededFaultSchedule.from_json(json.dumps(raw["minimized_faults"]))
            lifecycle = SeededLifecycleSchedule.from_json(
                json.dumps(raw["minimized_lifecycle"])
            )
            link_faults = SeededLinkFaultSchedule.from_json(
                json.dumps(raw["minimized_link_faults"])
            )
            kept_workload = tuple(raw["kept_workload_action_indices"])
            removed_workload = tuple(raw["removed_workload_action_indices"])
            kept_faults = tuple(raw["kept_fault_rule_indices"])
            removed_faults = tuple(raw["removed_fault_rule_indices"])
            kept_lifecycle = tuple(raw["kept_lifecycle_action_indices"])
            removed_lifecycle = tuple(raw["removed_lifecycle_action_indices"])
            kept_links = tuple(raw["kept_link_fault_action_indices"])
            removed_links = tuple(raw["removed_link_fault_action_indices"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid joint combined fault failure artifact") from exc
        if len(
            {workload.seed, faults.seed, lifecycle.seed, link_faults.seed, failure.seed}
        ) != 1:
            raise ValueError("joint minimized schedules must match artifact seed")
        cls._validate_projection(
            kept_workload,
            removed_workload,
            failure.workload.actions,
            workload.actions,
            "workload",
        )
        cls._validate_projection(
            kept_faults,
            removed_faults,
            failure.faults.rules,
            faults.rules,
            "message fault",
        )
        cls._validate_projection(
            kept_lifecycle,
            removed_lifecycle,
            failure.lifecycle.actions,
            lifecycle.actions,
            "lifecycle",
        )
        cls._validate_projection(
            kept_links,
            removed_links,
            failure.link_faults.actions,
            link_faults.actions,
            "link fault",
        )
        return cls(
            failure=failure,
            minimized_workload=workload,
            minimized_faults=faults,
            minimized_lifecycle=lifecycle,
            minimized_link_faults=link_faults,
            kept_workload_action_indices=kept_workload,
            removed_workload_action_indices=removed_workload,
            kept_fault_rule_indices=kept_faults,
            removed_fault_rule_indices=removed_faults,
            kept_lifecycle_action_indices=kept_lifecycle,
            removed_lifecycle_action_indices=removed_lifecycle,
            kept_link_fault_action_indices=kept_links,
            removed_link_fault_action_indices=removed_links,
        )

    @staticmethod
    def _validate_projection(kept, removed, original, minimized, label: str) -> None:
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in kept + removed
        ):
            raise ValueError(f"joint {label} indices must be integers")
        if sorted(kept + removed) != list(range(len(original))):
            raise ValueError(
                f"joint {label} indices must partition the original schedule"
            )
        if minimized != tuple(original[index] for index in kept):
            raise ValueError(f"joint minimized {label} schedule must match kept indices")

    def replay(
        self,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> ReplicatedKVScenarioResult:
        result = self.failure.replay(node_ids=node_ids, leader_id=leader_id)
        reduction = NonLinearizableJointScenarioMinimizer().minimize(
            self.failure.workload,
            self.failure.faults,
            self.failure.lifecycle,
            self.failure.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        actual = (
            reduction.workload,
            reduction.faults,
            reduction.lifecycle,
            reduction.link_faults,
            reduction.kept_workload_original_indices,
            reduction.removed_workload_original_indices,
            reduction.kept_fault_original_indices,
            reduction.removed_fault_original_indices,
            reduction.kept_lifecycle_original_indices,
            reduction.removed_lifecycle_original_indices,
            reduction.kept_link_fault_original_indices,
            reduction.removed_link_fault_original_indices,
        )
        expected = (
            self.minimized_workload,
            self.minimized_faults,
            self.minimized_lifecycle,
            self.minimized_link_faults,
            self.kept_workload_action_indices,
            self.removed_workload_action_indices,
            self.kept_fault_rule_indices,
            self.removed_fault_rule_indices,
            self.kept_lifecycle_action_indices,
            self.removed_lifecycle_action_indices,
            self.kept_link_fault_action_indices,
            self.removed_link_fault_action_indices,
        )
        if actual != expected:
            raise FailureArtifactReplayMismatch("joint combined fault reduction changed")
        minimized = ReplicatedKVScenarioRunner(
            self.minimized_workload,
            self.minimized_faults,
            lifecycle=self.minimized_lifecycle,
            link_faults=self.minimized_link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if minimized.linearizability.linearizable:
            raise FailureArtifactReplayMismatch(
                "jointly minimized combined scenario became linearizable"
            )
        return result
