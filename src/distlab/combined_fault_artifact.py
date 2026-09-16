from __future__ import annotations

import json
from dataclasses import dataclass

from .campaign import FailureArtifactReplayMismatch, _encode_trace
from .history_minimizer import NonLinearizableHistoryMinimizer
from .lifecycle import SeededLifecycleSchedule
from .lifecycle_minimizer import NonLinearizableLifecycleScheduleMinimizer
from .link_fault_schedule import SeededLinkFaultSchedule
from .link_fault_schedule_minimizer import NonLinearizableLinkFaultScheduleMinimizer
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioResult, ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class CombinedFaultFailureArtifact:
    """Exact-replay evidence for failures combining lifecycle and directional link faults."""

    seed: int
    workload: SeededClientWorkloadSchedule
    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    link_faults: SeededLinkFaultSchedule
    minimized_lifecycle: SeededLifecycleSchedule
    kept_lifecycle_action_indices: tuple[int, ...]
    removed_lifecycle_action_indices: tuple[int, ...]
    minimized_link_faults: SeededLinkFaultSchedule
    kept_link_fault_action_indices: tuple[int, ...]
    removed_link_fault_action_indices: tuple[int, ...]
    trace_json: str
    minimized_operation_ids: tuple[str, ...]

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
    ) -> CombinedFaultFailureArtifact:
        if result.linearizability.linearizable:
            raise ValueError("failure artifact requires a non-linearizable result")
        seeds = {workload.seed, faults.seed, lifecycle.seed, link_faults.seed}
        if len(seeds) != 1:
            raise ValueError("failure artifact schedules must share one seed")
        lifecycle_reduction = NonLinearizableLifecycleScheduleMinimizer().minimize(
            workload,
            faults,
            lifecycle,
            link_faults=link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        link_reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
            workload,
            faults,
            link_faults,
            lifecycle=lifecycle,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        history = NonLinearizableHistoryMinimizer().minimize(result.history)
        return cls(
            seed=workload.seed,
            workload=workload,
            faults=faults,
            lifecycle=lifecycle,
            link_faults=link_faults,
            minimized_lifecycle=lifecycle_reduction.schedule,
            kept_lifecycle_action_indices=lifecycle_reduction.kept_original_indices,
            removed_lifecycle_action_indices=lifecycle_reduction.removed_original_indices,
            minimized_link_faults=link_reduction.schedule,
            kept_link_fault_action_indices=link_reduction.kept_original_indices,
            removed_link_fault_action_indices=link_reduction.removed_original_indices,
            trace_json=_encode_trace(result.trace),
            minimized_operation_ids=history.operation_ids,
        )

    def to_json(self) -> str:
        payload = {
            "version": 2,
            "seed": self.seed,
            "workload": json.loads(self.workload.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "lifecycle": json.loads(self.lifecycle.to_json()),
            "link_faults": json.loads(self.link_faults.to_json()),
            "minimized_lifecycle": json.loads(self.minimized_lifecycle.to_json()),
            "kept_lifecycle_action_indices": list(self.kept_lifecycle_action_indices),
            "removed_lifecycle_action_indices": list(self.removed_lifecycle_action_indices),
            "minimized_link_faults": json.loads(self.minimized_link_faults.to_json()),
            "kept_link_fault_action_indices": list(self.kept_link_fault_action_indices),
            "removed_link_fault_action_indices": list(self.removed_link_fault_action_indices),
            "trace": json.loads(self.trace_json),
            "minimized_operation_ids": list(self.minimized_operation_ids),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> CombinedFaultFailureArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 2:
            raise ValueError("unsupported combined fault failure artifact format")
        try:
            seed = raw["seed"]
            workload = SeededClientWorkloadSchedule.from_json(json.dumps(raw["workload"]))
            faults = SeededFaultSchedule.from_json(json.dumps(raw["faults"]))
            lifecycle = SeededLifecycleSchedule.from_json(json.dumps(raw["lifecycle"]))
            link_faults = SeededLinkFaultSchedule.from_json(json.dumps(raw["link_faults"]))
            minimized_lifecycle = SeededLifecycleSchedule.from_json(
                json.dumps(raw["minimized_lifecycle"])
            )
            kept_lifecycle = tuple(raw["kept_lifecycle_action_indices"])
            removed_lifecycle = tuple(raw["removed_lifecycle_action_indices"])
            minimized_link_faults = SeededLinkFaultSchedule.from_json(
                json.dumps(raw["minimized_link_faults"])
            )
            kept_link_faults = tuple(raw["kept_link_fault_action_indices"])
            removed_link_faults = tuple(raw["removed_link_fault_action_indices"])
            trace_json = json.dumps(raw["trace"], sort_keys=True, separators=(",", ":"))
            operation_ids = tuple(raw["minimized_operation_ids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid combined fault failure artifact") from exc
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("artifact seed must be an integer")
        schedule_seeds = {
            workload.seed,
            faults.seed,
            lifecycle.seed,
            link_faults.seed,
            minimized_lifecycle.seed,
            minimized_link_faults.seed,
        }
        if schedule_seeds != {seed}:
            raise ValueError("artifact seed must match all schedule seeds")
        cls._validate_projection(
            kept_lifecycle,
            removed_lifecycle,
            lifecycle.actions,
            minimized_lifecycle.actions,
            "lifecycle",
        )
        cls._validate_projection(
            kept_link_faults,
            removed_link_faults,
            link_faults.actions,
            minimized_link_faults.actions,
            "link fault",
        )
        if any(not isinstance(item, str) for item in operation_ids):
            raise ValueError("artifact operation ids must be strings")
        return cls(
            seed=seed,
            workload=workload,
            faults=faults,
            lifecycle=lifecycle,
            link_faults=link_faults,
            minimized_lifecycle=minimized_lifecycle,
            kept_lifecycle_action_indices=kept_lifecycle,
            removed_lifecycle_action_indices=removed_lifecycle,
            minimized_link_faults=minimized_link_faults,
            kept_link_fault_action_indices=kept_link_faults,
            removed_link_fault_action_indices=removed_link_faults,
            trace_json=trace_json,
            minimized_operation_ids=operation_ids,
        )

    @staticmethod
    def _validate_projection(kept, removed, original, minimized, label: str) -> None:
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in kept + removed
        ):
            raise ValueError(f"artifact {label} indices must be integers")
        if sorted(kept + removed) != list(range(len(original))):
            raise ValueError(
                f"artifact {label} indices must partition the original schedule"
            )
        if minimized != tuple(original[index] for index in kept):
            raise ValueError(f"artifact minimized {label} schedule must match kept indices")

    def replay(
        self,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> ReplicatedKVScenarioResult:
        result = ReplicatedKVScenarioRunner(
            self.workload,
            self.faults,
            lifecycle=self.lifecycle,
            link_faults=self.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch("persisted combined failure became linearizable")
        if _encode_trace(result.trace) != self.trace_json:
            raise FailureArtifactReplayMismatch("persisted combined failure trace changed")
        history = NonLinearizableHistoryMinimizer().minimize(result.history)
        if history.operation_ids != self.minimized_operation_ids:
            raise FailureArtifactReplayMismatch("minimized combined failure witness changed")
        lifecycle_reduction = NonLinearizableLifecycleScheduleMinimizer().minimize(
            self.workload,
            self.faults,
            self.lifecycle,
            link_faults=self.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        if lifecycle_reduction.schedule != self.minimized_lifecycle:
            raise FailureArtifactReplayMismatch("combined lifecycle reduction changed")
        if lifecycle_reduction.kept_original_indices != self.kept_lifecycle_action_indices:
            raise FailureArtifactReplayMismatch("combined kept lifecycle set changed")
        if (
            lifecycle_reduction.removed_original_indices
            != self.removed_lifecycle_action_indices
        ):
            raise FailureArtifactReplayMismatch("combined removed lifecycle set changed")
        link_reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
            self.workload,
            self.faults,
            self.link_faults,
            lifecycle=self.lifecycle,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        if link_reduction.schedule != self.minimized_link_faults:
            raise FailureArtifactReplayMismatch("combined link fault reduction changed")
        if link_reduction.kept_original_indices != self.kept_link_fault_action_indices:
            raise FailureArtifactReplayMismatch("combined kept link fault set changed")
        if link_reduction.removed_original_indices != self.removed_link_fault_action_indices:
            raise FailureArtifactReplayMismatch("combined removed link fault set changed")
        minimized_lifecycle_result = ReplicatedKVScenarioRunner(
            self.workload,
            self.faults,
            lifecycle=self.minimized_lifecycle,
            link_faults=self.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if minimized_lifecycle_result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch(
                "lifecycle-minimized combined scenario became linearizable"
            )
        minimized_link_result = ReplicatedKVScenarioRunner(
            self.workload,
            self.faults,
            lifecycle=self.lifecycle,
            link_faults=self.minimized_link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if minimized_link_result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch(
                "link-minimized combined scenario became linearizable"
            )
        return result
