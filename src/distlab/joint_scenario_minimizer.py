from __future__ import annotations

from dataclasses import dataclass

from .combined_fault_minimizer import NonLinearizableCombinedFaultScheduleMinimizer
from .lifecycle import SeededLifecycleSchedule
from .link_fault_schedule import SeededLinkFaultSchedule
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner
from .workload_minimizer import NonLinearizableClientWorkloadMinimizer


@dataclass(frozen=True, slots=True)
class JointScenarioMinimizationResult:
    """Coordinate-wise minimal workload and explicit-fault witness."""

    workload: SeededClientWorkloadSchedule
    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    link_faults: SeededLinkFaultSchedule
    kept_workload_original_indices: tuple[int, ...]
    removed_workload_original_indices: tuple[int, ...]
    kept_fault_original_indices: tuple[int, ...]
    removed_fault_original_indices: tuple[int, ...]
    kept_lifecycle_original_indices: tuple[int, ...]
    removed_lifecycle_original_indices: tuple[int, ...]
    kept_link_fault_original_indices: tuple[int, ...]
    removed_link_fault_original_indices: tuple[int, ...]


class NonLinearizableJointScenarioMinimizer:
    """Reduce workload and all explicit fault dimensions to a fixed point."""

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        lifecycle: SeededLifecycleSchedule,
        link_faults: SeededLinkFaultSchedule,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> JointScenarioMinimizationResult:
        if len({workload.seed, faults.seed, lifecycle.seed, link_faults.seed}) != 1:
            raise ValueError("joint scenario schedules must share one seed")
        baseline = ReplicatedKVScenarioRunner(
            workload,
            faults,
            lifecycle=lifecycle,
            link_faults=link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if baseline.linearizability.linearizable:
            raise ValueError(
                "joint scenario minimization requires a non-linearizable scenario"
            )

        current_workload = workload
        current_faults = faults
        current_lifecycle = lifecycle
        current_link_faults = link_faults
        workload_projection = tuple(range(len(workload.actions)))
        fault_projection = tuple(range(len(faults.rules)))
        lifecycle_projection = tuple(range(len(lifecycle.actions)))
        link_projection = tuple(range(len(link_faults.actions)))

        while True:
            before = self._sizes(
                current_workload,
                current_faults,
                current_lifecycle,
                current_link_faults,
            )
            fault_reduction = NonLinearizableCombinedFaultScheduleMinimizer().minimize(
                current_workload,
                current_faults,
                current_lifecycle,
                current_link_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            fault_projection = tuple(
                fault_projection[index]
                for index in fault_reduction.kept_fault_original_indices
            )
            lifecycle_projection = tuple(
                lifecycle_projection[index]
                for index in fault_reduction.kept_lifecycle_original_indices
            )
            link_projection = tuple(
                link_projection[index]
                for index in fault_reduction.kept_link_fault_original_indices
            )
            current_faults = fault_reduction.faults
            current_lifecycle = fault_reduction.lifecycle
            current_link_faults = fault_reduction.link_faults

            workload_reduction = NonLinearizableClientWorkloadMinimizer().minimize(
                current_workload,
                current_faults,
                lifecycle=current_lifecycle,
                link_faults=current_link_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            workload_projection = tuple(
                workload_projection[index]
                for index in workload_reduction.kept_original_indices
            )
            current_workload = workload_reduction.schedule

            after = self._sizes(
                current_workload,
                current_faults,
                current_lifecycle,
                current_link_faults,
            )
            if after == before:
                break

        return JointScenarioMinimizationResult(
            workload=current_workload,
            faults=current_faults,
            lifecycle=current_lifecycle,
            link_faults=current_link_faults,
            kept_workload_original_indices=workload_projection,
            removed_workload_original_indices=self._removed(
                len(workload.actions), workload_projection
            ),
            kept_fault_original_indices=fault_projection,
            removed_fault_original_indices=self._removed(
                len(faults.rules), fault_projection
            ),
            kept_lifecycle_original_indices=lifecycle_projection,
            removed_lifecycle_original_indices=self._removed(
                len(lifecycle.actions), lifecycle_projection
            ),
            kept_link_fault_original_indices=link_projection,
            removed_link_fault_original_indices=self._removed(
                len(link_faults.actions), link_projection
            ),
        )

    @staticmethod
    def _sizes(workload, faults, lifecycle, link_faults) -> tuple[int, int, int, int]:
        return (
            len(workload.actions),
            len(faults.rules),
            len(lifecycle.actions),
            len(link_faults.actions),
        )

    @staticmethod
    def _removed(size: int, kept: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(index for index in range(size) if index not in kept)
