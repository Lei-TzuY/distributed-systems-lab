from __future__ import annotations

from dataclasses import dataclass

from .fault_schedule_minimizer import NonLinearizableFaultScheduleMinimizer
from .lifecycle import SeededLifecycleSchedule
from .lifecycle_minimizer import NonLinearizableLifecycleScheduleMinimizer
from .link_fault_schedule import SeededLinkFaultSchedule
from .link_fault_schedule_minimizer import NonLinearizableLinkFaultScheduleMinimizer
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class CombinedFaultScheduleMinimizationResult:
    """Deterministic coordinate-wise minimal explicit-fault witness."""

    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    link_faults: SeededLinkFaultSchedule
    kept_fault_original_indices: tuple[int, ...]
    removed_fault_original_indices: tuple[int, ...]
    kept_lifecycle_original_indices: tuple[int, ...]
    removed_lifecycle_original_indices: tuple[int, ...]
    kept_link_fault_original_indices: tuple[int, ...]
    removed_link_fault_original_indices: tuple[int, ...]


class NonLinearizableCombinedFaultScheduleMinimizer:
    """Reduce message, lifecycle, and directional link faults to a fixed point.

    Every reduction pass holds the other fault dimensions fixed and replays the
    deterministic scenario. Retained-index projections are carried back to the
    original schedules so the final witness remains exact reproducible evidence.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        lifecycle: SeededLifecycleSchedule,
        link_faults: SeededLinkFaultSchedule,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> CombinedFaultScheduleMinimizationResult:
        seeds = {workload.seed, faults.seed, lifecycle.seed, link_faults.seed}
        if len(seeds) != 1:
            raise ValueError("combined fault schedules must share one seed")
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
                "combined fault schedule minimization requires a non-linearizable scenario"
            )

        current_faults = faults
        current_lifecycle = lifecycle
        current_link_faults = link_faults
        fault_projection = tuple(range(len(faults.rules)))
        lifecycle_projection = tuple(range(len(lifecycle.actions)))
        link_projection = tuple(range(len(link_faults.actions)))

        while True:
            before = (
                len(current_faults.rules),
                len(current_lifecycle.actions),
                len(current_link_faults.actions),
            )

            fault_reduction = NonLinearizableFaultScheduleMinimizer().minimize(
                workload,
                current_faults,
                lifecycle=current_lifecycle,
                link_faults=current_link_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            fault_projection = tuple(
                fault_projection[index] for index in fault_reduction.kept_original_indices
            )
            current_faults = fault_reduction.schedule

            lifecycle_reduction = NonLinearizableLifecycleScheduleMinimizer().minimize(
                workload,
                current_faults,
                current_lifecycle,
                link_faults=current_link_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            lifecycle_projection = tuple(
                lifecycle_projection[index]
                for index in lifecycle_reduction.kept_original_indices
            )
            current_lifecycle = lifecycle_reduction.schedule

            link_reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
                workload,
                current_faults,
                current_link_faults,
                lifecycle=current_lifecycle,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            link_projection = tuple(
                link_projection[index] for index in link_reduction.kept_original_indices
            )
            current_link_faults = link_reduction.schedule

            after = (
                len(current_faults.rules),
                len(current_lifecycle.actions),
                len(current_link_faults.actions),
            )
            if after == before:
                break

        return CombinedFaultScheduleMinimizationResult(
            faults=current_faults,
            lifecycle=current_lifecycle,
            link_faults=current_link_faults,
            kept_fault_original_indices=fault_projection,
            removed_fault_original_indices=tuple(
                index for index in range(len(faults.rules)) if index not in fault_projection
            ),
            kept_lifecycle_original_indices=lifecycle_projection,
            removed_lifecycle_original_indices=tuple(
                index
                for index in range(len(lifecycle.actions))
                if index not in lifecycle_projection
            ),
            kept_link_fault_original_indices=link_projection,
            removed_link_fault_original_indices=tuple(
                index
                for index in range(len(link_faults.actions))
                if index not in link_projection
            ),
        )
