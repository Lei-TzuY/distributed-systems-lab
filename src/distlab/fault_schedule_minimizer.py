from __future__ import annotations

from dataclasses import dataclass

from ._deletion_minimizer import minimize_indexed_sequence
from .lifecycle import SeededLifecycleSchedule
from .link_fault_schedule import SeededLinkFaultSchedule
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner
from .simulator import FaultRule


@dataclass(frozen=True, slots=True)
class FaultScheduleMinimizationResult:
    """A deterministic 1-minimal non-linearizable fault schedule."""

    schedule: SeededFaultSchedule
    kept_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]


class NonLinearizableFaultScheduleMinimizer:
    """Delete message-fault rules while preserving a non-linearizable failure.

    Optional lifecycle and directional link-fault schedules are held fixed for
    every candidate. This lets higher-level combined-fault reduction minimize
    message faults without silently changing the rest of the failure witness.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        link_faults: SeededLinkFaultSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> FaultScheduleMinimizationResult:
        baseline = ReplicatedKVScenarioRunner(
            workload,
            faults,
            lifecycle=lifecycle,
            link_faults=link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if baseline.linearizability.linearizable:
            raise ValueError("fault schedule minimization requires a non-linearizable scenario")

        def preserves_failure(rules: tuple[FaultRule, ...]) -> bool:
            candidate = SeededFaultSchedule(seed=faults.seed, rules=rules)
            result = ReplicatedKVScenarioRunner(
                workload,
                candidate,
                lifecycle=lifecycle,
                link_faults=link_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            ).run()
            return not result.linearizability.linearizable

        current, removed = minimize_indexed_sequence(
            faults.rules,
            preserves_failure=preserves_failure,
        )
        kept = tuple(index for index, _ in current)
        minimized = SeededFaultSchedule(
            seed=faults.seed,
            rules=tuple(rule for _, rule in current),
        )
        return FaultScheduleMinimizationResult(
            schedule=minimized,
            kept_original_indices=kept,
            removed_original_indices=removed,
        )
