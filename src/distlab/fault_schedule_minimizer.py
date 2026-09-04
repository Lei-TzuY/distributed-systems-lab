from __future__ import annotations

from dataclasses import dataclass

from ._deletion_minimizer import minimize_indexed_sequence
from .randomized_faults import FaultRule, SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class FaultScheduleMinimizationResult:
    """A deterministic 1-minimal non-linearizable fault schedule."""

    schedule: SeededFaultSchedule
    kept_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]


class NonLinearizableFaultScheduleMinimizer:
    """Delete fault rules while preserving a non-linearizable scenario failure.

    The minimizer is deliberately narrow: it keeps the client workload fixed and
    only removes explicit fault rules. Candidates are replayed through the same
    deterministic scenario runner, so no randomness is consulted during reduction.
    The result is 1-minimal with respect to rule deletion: removing any one
    remaining rule makes the scenario linearizable.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> FaultScheduleMinimizationResult:
        baseline = ReplicatedKVScenarioRunner(
            workload,
            faults,
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
