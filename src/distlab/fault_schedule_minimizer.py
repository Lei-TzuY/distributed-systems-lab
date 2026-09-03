from __future__ import annotations

from dataclasses import dataclass

from .randomized_faults import SeededFaultSchedule
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

        current = list(enumerate(faults.rules))
        removed: list[int] = []

        while True:
            changed = False
            for position in range(len(current)):
                candidate_entries = current[:position] + current[position + 1 :]
                candidate = SeededFaultSchedule(
                    seed=faults.seed,
                    rules=tuple(rule for _, rule in candidate_entries),
                )
                result = ReplicatedKVScenarioRunner(
                    workload,
                    candidate,
                    node_ids=node_ids,
                    leader_id=leader_id,
                ).run()
                if not result.linearizability.linearizable:
                    original_index, _ = current[position]
                    removed.append(original_index)
                    current = candidate_entries
                    changed = True
                    break
            if not changed:
                break

        kept = tuple(index for index, _ in current)
        minimized = SeededFaultSchedule(
            seed=faults.seed,
            rules=tuple(rule for _, rule in current),
        )
        return FaultScheduleMinimizationResult(
            schedule=minimized,
            kept_original_indices=kept,
            removed_original_indices=tuple(sorted(removed)),
        )
