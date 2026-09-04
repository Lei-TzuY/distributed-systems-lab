from __future__ import annotations

from dataclasses import dataclass

from ._deletion_minimizer import minimize_indexed_sequence
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import ClientAction, SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class ClientWorkloadMinimizationResult:
    """A deterministic 1-minimal non-linearizable client workload."""

    schedule: SeededClientWorkloadSchedule
    kept_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]


class NonLinearizableClientWorkloadMinimizer:
    """Delete client actions while preserving a non-linearizable scenario failure.

    The fault schedule is held fixed. Candidates are replayed through the same
    deterministic scenario runner, and reduction never consults randomness. The
    result is 1-minimal with respect to action deletion: deleting any one
    remaining client action makes the scenario linearizable.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> ClientWorkloadMinimizationResult:
        baseline = ReplicatedKVScenarioRunner(
            workload,
            faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if baseline.linearizability.linearizable:
            raise ValueError("client workload minimization requires a non-linearizable scenario")

        def preserves_failure(actions: tuple[ClientAction, ...]) -> bool:
            candidate = SeededClientWorkloadSchedule(seed=workload.seed, actions=actions)
            result = ReplicatedKVScenarioRunner(
                candidate,
                faults,
                node_ids=node_ids,
                leader_id=leader_id,
            ).run()
            return not result.linearizability.linearizable

        current, removed = minimize_indexed_sequence(
            workload.actions,
            preserves_failure=preserves_failure,
        )
        kept = tuple(index for index, _ in current)
        minimized = SeededClientWorkloadSchedule(
            seed=workload.seed,
            actions=tuple(action for _, action in current),
        )
        return ClientWorkloadMinimizationResult(
            schedule=minimized,
            kept_original_indices=kept,
            removed_original_indices=removed,
        )
