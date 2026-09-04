from __future__ import annotations

from dataclasses import dataclass

from .lifecycle import SeededLifecycleSchedule
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner, ScenarioExecutionError


@dataclass(frozen=True, slots=True)
class LifecycleScheduleMinimizationResult:
    """A deterministic 1-minimal non-linearizable lifecycle schedule."""

    schedule: SeededLifecycleSchedule
    kept_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]


class NonLinearizableLifecycleScheduleMinimizer:
    """Delete lifecycle actions while preserving one non-linearizable failure.

    Workload actions, message faults, and lifecycle boundary indices are held
    fixed. Candidates are replayed through the deterministic scenario runner and
    never consult randomness. Invalid projections, such as deleting a crash
    while retaining its restart, are rejected rather than treated as failures.

    The result is 1-minimal with respect to lifecycle-action deletion: deleting
    any one remaining lifecycle action either makes the scenario linearizable
    or makes the explicit lifecycle schedule unexecutable.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        lifecycle: SeededLifecycleSchedule,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> LifecycleScheduleMinimizationResult:
        baseline = ReplicatedKVScenarioRunner(
            workload,
            faults,
            lifecycle=lifecycle,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if baseline.linearizability.linearizable:
            raise ValueError(
                "lifecycle schedule minimization requires a non-linearizable scenario"
            )

        current = list(enumerate(lifecycle.actions))
        removed: list[int] = []

        while True:
            changed = False
            for position in range(len(current)):
                candidate_entries = current[:position] + current[position + 1 :]
                candidate = SeededLifecycleSchedule(
                    seed=lifecycle.seed,
                    actions=tuple(action for _, action in candidate_entries),
                )
                try:
                    result = ReplicatedKVScenarioRunner(
                        workload,
                        faults,
                        lifecycle=candidate,
                        node_ids=node_ids,
                        leader_id=leader_id,
                    ).run()
                except ScenarioExecutionError:
                    continue
                if not result.linearizability.linearizable:
                    original_index, _ = current[position]
                    removed.append(original_index)
                    current = candidate_entries
                    changed = True
                    break
            if not changed:
                break

        kept = tuple(index for index, _ in current)
        minimized = SeededLifecycleSchedule(
            seed=lifecycle.seed,
            actions=tuple(action for _, action in current),
        )
        return LifecycleScheduleMinimizationResult(
            schedule=minimized,
            kept_original_indices=kept,
            removed_original_indices=tuple(sorted(removed)),
        )
