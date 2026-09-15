from __future__ import annotations

from dataclasses import dataclass

from ._deletion_minimizer import minimize_indexed_sequence
from .lifecycle import SeededLifecycleSchedule
from .link_fault_schedule import LinkFaultAction, SeededLinkFaultSchedule
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioRunner, ScenarioExecutionError


@dataclass(frozen=True, slots=True)
class LinkFaultScheduleMinimizationResult:
    """A deterministic 1-minimal non-linearizable link-fault schedule."""

    schedule: SeededLinkFaultSchedule
    kept_original_indices: tuple[int, ...]
    removed_original_indices: tuple[int, ...]


class NonLinearizableLinkFaultScheduleMinimizer:
    """Delete directional link faults while preserving a non-linearizable failure.

    Workload actions, message faults, lifecycle actions, and link-fault boundary
    indices are held fixed. Every candidate is replayed through the deterministic
    scenario runner; randomness is never regenerated during reduction.
    """

    def minimize(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        link_faults: SeededLinkFaultSchedule,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> LinkFaultScheduleMinimizationResult:
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(workload.seed)
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
                "link fault schedule minimization requires a non-linearizable scenario"
            )

        def preserves_failure(actions: tuple[LinkFaultAction, ...]) -> bool:
            candidate = SeededLinkFaultSchedule(seed=link_faults.seed, actions=actions)
            try:
                result = ReplicatedKVScenarioRunner(
                    workload,
                    faults,
                    lifecycle=lifecycle,
                    link_faults=candidate,
                    node_ids=node_ids,
                    leader_id=leader_id,
                ).run()
            except (ScenarioExecutionError, ValueError):
                return False
            return not result.linearizability.linearizable

        current, removed = minimize_indexed_sequence(
            link_faults.actions,
            preserves_failure=preserves_failure,
        )
        kept = tuple(index for index, _ in current)
        minimized = SeededLinkFaultSchedule(
            seed=link_faults.seed,
            actions=tuple(action for _, action in current),
        )
        return LinkFaultScheduleMinimizationResult(
            schedule=minimized,
            kept_original_indices=kept,
            removed_original_indices=removed,
        )
