from __future__ import annotations

from dataclasses import dataclass

from .combined_fault_artifact import CombinedFaultFailureArtifact
from .lifecycle import SeededLifecycleGenerator
from .link_fault_schedule import SeededLinkFaultGenerator
from .randomized_faults import FaultOpportunity, SeededFaultGenerator
from .randomized_workload import SeededClientWorkloadGenerator
from .scenario_runner import ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class CombinedFaultScenarioCampaignResult:
    attempted_seeds: tuple[int, ...]
    failure: CombinedFaultFailureArtifact | None


@dataclass(frozen=True, slots=True)
class SeededCombinedFaultScenarioCampaign:
    """Run bounded seeded KV scenarios with lifecycle and directional link faults."""

    workload_generator: SeededClientWorkloadGenerator
    fault_generator: SeededFaultGenerator
    lifecycle_generator: SeededLifecycleGenerator
    link_fault_generator: SeededLinkFaultGenerator
    fault_opportunities: tuple[FaultOpportunity, ...]
    operation_count: int
    node_ids: tuple[str, ...] = ("n1", "n2", "n3")
    leader_id: str = "n1"

    def __post_init__(self) -> None:
        if not isinstance(self.operation_count, int) or isinstance(self.operation_count, bool):
            raise ValueError("operation_count must be an integer")
        if self.operation_count < 0:
            raise ValueError("operation_count must be non-negative")

    def run(self, seeds: tuple[int, ...]) -> CombinedFaultScenarioCampaignResult:
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError("campaign seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("campaign seeds must be unique")

        attempted: list[int] = []
        for seed in seeds:
            attempted.append(seed)
            workload = self.workload_generator.compile(seed, self.operation_count)
            faults = self.fault_generator.compile(seed, self.fault_opportunities)
            lifecycle = self.lifecycle_generator.compile(seed, len(workload.actions))
            link_faults = self.link_fault_generator.compile(seed, len(workload.actions))
            result = ReplicatedKVScenarioRunner(
                workload,
                faults,
                lifecycle=lifecycle,
                link_faults=link_faults,
                node_ids=self.node_ids,
                leader_id=self.leader_id,
            ).run()
            if not result.linearizability.linearizable:
                return CombinedFaultScenarioCampaignResult(
                    attempted_seeds=tuple(attempted),
                    failure=CombinedFaultFailureArtifact.capture(
                        workload,
                        faults,
                        lifecycle,
                        link_faults,
                        result,
                        node_ids=self.node_ids,
                        leader_id=self.leader_id,
                    ),
                )
        return CombinedFaultScenarioCampaignResult(
            attempted_seeds=tuple(attempted),
            failure=None,
        )
