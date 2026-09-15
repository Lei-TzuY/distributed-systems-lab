from __future__ import annotations

import json
from dataclasses import dataclass

from .campaign import FailureArtifactReplayMismatch, _encode_trace
from .history_minimizer import NonLinearizableHistoryMinimizer
from .link_fault_schedule import SeededLinkFaultGenerator, SeededLinkFaultSchedule
from .link_fault_schedule_minimizer import NonLinearizableLinkFaultScheduleMinimizer
from .randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadGenerator, SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioResult, ReplicatedKVScenarioRunner


@dataclass(frozen=True, slots=True)
class LinkFaultCampaignFailureArtifact:
    """Exact-replay evidence for a failure caused under directional link faults."""

    seed: int
    workload: SeededClientWorkloadSchedule
    faults: SeededFaultSchedule
    link_faults: SeededLinkFaultSchedule
    minimized_link_faults: SeededLinkFaultSchedule
    kept_link_fault_action_indices: tuple[int, ...]
    removed_link_fault_action_indices: tuple[int, ...]
    trace_json: str
    minimized_operation_ids: tuple[str, ...]
    removed_operation_ids: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        link_faults: SeededLinkFaultSchedule,
        result: ReplicatedKVScenarioResult,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> LinkFaultCampaignFailureArtifact:
        if result.linearizability.linearizable:
            raise ValueError("failure artifact requires a non-linearizable result")
        if workload.seed != faults.seed or workload.seed != link_faults.seed:
            raise ValueError("failure artifact schedules must share one seed")
        reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
            workload,
            faults,
            link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        history = NonLinearizableHistoryMinimizer().minimize(result.history)
        return cls(
            seed=workload.seed,
            workload=workload,
            faults=faults,
            link_faults=link_faults,
            minimized_link_faults=reduction.schedule,
            kept_link_fault_action_indices=reduction.kept_original_indices,
            removed_link_fault_action_indices=reduction.removed_original_indices,
            trace_json=_encode_trace(result.trace),
            minimized_operation_ids=history.operation_ids,
            removed_operation_ids=history.removed_operation_ids,
        )

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "workload": json.loads(self.workload.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "link_faults": json.loads(self.link_faults.to_json()),
            "minimized_link_faults": json.loads(self.minimized_link_faults.to_json()),
            "kept_link_fault_action_indices": list(self.kept_link_fault_action_indices),
            "removed_link_fault_action_indices": list(self.removed_link_fault_action_indices),
            "trace": json.loads(self.trace_json),
            "minimized_operation_ids": list(self.minimized_operation_ids),
            "removed_operation_ids": list(self.removed_operation_ids),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> LinkFaultCampaignFailureArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported link fault campaign failure artifact format")
        try:
            seed = raw["seed"]
            workload = SeededClientWorkloadSchedule.from_json(
                json.dumps(raw["workload"], sort_keys=True, separators=(",", ":"))
            )
            faults = SeededFaultSchedule.from_json(
                json.dumps(raw["faults"], sort_keys=True, separators=(",", ":"))
            )
            link_faults = SeededLinkFaultSchedule.from_json(
                json.dumps(raw["link_faults"], sort_keys=True, separators=(",", ":"))
            )
            minimized = SeededLinkFaultSchedule.from_json(
                json.dumps(
                    raw["minimized_link_faults"], sort_keys=True, separators=(",", ":")
                )
            )
            kept = tuple(raw["kept_link_fault_action_indices"])
            removed = tuple(raw["removed_link_fault_action_indices"])
            trace_json = json.dumps(raw["trace"], sort_keys=True, separators=(",", ":"))
            operation_ids = tuple(raw["minimized_operation_ids"])
            removed_operation_ids = tuple(raw["removed_operation_ids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid link fault campaign failure artifact") from exc
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("artifact seed must be an integer")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in kept + removed):
            raise ValueError("artifact link fault indices must be integers")
        if sorted(kept + removed) != list(range(len(link_faults.actions))):
            raise ValueError("artifact link fault indices must partition the original schedule")
        if minimized.actions != tuple(link_faults.actions[index] for index in kept):
            raise ValueError("artifact minimized link faults must match kept indices")
        if any(not isinstance(item, str) for item in operation_ids + removed_operation_ids):
            raise ValueError("artifact operation ids must be strings")
        if workload.seed != seed or faults.seed != seed or link_faults.seed != seed:
            raise ValueError("artifact seed must match schedule seeds")
        if minimized.seed != seed:
            raise ValueError("artifact seed must match minimized link fault seed")
        return cls(
            seed=seed,
            workload=workload,
            faults=faults,
            link_faults=link_faults,
            minimized_link_faults=minimized,
            kept_link_fault_action_indices=kept,
            removed_link_fault_action_indices=removed,
            trace_json=trace_json,
            minimized_operation_ids=operation_ids,
            removed_operation_ids=removed_operation_ids,
        )

    def replay(
        self,
        *,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> ReplicatedKVScenarioResult:
        result = ReplicatedKVScenarioRunner(
            self.workload,
            self.faults,
            link_faults=self.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch("persisted failure replay became linearizable")
        if _encode_trace(result.trace) != self.trace_json:
            raise FailureArtifactReplayMismatch("persisted failure trace did not replay exactly")
        history = NonLinearizableHistoryMinimizer().minimize(result.history)
        if history.operation_ids != self.minimized_operation_ids:
            raise FailureArtifactReplayMismatch("minimized failure witness changed during replay")
        if history.removed_operation_ids != self.removed_operation_ids:
            raise FailureArtifactReplayMismatch("removed operation set changed during replay")
        reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
            self.workload,
            self.faults,
            self.link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        )
        if reduction.schedule != self.minimized_link_faults:
            raise FailureArtifactReplayMismatch(
                "minimized link fault schedule changed during replay"
            )
        if reduction.kept_original_indices != self.kept_link_fault_action_indices:
            raise FailureArtifactReplayMismatch("kept link fault action set changed during replay")
        if reduction.removed_original_indices != self.removed_link_fault_action_indices:
            raise FailureArtifactReplayMismatch(
                "removed link fault action set changed during replay"
            )
        minimized = ReplicatedKVScenarioRunner(
            self.workload,
            self.faults,
            link_faults=self.minimized_link_faults,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if minimized.linearizability.linearizable:
            raise FailureArtifactReplayMismatch("minimized executable scenario became linearizable")
        return result


@dataclass(frozen=True, slots=True)
class LinkFaultScenarioCampaignResult:
    attempted_seeds: tuple[int, ...]
    failure: LinkFaultCampaignFailureArtifact | None


@dataclass(frozen=True, slots=True)
class SeededLinkFaultScenarioCampaign:
    """Run bounded seeded KV scenarios with replayable directional link faults."""

    workload_generator: SeededClientWorkloadGenerator
    fault_generator: SeededFaultGenerator
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

    def run(self, seeds: tuple[int, ...]) -> LinkFaultScenarioCampaignResult:
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError("campaign seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("campaign seeds must be unique")
        attempted: list[int] = []
        for seed in seeds:
            attempted.append(seed)
            workload = self.workload_generator.compile(seed, self.operation_count)
            faults = self.fault_generator.compile(seed, self.fault_opportunities)
            link_faults = self.link_fault_generator.compile(seed, len(workload.actions))
            result = ReplicatedKVScenarioRunner(
                workload,
                faults,
                link_faults=link_faults,
                node_ids=self.node_ids,
                leader_id=self.leader_id,
            ).run()
            if not result.linearizability.linearizable:
                return LinkFaultScenarioCampaignResult(
                    attempted_seeds=tuple(attempted),
                    failure=LinkFaultCampaignFailureArtifact.capture(
                        workload,
                        faults,
                        link_faults,
                        result,
                        node_ids=self.node_ids,
                        leader_id=self.leader_id,
                    ),
                )
        return LinkFaultScenarioCampaignResult(attempted_seeds=tuple(attempted), failure=None)
