from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from .history_minimizer import NonLinearizableHistoryMinimizer
from .randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadGenerator, SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioResult, ReplicatedKVScenarioRunner
from .simulator import TraceRecord


class FailureArtifactReplayMismatch(AssertionError):
    """Raised when a persisted failing scenario no longer replays exactly."""


@dataclass(frozen=True, slots=True)
class CampaignFailureArtifact:
    """Persistable exact-replay evidence for one non-linearizable scenario."""

    seed: int
    workload: SeededClientWorkloadSchedule
    faults: SeededFaultSchedule
    trace_json: str
    minimized_operation_ids: tuple[str, ...]
    removed_operation_ids: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        result: ReplicatedKVScenarioResult,
    ) -> CampaignFailureArtifact:
        if result.linearizability.linearizable:
            raise ValueError("failure artifact requires a non-linearizable result")
        minimized = NonLinearizableHistoryMinimizer().minimize(result.history)
        return cls(
            seed=workload.seed,
            workload=workload,
            faults=faults,
            trace_json=_encode_trace(result.trace),
            minimized_operation_ids=minimized.operation_ids,
            removed_operation_ids=minimized.removed_operation_ids,
        )

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "workload": json.loads(self.workload.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "trace": json.loads(self.trace_json),
            "minimized_operation_ids": list(self.minimized_operation_ids),
            "removed_operation_ids": list(self.removed_operation_ids),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> CampaignFailureArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported campaign failure artifact format")
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        workload_raw = raw.get("workload")
        faults_raw = raw.get("faults")
        trace_raw = raw.get("trace")
        minimized = raw.get("minimized_operation_ids")
        removed = raw.get("removed_operation_ids")
        if not isinstance(workload_raw, dict) or not isinstance(faults_raw, dict):
            raise ValueError("artifact schedules must be objects")
        if not isinstance(trace_raw, list):
            raise ValueError("artifact trace must be a list")
        if not _string_list(minimized) or not _string_list(removed):
            raise ValueError("artifact operation id lists must contain strings")

        workload = SeededClientWorkloadSchedule.from_json(_canonical_json(workload_raw))
        faults = SeededFaultSchedule.from_json(_canonical_json(faults_raw))
        if workload.seed != seed:
            raise ValueError("artifact seed must match workload seed")
        return cls(
            seed=seed,
            workload=workload,
            faults=faults,
            trace_json=_canonical_json(trace_raw),
            minimized_operation_ids=tuple(minimized),
            removed_operation_ids=tuple(removed),
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
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch("persisted failure replay became linearizable")
        if _encode_trace(result.trace) != self.trace_json:
            raise FailureArtifactReplayMismatch("persisted failure trace did not replay exactly")
        minimized = NonLinearizableHistoryMinimizer().minimize(result.history)
        if minimized.operation_ids != self.minimized_operation_ids:
            raise FailureArtifactReplayMismatch("minimized failure witness changed during replay")
        if minimized.removed_operation_ids != self.removed_operation_ids:
            raise FailureArtifactReplayMismatch("removed operation set changed during replay")
        return result


@dataclass(frozen=True, slots=True)
class ScenarioCampaignResult:
    attempted_seeds: tuple[int, ...]
    failure: CampaignFailureArtifact | None


@dataclass(frozen=True, slots=True)
class SeededScenarioCampaign:
    """Compile and run bounded seeded scenarios, stopping on the first failure."""

    workload_generator: SeededClientWorkloadGenerator
    fault_generator: SeededFaultGenerator
    fault_opportunities: tuple[FaultOpportunity, ...]
    operation_count: int
    node_ids: tuple[str, ...] = ("n1", "n2", "n3")
    leader_id: str = "n1"

    def __post_init__(self) -> None:
        if not isinstance(self.operation_count, int) or isinstance(self.operation_count, bool):
            raise ValueError("operation_count must be an integer")
        if self.operation_count < 0:
            raise ValueError("operation_count must be non-negative")

    def run(self, seeds: tuple[int, ...]) -> ScenarioCampaignResult:
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError("campaign seeds must be integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("campaign seeds must be unique")

        attempted: list[int] = []
        for seed in seeds:
            attempted.append(seed)
            workload = self.workload_generator.compile(seed, self.operation_count)
            faults = self.fault_generator.compile(seed, self.fault_opportunities)
            result = ReplicatedKVScenarioRunner(
                workload,
                faults,
                node_ids=self.node_ids,
                leader_id=self.leader_id,
            ).run()
            if not result.linearizability.linearizable:
                return ScenarioCampaignResult(
                    attempted_seeds=tuple(attempted),
                    failure=CampaignFailureArtifact.capture(workload, faults, result),
                )
        return ScenarioCampaignResult(attempted_seeds=tuple(attempted), failure=None)


def _encode_trace(trace: tuple[TraceRecord, ...]) -> str:
    encoded = [
        {
            "time": record.time,
            "kind": record.kind,
            "details": _stable_value(record.details),
        }
        for record in trace
    ]
    return _canonical_json(encoded)


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__qualname__,
            "fields": {
                field.name: _stable_value(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): _stable_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_stable_value(item) for item in value]
    raise TypeError(f"trace contains unsupported value {type(value).__qualname__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
