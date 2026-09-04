from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from .fault_schedule_minimizer import NonLinearizableFaultScheduleMinimizer
from .history_minimizer import NonLinearizableHistoryMinimizer
from .lifecycle import SeededLifecycleGenerator, SeededLifecycleSchedule
from .randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from .randomized_workload import SeededClientWorkloadGenerator, SeededClientWorkloadSchedule
from .scenario_runner import ReplicatedKVScenarioResult, ReplicatedKVScenarioRunner
from .simulator import TraceRecord
from .workload_minimizer import NonLinearizableClientWorkloadMinimizer


class FailureArtifactReplayMismatch(AssertionError):
    """Raised when a persisted failing scenario no longer replays exactly."""


@dataclass(frozen=True, slots=True)
class CampaignFailureArtifact:
    """Persistable exact-replay evidence for one non-linearizable scenario."""

    seed: int
    workload: SeededClientWorkloadSchedule
    minimized_workload: SeededClientWorkloadSchedule
    kept_workload_action_indices: tuple[int, ...]
    removed_workload_action_indices: tuple[int, ...]
    faults: SeededFaultSchedule
    minimized_faults: SeededFaultSchedule
    kept_fault_rule_indices: tuple[int, ...]
    removed_fault_rule_indices: tuple[int, ...]
    lifecycle: SeededLifecycleSchedule
    trace_json: str
    minimized_operation_ids: tuple[str, ...]
    removed_operation_ids: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        result: ReplicatedKVScenarioResult,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> CampaignFailureArtifact:
        if result.linearizability.linearizable:
            raise ValueError("failure artifact requires a non-linearizable result")
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(workload.seed)
        minimized_history = NonLinearizableHistoryMinimizer().minimize(result.history)

        if lifecycle.actions:
            minimized_faults = faults
            kept_fault_rules = tuple(range(len(faults.rules)))
            removed_fault_rules: tuple[int, ...] = ()
            minimized_workload = workload
            kept_workload_actions = tuple(range(len(workload.actions)))
            removed_workload_actions: tuple[int, ...] = ()
        else:
            fault_reduction = NonLinearizableFaultScheduleMinimizer().minimize(
                workload,
                faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            workload_reduction = NonLinearizableClientWorkloadMinimizer().minimize(
                workload,
                fault_reduction.schedule,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            minimized_faults = fault_reduction.schedule
            kept_fault_rules = fault_reduction.kept_original_indices
            removed_fault_rules = fault_reduction.removed_original_indices
            minimized_workload = workload_reduction.schedule
            kept_workload_actions = workload_reduction.kept_original_indices
            removed_workload_actions = workload_reduction.removed_original_indices

        return cls(
            seed=workload.seed,
            workload=workload,
            minimized_workload=minimized_workload,
            kept_workload_action_indices=kept_workload_actions,
            removed_workload_action_indices=removed_workload_actions,
            faults=faults,
            minimized_faults=minimized_faults,
            kept_fault_rule_indices=kept_fault_rules,
            removed_fault_rule_indices=removed_fault_rules,
            lifecycle=lifecycle,
            trace_json=_encode_trace(result.trace),
            minimized_operation_ids=minimized_history.operation_ids,
            removed_operation_ids=minimized_history.removed_operation_ids,
        )

    def to_json(self) -> str:
        payload = {
            "version": 4,
            "seed": self.seed,
            "workload": json.loads(self.workload.to_json()),
            "minimized_workload": json.loads(self.minimized_workload.to_json()),
            "kept_workload_action_indices": list(self.kept_workload_action_indices),
            "removed_workload_action_indices": list(self.removed_workload_action_indices),
            "faults": json.loads(self.faults.to_json()),
            "minimized_faults": json.loads(self.minimized_faults.to_json()),
            "kept_fault_rule_indices": list(self.kept_fault_rule_indices),
            "removed_fault_rule_indices": list(self.removed_fault_rule_indices),
            "lifecycle": json.loads(self.lifecycle.to_json()),
            "trace": json.loads(self.trace_json),
            "minimized_operation_ids": list(self.minimized_operation_ids),
            "removed_operation_ids": list(self.removed_operation_ids),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> CampaignFailureArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") not in (3, 4):
            raise ValueError("unsupported campaign failure artifact format")
        version = raw["version"]
        seed = raw.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        workload_raw = raw.get("workload")
        minimized_workload_raw = raw.get("minimized_workload")
        kept_workload_actions = raw.get("kept_workload_action_indices")
        removed_workload_actions = raw.get("removed_workload_action_indices")
        faults_raw = raw.get("faults")
        minimized_faults_raw = raw.get("minimized_faults")
        kept_fault_rules = raw.get("kept_fault_rule_indices")
        removed_fault_rules = raw.get("removed_fault_rule_indices")
        lifecycle_raw = raw.get("lifecycle") if version == 4 else None
        trace_raw = raw.get("trace")
        minimized = raw.get("minimized_operation_ids")
        removed = raw.get("removed_operation_ids")
        if not isinstance(workload_raw, dict) or not isinstance(faults_raw, dict):
            raise ValueError("artifact schedules must be objects")
        if not isinstance(minimized_workload_raw, dict):
            raise ValueError("artifact minimized workload schedule must be an object")
        if not _integer_list(kept_workload_actions) or not _integer_list(
            removed_workload_actions
        ):
            raise ValueError("artifact workload action index lists must contain integers")
        if not isinstance(minimized_faults_raw, dict):
            raise ValueError("artifact minimized fault schedule must be an object")
        if not _integer_list(kept_fault_rules) or not _integer_list(removed_fault_rules):
            raise ValueError("artifact fault rule index lists must contain integers")
        if version == 4 and not isinstance(lifecycle_raw, dict):
            raise ValueError("artifact lifecycle schedule must be an object")
        if not isinstance(trace_raw, list):
            raise ValueError("artifact trace must be a list")
        if not _string_list(minimized) or not _string_list(removed):
            raise ValueError("artifact operation id lists must contain strings")

        workload = SeededClientWorkloadSchedule.from_json(_canonical_json(workload_raw))
        minimized_workload = SeededClientWorkloadSchedule.from_json(
            _canonical_json(minimized_workload_raw)
        )
        faults = SeededFaultSchedule.from_json(_canonical_json(faults_raw))
        minimized_faults = SeededFaultSchedule.from_json(
            _canonical_json(minimized_faults_raw)
        )
        lifecycle = (
            SeededLifecycleSchedule.from_json(_canonical_json(lifecycle_raw))
            if lifecycle_raw is not None
            else SeededLifecycleSchedule.empty(seed)
        )
        if workload.seed != seed or minimized_workload.seed != seed:
            raise ValueError("artifact seed must match workload schedule seeds")
        if faults.seed != seed or minimized_faults.seed != seed:
            raise ValueError("artifact seed must match fault schedule seeds")
        if lifecycle.seed != seed:
            raise ValueError("artifact seed must match lifecycle schedule seed")
        _validate_index_partition(
            len(workload.actions),
            tuple(kept_workload_actions),
            tuple(removed_workload_actions),
            kind="workload action",
        )
        if minimized_workload.actions != tuple(
            workload.actions[index] for index in kept_workload_actions
        ):
            raise ValueError(
                "artifact minimized workload must match kept workload action indices"
            )
        _validate_index_partition(
            len(faults.rules),
            tuple(kept_fault_rules),
            tuple(removed_fault_rules),
            kind="fault rule",
        )
        if minimized_faults.rules != tuple(faults.rules[index] for index in kept_fault_rules):
            raise ValueError("artifact minimized faults must match kept fault rule indices")
        if lifecycle.actions:
            if tuple(kept_workload_actions) != tuple(range(len(workload.actions))):
                raise ValueError("lifecycle artifacts must preserve the full workload schedule")
            if tuple(kept_fault_rules) != tuple(range(len(faults.rules))):
                raise ValueError("lifecycle artifacts must preserve the full fault schedule")
        return cls(
            seed=seed,
            workload=workload,
            minimized_workload=minimized_workload,
            kept_workload_action_indices=tuple(kept_workload_actions),
            removed_workload_action_indices=tuple(removed_workload_actions),
            faults=faults,
            minimized_faults=minimized_faults,
            kept_fault_rule_indices=tuple(kept_fault_rules),
            removed_fault_rule_indices=tuple(removed_fault_rules),
            lifecycle=lifecycle,
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
            lifecycle=self.lifecycle,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch("persisted failure replay became linearizable")
        if _encode_trace(result.trace) != self.trace_json:
            raise FailureArtifactReplayMismatch("persisted failure trace did not replay exactly")
        minimized_history = NonLinearizableHistoryMinimizer().minimize(result.history)
        if minimized_history.operation_ids != self.minimized_operation_ids:
            raise FailureArtifactReplayMismatch("minimized failure witness changed during replay")
        if minimized_history.removed_operation_ids != self.removed_operation_ids:
            raise FailureArtifactReplayMismatch("removed operation set changed during replay")

        if self.lifecycle.actions:
            if self.minimized_faults != self.faults:
                raise FailureArtifactReplayMismatch(
                    "lifecycle artifact unexpectedly minimized its fault schedule"
                )
            if self.minimized_workload != self.workload:
                raise FailureArtifactReplayMismatch(
                    "lifecycle artifact unexpectedly minimized its workload schedule"
                )
        else:
            minimized_faults = NonLinearizableFaultScheduleMinimizer().minimize(
                self.workload,
                self.faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            if minimized_faults.schedule != self.minimized_faults:
                raise FailureArtifactReplayMismatch("minimized fault schedule changed during replay")
            if minimized_faults.kept_original_indices != self.kept_fault_rule_indices:
                raise FailureArtifactReplayMismatch("kept fault rule set changed during replay")
            if minimized_faults.removed_original_indices != self.removed_fault_rule_indices:
                raise FailureArtifactReplayMismatch("removed fault rule set changed during replay")

            minimized_workload = NonLinearizableClientWorkloadMinimizer().minimize(
                self.workload,
                self.minimized_faults,
                node_ids=node_ids,
                leader_id=leader_id,
            )
            if minimized_workload.schedule != self.minimized_workload:
                raise FailureArtifactReplayMismatch("minimized workload changed during replay")
            if minimized_workload.kept_original_indices != self.kept_workload_action_indices:
                raise FailureArtifactReplayMismatch("kept workload action set changed during replay")
            if minimized_workload.removed_original_indices != self.removed_workload_action_indices:
                raise FailureArtifactReplayMismatch("removed workload action set changed during replay")

        minimized_result = ReplicatedKVScenarioRunner(
            self.minimized_workload,
            self.minimized_faults,
            lifecycle=self.lifecycle,
            node_ids=node_ids,
            leader_id=leader_id,
        ).run()
        if minimized_result.linearizability.linearizable:
            raise FailureArtifactReplayMismatch(
                "minimized executable scenario became linearizable"
            )
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
    lifecycle_generator: SeededLifecycleGenerator | None = None
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
            lifecycle = (
                self.lifecycle_generator.compile(seed, len(workload.actions))
                if self.lifecycle_generator is not None
                else SeededLifecycleSchedule.empty(seed)
            )
            result = ReplicatedKVScenarioRunner(
                workload,
                faults,
                lifecycle=lifecycle,
                node_ids=self.node_ids,
                leader_id=self.leader_id,
            ).run()
            if not result.linearizability.linearizable:
                return ScenarioCampaignResult(
                    attempted_seeds=tuple(attempted),
                    failure=CampaignFailureArtifact.capture(
                        workload,
                        faults,
                        result,
                        lifecycle=lifecycle,
                        node_ids=self.node_ids,
                        leader_id=self.leader_id,
                    ),
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


def _integer_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


def _validate_index_partition(
    item_count: int,
    kept: tuple[int, ...],
    removed: tuple[int, ...],
    *,
    kind: str,
) -> None:
    all_indices = kept + removed
    if len(set(all_indices)) != len(all_indices):
        raise ValueError(f"artifact {kind} indices must be unique")
    if any(index < 0 or index >= item_count for index in all_indices):
        raise ValueError(f"artifact {kind} index out of range")
    if set(all_indices) != set(range(item_count)):
        raise ValueError(f"artifact {kind} indices must partition original items")
