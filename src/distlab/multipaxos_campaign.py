from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import StrEnum

from .lifecycle import NodeLifecycleKind, SeededLifecycleGenerator, SeededLifecycleSchedule
from .multipaxos import MultiPaxosCluster
from .paxos import PaxosError, PaxosSafetyViolation, ProposalNumber
from .paxos_log import SlotLearnedDecision
from .randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from .replay_codec import canonical_json, encode_trace
from .simulator import Simulator, TraceRecord


class MultiPaxosScenarioExecutionError(RuntimeError):
    """Raised when one explicit Multi-Paxos replay schedule is inconsistent."""


class MultiPaxosReplayMismatch(AssertionError):
    """Raised when persisted Multi-Paxos evidence no longer replays exactly."""


class MultiPaxosActionKind(StrEnum):
    PREPARE = "prepare"
    PROPOSE = "propose"


class MultiPaxosScenarioOutcome(StrEnum):
    PROGRESS = "progress"
    INCOMPLETE = "incomplete"
    SAFETY_VIOLATION = "safety_violation"


@dataclass(frozen=True, slots=True)
class MultiPaxosAction:
    action_id: str
    kind: MultiPaxosActionKind
    node_id: str
    events_after: int = 0
    round_number: int | None = None
    slot: int | None = None
    value: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MultiPaxosActionKind):
            raise ValueError("kind must be a MultiPaxosActionKind")
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if not isinstance(self.events_after, int) or isinstance(self.events_after, bool):
            raise ValueError("events_after must be an integer")
        if self.events_after < 0:
            raise ValueError("events_after must be non-negative")

        if self.kind is MultiPaxosActionKind.PREPARE:
            if self.slot is not None or self.value is not None:
                raise ValueError("prepare action cannot include slot/value")
            if self.round_number is not None:
                if not isinstance(self.round_number, int) or isinstance(
                    self.round_number, bool
                ):
                    raise ValueError("round_number must be an integer or None")
                if self.round_number <= 0:
                    raise ValueError("round_number must be positive when specified")
            return

        if self.round_number is not None:
            raise ValueError("propose action cannot include round_number")
        if not isinstance(self.slot, int) or isinstance(self.slot, bool) or self.slot <= 0:
            raise ValueError("propose action slot must be a positive integer")
        if not isinstance(self.value, str):
            raise ValueError("propose action value must be a string")


@dataclass(frozen=True, slots=True)
class SeededMultiPaxosActionSchedule:
    seed: int
    actions: tuple[MultiPaxosAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Multi-Paxos action ids must be unique")

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": [
                {
                    "action_id": action.action_id,
                    "kind": action.kind.value,
                    "node_id": action.node_id,
                    "events_after": action.events_after,
                    "round_number": action.round_number,
                    "slot": action.slot,
                    "value": action.value,
                }
                for action in self.actions
            ],
        }
        return canonical_json(payload)

    @classmethod
    def from_json(cls, encoded: str) -> SeededMultiPaxosActionSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded Multi-Paxos action schedule format")
        seed = raw.get("seed")
        actions = raw.get("actions")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")

        decoded: list[MultiPaxosAction] = []
        for item in actions:
            if not isinstance(item, dict):
                raise ValueError("each Multi-Paxos action must be an object")
            try:
                decoded.append(
                    MultiPaxosAction(
                        action_id=item["action_id"],
                        kind=MultiPaxosActionKind(item["kind"]),
                        node_id=item["node_id"],
                        events_after=item["events_after"],
                        round_number=item.get("round_number"),
                        slot=item.get("slot"),
                        value=item.get("value"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid Multi-Paxos action") from exc
        return cls(seed=seed, actions=tuple(decoded))


@dataclass(frozen=True, slots=True)
class SeededMultiPaxosActionGenerator:
    node_ids: tuple[str, ...]
    values: tuple[str, ...] = ("alpha", "beta", "gamma")
    leadership_change_rate: float = 0.25
    max_events_after: int = 6

    def __post_init__(self) -> None:
        if not self.node_ids or any(not node_id for node_id in self.node_ids):
            raise ValueError("node_ids must contain non-empty strings")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if not self.values or any(not isinstance(value, str) for value in self.values):
            raise ValueError("values must contain strings")
        if not 0 <= self.leadership_change_rate <= 1:
            raise ValueError("leadership_change_rate must be between 0 and 1")
        if not isinstance(self.max_events_after, int) or isinstance(
            self.max_events_after, bool
        ):
            raise ValueError("max_events_after must be an integer")
        if self.max_events_after < 0:
            raise ValueError("max_events_after must be non-negative")

    def compile(self, seed: int, action_count: int) -> SeededMultiPaxosActionSchedule:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(action_count, int) or isinstance(action_count, bool):
            raise ValueError("action_count must be an integer")
        if action_count <= 0:
            raise ValueError("action_count must be positive")

        rng = random.Random(seed)
        nodes = tuple(sorted(self.node_ids))
        leader = rng.choice(nodes)
        next_slot = 1
        actions: list[MultiPaxosAction] = []

        for index in range(action_count):
            prepare = index == 0 or rng.random() < self.leadership_change_rate
            if prepare:
                leader = rng.choice(nodes)
                minimum = min(self.max_events_after, 2 * (len(nodes) - 1))
                events_after = rng.randint(minimum, self.max_events_after)
                actions.append(
                    MultiPaxosAction(
                        action_id=f"multipaxos-action-{index + 1:06d}",
                        kind=MultiPaxosActionKind.PREPARE,
                        node_id=leader,
                        events_after=events_after,
                    )
                )
                continue

            actions.append(
                MultiPaxosAction(
                    action_id=f"multipaxos-action-{index + 1:06d}",
                    kind=MultiPaxosActionKind.PROPOSE,
                    node_id=leader,
                    slot=next_slot,
                    value=rng.choice(self.values),
                    events_after=rng.randint(0, self.max_events_after),
                )
            )
            next_slot += 1

        return SeededMultiPaxosActionSchedule(seed=seed, actions=tuple(actions))


@dataclass(frozen=True, slots=True)
class MultiPaxosScenarioResult:
    outcome: MultiPaxosScenarioOutcome
    chosen: tuple[SlotLearnedDecision, ...]
    violation: str | None
    trace: tuple[TraceRecord, ...]


class MultiPaxosScenarioRunner:
    """Replay explicit Multi-Paxos actions, faults, and lifecycle schedules."""

    def __init__(
        self,
        actions: SeededMultiPaxosActionSchedule,
        faults: SeededFaultSchedule,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        final_event_budget: int = 128,
    ) -> None:
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        unknown_nodes = sorted(
            {action.node_id for action in actions.actions} - set(node_ids)
        )
        if unknown_nodes:
            raise ValueError(
                f"Multi-Paxos action schedule references unknown nodes: {unknown_nodes!r}"
            )
        if actions.seed != faults.seed:
            raise ValueError("action and fault schedules must use the same seed")
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(actions.seed)
        if lifecycle.seed != actions.seed:
            raise ValueError("action and lifecycle schedules must use the same seed")
        unknown_lifecycle = sorted(
            {action.node_id for action in lifecycle.actions} - set(node_ids)
        )
        if unknown_lifecycle:
            raise ValueError(
                "Multi-Paxos lifecycle schedule references unknown nodes: "
                f"{unknown_lifecycle!r}"
            )
        if any(
            action.before_action_index > len(actions.actions)
            for action in lifecycle.actions
        ):
            raise ValueError(
                "Multi-Paxos lifecycle action references an action boundary out of range"
            )
        if not isinstance(final_event_budget, int) or isinstance(final_event_budget, bool):
            raise ValueError("final_event_budget must be an integer")
        if final_event_budget < 0:
            raise ValueError("final_event_budget must be non-negative")

        self.actions = actions
        self.faults = faults
        self.lifecycle = lifecycle
        self.node_ids = node_ids
        self.final_event_budget = final_event_budget

    def run(self) -> MultiPaxosScenarioResult:
        sim = Simulator(fault_plan=self.faults.to_fault_plan())
        cluster = MultiPaxosCluster(sim, self.node_ids)
        lifecycle_position = 0
        violation: str | None = None

        try:
            for boundary, action in enumerate(self.actions.actions):
                lifecycle_position = self._apply_lifecycle_boundary(
                    boundary,
                    lifecycle_position,
                    sim,
                    cluster,
                )
                self._execute_action(action, sim, cluster)
                if action.events_after:
                    sim.run(max_events=action.events_after)
                cluster.assert_safety()

            lifecycle_position = self._apply_lifecycle_boundary(
                len(self.actions.actions),
                lifecycle_position,
                sim,
                cluster,
            )
            if lifecycle_position != len(self.lifecycle.actions):
                raise AssertionError("Multi-Paxos lifecycle schedule was not fully consumed")
            sim.run(max_events=self.final_event_budget)
            cluster.assert_safety()
        except PaxosSafetyViolation as exc:
            violation = str(exc)

        chosen = cluster.chosen_slots()
        if violation is not None:
            outcome = MultiPaxosScenarioOutcome.SAFETY_VIOLATION
        elif chosen:
            outcome = MultiPaxosScenarioOutcome.PROGRESS
        else:
            outcome = MultiPaxosScenarioOutcome.INCOMPLETE
        return MultiPaxosScenarioResult(
            outcome=outcome,
            chosen=chosen,
            violation=violation,
            trace=tuple(sim.trace),
        )

    def _execute_action(
        self,
        action: MultiPaxosAction,
        sim: Simulator,
        cluster: MultiPaxosCluster,
    ) -> None:
        if not sim.is_alive(action.node_id):
            sim._record(
                "scenario-multipaxos-action-skipped",
                action_id=action.action_id,
                action=action.kind.value,
                node=action.node_id,
                reason="node-crashed",
            )
            return

        node = cluster.node(action.node_id)
        if action.kind is MultiPaxosActionKind.PREPARE:
            try:
                ballot = node.prepare_leadership(round_number=action.round_number)
            except (PaxosError, ValueError) as exc:
                sim._record(
                    "scenario-multipaxos-prepare-rejected",
                    action_id=action.action_id,
                    node=action.node_id,
                    detail=str(exc),
                )
            else:
                sim._record(
                    "scenario-multipaxos-prepare",
                    action_id=action.action_id,
                    node=action.node_id,
                    ballot=ballot,
                    events_after=action.events_after,
                )
            return

        assert action.slot is not None
        assert action.value is not None
        try:
            node.propose(action.slot, action.value)
        except PaxosError as exc:
            sim._record(
                "scenario-multipaxos-propose-rejected",
                action_id=action.action_id,
                node=action.node_id,
                slot=action.slot,
                value=action.value,
                detail=str(exc),
            )
        else:
            sim._record(
                "scenario-multipaxos-propose",
                action_id=action.action_id,
                node=action.node_id,
                slot=action.slot,
                value=action.value,
                events_after=action.events_after,
            )

    def _apply_lifecycle_boundary(
        self,
        boundary: int,
        position: int,
        sim: Simulator,
        cluster: MultiPaxosCluster,
    ) -> int:
        while position < len(self.lifecycle.actions):
            action = self.lifecycle.actions[position]
            if action.before_action_index != boundary:
                break
            if action.kind is NodeLifecycleKind.CRASH:
                if not sim.is_alive(action.node_id):
                    raise MultiPaxosScenarioExecutionError(
                        f"cannot crash already crashed node {action.node_id!r}"
                    )
                sim.crash(action.node_id)
            else:
                if sim.is_alive(action.node_id):
                    raise MultiPaxosScenarioExecutionError(
                        f"cannot restart live node {action.node_id!r}"
                    )
                sim.restart(action.node_id)
            sim._record(
                "scenario-multipaxos-lifecycle",
                action_id=action.action_id,
                node=action.node_id,
                action=action.kind.value,
                before_action_index=boundary,
            )
            cluster.assert_safety()
            position += 1
        return position


_MULTIPAXOS_PAYLOAD_TYPES = (
    "LeaderAccept",
    "LeaderAccepted",
    "LeaderPrepare",
    "LeaderPromise",
)


def multipaxos_fault_opportunities(
    node_ids: tuple[str, ...],
    *,
    max_message_ordinal: int = 12,
) -> tuple[FaultOpportunity, ...]:
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("node_ids must be non-empty and unique")
    if not isinstance(max_message_ordinal, int) or isinstance(max_message_ordinal, bool):
        raise ValueError("max_message_ordinal must be an integer")
    if max_message_ordinal <= 0:
        raise ValueError("max_message_ordinal must be positive")
    return tuple(
        FaultOpportunity(src, dst, ordinal, payload_type=payload_type)
        for src in sorted(node_ids)
        for dst in sorted(node_ids)
        if src != dst
        for ordinal in range(1, max_message_ordinal + 1)
        for payload_type in _MULTIPAXOS_PAYLOAD_TYPES
    )


@dataclass(frozen=True, slots=True)
class MultiPaxosTrialArtifact:
    seed: int
    actions: SeededMultiPaxosActionSchedule
    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    node_ids: tuple[str, ...]
    final_event_budget: int
    outcome: MultiPaxosScenarioOutcome
    chosen: tuple[SlotLearnedDecision, ...]
    violation: str | None
    trace_json: str

    @classmethod
    def capture(
        cls,
        actions: SeededMultiPaxosActionSchedule,
        faults: SeededFaultSchedule,
        result: MultiPaxosScenarioResult,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        final_event_budget: int = 128,
    ) -> MultiPaxosTrialArtifact:
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(actions.seed)
        if actions.seed != faults.seed or actions.seed != lifecycle.seed:
            raise ValueError("Multi-Paxos trial schedules must use the same seed")
        return cls(
            seed=actions.seed,
            actions=actions,
            faults=faults,
            lifecycle=lifecycle,
            node_ids=node_ids,
            final_event_budget=final_event_budget,
            outcome=result.outcome,
            chosen=result.chosen,
            violation=result.violation,
            trace_json=encode_trace(result.trace),
        )

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": json.loads(self.actions.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "lifecycle": json.loads(self.lifecycle.to_json()),
            "node_ids": list(self.node_ids),
            "final_event_budget": self.final_event_budget,
            "outcome": self.outcome.value,
            "chosen": [
                {
                    "slot": decision.slot,
                    "proposal": {
                        "round": decision.proposal.round,
                        "proposer_id": decision.proposal.proposer_id,
                    },
                    "value": decision.value,
                }
                for decision in self.chosen
            ],
            "violation": self.violation,
            "trace": json.loads(self.trace_json),
        }
        return canonical_json(payload)

    @classmethod
    def from_json(cls, encoded: str) -> MultiPaxosTrialArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported Multi-Paxos trial artifact format")
        seed = raw.get("seed")
        node_ids = raw.get("node_ids")
        final_event_budget = raw.get("final_event_budget")
        chosen_raw = raw.get("chosen")
        trace = raw.get("trace")
        violation = raw.get("violation")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(node_ids, list)
            or not node_ids
            or not all(isinstance(node_id, str) and node_id for node_id in node_ids)
        ):
            raise ValueError("node_ids must contain non-empty strings")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique")
        if not isinstance(final_event_budget, int) or isinstance(final_event_budget, bool):
            raise ValueError("final_event_budget must be an integer")
        if final_event_budget < 0:
            raise ValueError("final_event_budget must be non-negative")
        if not isinstance(chosen_raw, list):
            raise ValueError("chosen must be a list")
        if not isinstance(trace, list):
            raise ValueError("trace must be a list")
        if violation is not None and not isinstance(violation, str):
            raise ValueError("violation must be a string or null")

        try:
            outcome = MultiPaxosScenarioOutcome(raw["outcome"])
            actions = SeededMultiPaxosActionSchedule.from_json(
                canonical_json(raw["actions"])
            )
            faults = SeededFaultSchedule.from_json(canonical_json(raw["faults"]))
            lifecycle = SeededLifecycleSchedule.from_json(
                canonical_json(raw["lifecycle"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Multi-Paxos trial artifact") from exc
        if actions.seed != seed or faults.seed != seed or lifecycle.seed != seed:
            raise ValueError("artifact seed must match all Multi-Paxos schedules")

        chosen: list[SlotLearnedDecision] = []
        for item in chosen_raw:
            if not isinstance(item, dict):
                raise ValueError("chosen entries must be objects")
            proposal_raw = item.get("proposal")
            if not isinstance(proposal_raw, dict):
                raise ValueError("chosen proposal must be an object")
            try:
                slot = item["slot"]
                proposal = ProposalNumber(
                    proposal_raw["round"],
                    proposal_raw["proposer_id"],
                )
                value = item["value"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid chosen Multi-Paxos decision") from exc
            if not isinstance(slot, int) or isinstance(slot, bool) or slot <= 0:
                raise ValueError("chosen slot must be a positive integer")
            if not isinstance(value, str):
                raise ValueError("chosen campaign value must be a string")
            chosen.append(SlotLearnedDecision(slot, proposal, value))
        chosen_slots = tuple(decision.slot for decision in chosen)
        if chosen_slots != tuple(sorted(chosen_slots)):
            raise ValueError("chosen slots must be sorted")
        if len(set(chosen_slots)) != len(chosen_slots):
            raise ValueError("chosen slots must be unique")

        return cls(
            seed=seed,
            actions=actions,
            faults=faults,
            lifecycle=lifecycle,
            node_ids=tuple(node_ids),
            final_event_budget=final_event_budget,
            outcome=outcome,
            chosen=tuple(chosen),
            violation=violation,
            trace_json=canonical_json(trace),
        )

    def replay(self) -> MultiPaxosScenarioResult:
        result = MultiPaxosScenarioRunner(
            self.actions,
            self.faults,
            lifecycle=self.lifecycle,
            node_ids=self.node_ids,
            final_event_budget=self.final_event_budget,
        ).run()
        if result.outcome is not self.outcome:
            raise MultiPaxosReplayMismatch(
                f"Multi-Paxos outcome changed from {self.outcome.value!r} "
                f"to {result.outcome.value!r}"
            )
        if result.chosen != self.chosen:
            raise MultiPaxosReplayMismatch(
                "Multi-Paxos chosen evidence changed during replay"
            )
        if result.violation != self.violation:
            raise MultiPaxosReplayMismatch(
                "Multi-Paxos safety violation changed during replay"
            )
        if encode_trace(result.trace) != self.trace_json:
            raise MultiPaxosReplayMismatch("Multi-Paxos trace did not replay exactly")
        return result


@dataclass(frozen=True, slots=True)
class MultiPaxosCampaignResult:
    attempted_seeds: tuple[int, ...]
    trials: tuple[MultiPaxosTrialArtifact, ...]
    safety_failure: MultiPaxosTrialArtifact | None


@dataclass(frozen=True, slots=True)
class SeededMultiPaxosCampaign:
    action_generator: SeededMultiPaxosActionGenerator
    fault_generator: SeededFaultGenerator
    action_count: int
    lifecycle_generator: SeededLifecycleGenerator | None = None
    node_ids: tuple[str, ...] = ("n1", "n2", "n3")
    max_message_ordinal: int = 12
    final_event_budget: int = 128

    def __post_init__(self) -> None:
        if self.action_generator.node_ids != self.node_ids:
            raise ValueError("action generator node_ids must match campaign node_ids")
        if (
            self.lifecycle_generator is not None
            and tuple(sorted(self.lifecycle_generator.nodes)) != tuple(sorted(self.node_ids))
        ):
            raise ValueError("lifecycle generator nodes must match campaign node_ids")
        if not isinstance(self.action_count, int) or isinstance(self.action_count, bool):
            raise ValueError("action_count must be an integer")
        if self.action_count <= 0:
            raise ValueError("action_count must be positive")
        multipaxos_fault_opportunities(
            self.node_ids,
            max_message_ordinal=self.max_message_ordinal,
        )
        if not isinstance(self.final_event_budget, int) or isinstance(
            self.final_event_budget, bool
        ):
            raise ValueError("final_event_budget must be an integer")
        if self.final_event_budget < 0:
            raise ValueError("final_event_budget must be non-negative")

    def run(self, seeds: tuple[int, ...]) -> MultiPaxosCampaignResult:
        attempted: list[int] = []
        trials: list[MultiPaxosTrialArtifact] = []
        opportunities = multipaxos_fault_opportunities(
            self.node_ids,
            max_message_ordinal=self.max_message_ordinal,
        )

        for seed in seeds:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("campaign seeds must be integers")
            actions = self.action_generator.compile(seed, self.action_count)
            faults = self.fault_generator.compile(seed, opportunities)
            lifecycle = (
                self.lifecycle_generator.compile(seed, len(actions.actions))
                if self.lifecycle_generator is not None
                else SeededLifecycleSchedule.empty(seed)
            )
            result = MultiPaxosScenarioRunner(
                actions,
                faults,
                lifecycle=lifecycle,
                node_ids=self.node_ids,
                final_event_budget=self.final_event_budget,
            ).run()
            artifact = MultiPaxosTrialArtifact.capture(
                actions,
                faults,
                result,
                lifecycle=lifecycle,
                node_ids=self.node_ids,
                final_event_budget=self.final_event_budget,
            )
            attempted.append(seed)
            trials.append(artifact)
            if result.outcome is MultiPaxosScenarioOutcome.SAFETY_VIOLATION:
                return MultiPaxosCampaignResult(
                    attempted_seeds=tuple(attempted),
                    trials=tuple(trials),
                    safety_failure=artifact,
                )

        return MultiPaxosCampaignResult(
            attempted_seeds=tuple(attempted),
            trials=tuple(trials),
            safety_failure=None,
        )
