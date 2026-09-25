from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import StrEnum

from .lifecycle import NodeLifecycleKind, SeededLifecycleGenerator, SeededLifecycleSchedule
from .paxos import (
    LearnedDecision,
    PaxosCluster,
    PaxosError,
    PaxosProposalActive,
    PaxosSafetyHarness,
    PaxosSafetyViolation,
    ProposalNumber,
)
from .randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from .replay_codec import canonical_json, encode_trace
from .simulator import Simulator, TraceRecord


class PaxosScenarioExecutionError(RuntimeError):
    """Raised when an explicit Paxos replay schedule is internally inconsistent."""


class PaxosReplayMismatch(AssertionError):
    """Raised when a persisted Paxos trial no longer replays exactly."""


class PaxosScenarioOutcome(StrEnum):
    CHOSEN = "chosen"
    INCOMPLETE = "incomplete"
    SAFETY_VIOLATION = "safety_violation"


@dataclass(frozen=True, slots=True)
class PaxosProposalAction:
    action_id: str
    proposer_id: str
    value: str
    events_after: int = 0
    round_number: int | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.proposer_id:
            raise ValueError("proposer_id must be non-empty")
        if not isinstance(self.value, str):
            raise TypeError("Paxos campaign values must be strings")
        if not isinstance(self.events_after, int) or isinstance(self.events_after, bool):
            raise ValueError("events_after must be an integer")
        if self.events_after < 0:
            raise ValueError("events_after must be non-negative")
        if self.round_number is not None:
            if not isinstance(self.round_number, int) or isinstance(self.round_number, bool):
                raise ValueError("round_number must be an integer or None")
            if self.round_number <= 0:
                raise ValueError("round_number must be positive when specified")


@dataclass(frozen=True, slots=True)
class SeededPaxosProposalSchedule:
    seed: int
    actions: tuple[PaxosProposalAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("Paxos proposal action ids must be unique")

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": [
                {
                    "action_id": action.action_id,
                    "proposer_id": action.proposer_id,
                    "value": action.value,
                    "events_after": action.events_after,
                    "round_number": action.round_number,
                }
                for action in self.actions
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> SeededPaxosProposalSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded Paxos proposal schedule format")
        seed = raw.get("seed")
        actions = raw.get("actions")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")

        decoded: list[PaxosProposalAction] = []
        for item in actions:
            if not isinstance(item, dict):
                raise ValueError("each Paxos proposal action must be an object")
            try:
                decoded.append(
                    PaxosProposalAction(
                        action_id=item["action_id"],
                        proposer_id=item["proposer_id"],
                        value=item["value"],
                        events_after=item["events_after"],
                        round_number=item.get("round_number"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid Paxos proposal action") from exc
        return cls(seed=seed, actions=tuple(decoded))


@dataclass(frozen=True, slots=True)
class SeededPaxosProposalGenerator:
    node_ids: tuple[str, ...]
    values: tuple[str, ...] = ("alpha", "beta", "gamma")
    max_events_after: int = 2

    def __post_init__(self) -> None:
        if not self.node_ids or any(not node_id for node_id in self.node_ids):
            raise ValueError("node_ids must contain non-empty strings")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        if not self.values or any(not isinstance(value, str) for value in self.values):
            raise ValueError("values must contain strings")
        if not isinstance(self.max_events_after, int) or isinstance(
            self.max_events_after, bool
        ):
            raise ValueError("max_events_after must be an integer")
        if self.max_events_after < 0:
            raise ValueError("max_events_after must be non-negative")

    def compile(self, seed: int, action_count: int) -> SeededPaxosProposalSchedule:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(action_count, int) or isinstance(action_count, bool):
            raise ValueError("action_count must be an integer")
        if action_count <= 0:
            raise ValueError("action_count must be positive")

        rng = random.Random(seed)
        node_ids = tuple(sorted(self.node_ids))
        actions = tuple(
            PaxosProposalAction(
                action_id=f"paxos-proposal-{index + 1:06d}",
                proposer_id=rng.choice(node_ids),
                value=rng.choice(self.values),
                events_after=rng.randint(0, self.max_events_after),
            )
            for index in range(action_count)
        )
        return SeededPaxosProposalSchedule(seed=seed, actions=actions)


@dataclass(frozen=True, slots=True)
class PaxosScenarioResult:
    outcome: PaxosScenarioOutcome
    chosen: LearnedDecision | None
    violation: str | None
    trace: tuple[TraceRecord, ...]


class PaxosScenarioRunner:
    """Replay explicit Paxos proposal, fault, and lifecycle schedules without RNG."""

    def __init__(
        self,
        proposals: SeededPaxosProposalSchedule,
        faults: SeededFaultSchedule,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        final_event_budget: int = 128,
    ) -> None:
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        unknown_proposers = sorted(
            {action.proposer_id for action in proposals.actions} - set(node_ids)
        )
        if unknown_proposers:
            raise ValueError(
                f"Paxos proposal schedule references unknown nodes: {unknown_proposers!r}"
            )
        if proposals.seed != faults.seed:
            raise ValueError("proposal and fault schedules must use the same seed")
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(proposals.seed)
        if lifecycle.seed != proposals.seed:
            raise ValueError("proposal and lifecycle schedules must use the same seed")
        unknown_lifecycle = sorted(
            {action.node_id for action in lifecycle.actions} - set(node_ids)
        )
        if unknown_lifecycle:
            raise ValueError(
                f"Paxos lifecycle schedule references unknown nodes: {unknown_lifecycle!r}"
            )
        if any(
            action.before_action_index > len(proposals.actions)
            for action in lifecycle.actions
        ):
            raise ValueError("Paxos lifecycle action references a proposal boundary out of range")
        if not isinstance(final_event_budget, int) or isinstance(final_event_budget, bool):
            raise ValueError("final_event_budget must be an integer")
        if final_event_budget < 0:
            raise ValueError("final_event_budget must be non-negative")

        self.proposals = proposals
        self.faults = faults
        self.lifecycle = lifecycle
        self.node_ids = node_ids
        self.final_event_budget = final_event_budget

    def run(self) -> PaxosScenarioResult:
        sim = Simulator(fault_plan=self.faults.to_fault_plan())
        cluster = PaxosCluster(sim, self.node_ids)
        safety = PaxosSafetyHarness(cluster)
        lifecycle_position = 0
        violation: str | None = None

        try:
            for boundary, action in enumerate(self.proposals.actions):
                lifecycle_position = self._apply_lifecycle_boundary(
                    boundary,
                    lifecycle_position,
                    sim,
                    safety,
                )
                if not sim.is_alive(action.proposer_id):
                    sim._record(
                        "scenario-paxos-proposal-skipped",
                        action_id=action.action_id,
                        proposer=action.proposer_id,
                        value=action.value,
                        reason="proposer-crashed",
                    )
                else:
                    try:
                        proposal = cluster.node(action.proposer_id).start_proposal(
                            action.value,
                            round_number=action.round_number,
                        )
                    except PaxosProposalActive as exc:
                        sim._record(
                            "scenario-paxos-proposal-skipped",
                            action_id=action.action_id,
                            proposer=action.proposer_id,
                            value=action.value,
                            reason="proposal-active",
                            detail=str(exc),
                        )
                    except PaxosError as exc:
                        raise PaxosScenarioExecutionError(str(exc)) from exc
                    else:
                        sim._record(
                            "scenario-paxos-proposal",
                            action_id=action.action_id,
                            proposer=action.proposer_id,
                            value=action.value,
                            proposal=proposal,
                            events_after=action.events_after,
                        )
                if action.events_after:
                    sim.run(max_events=action.events_after)
                safety.checkpoint()

            lifecycle_position = self._apply_lifecycle_boundary(
                len(self.proposals.actions),
                lifecycle_position,
                sim,
                safety,
            )
            if lifecycle_position != len(self.lifecycle.actions):
                raise AssertionError("Paxos lifecycle schedule was not fully consumed")
            sim.run(max_events=self.final_event_budget)
            safety.checkpoint()
        except PaxosSafetyViolation as exc:
            violation = str(exc)

        if violation is not None:
            outcome = PaxosScenarioOutcome.SAFETY_VIOLATION
        elif cluster.chosen is not None:
            outcome = PaxosScenarioOutcome.CHOSEN
        else:
            outcome = PaxosScenarioOutcome.INCOMPLETE
        return PaxosScenarioResult(
            outcome=outcome,
            chosen=cluster.chosen,
            violation=violation,
            trace=tuple(sim.trace),
        )

    def _apply_lifecycle_boundary(
        self,
        boundary: int,
        position: int,
        sim: Simulator,
        safety: PaxosSafetyHarness,
    ) -> int:
        while position < len(self.lifecycle.actions):
            action = self.lifecycle.actions[position]
            if action.before_action_index != boundary:
                break
            if action.kind is NodeLifecycleKind.CRASH:
                if not sim.is_alive(action.node_id):
                    raise PaxosScenarioExecutionError(
                        f"cannot crash already crashed node {action.node_id!r}"
                    )
                sim.crash(action.node_id)
            else:
                if sim.is_alive(action.node_id):
                    raise PaxosScenarioExecutionError(
                        f"cannot restart live node {action.node_id!r}"
                    )
                sim.restart(action.node_id)
            sim._record(
                "scenario-paxos-lifecycle",
                action_id=action.action_id,
                node=action.node_id,
                action=action.kind.value,
                before_action_index=boundary,
            )
            safety.checkpoint()
            position += 1
        return position


_PAXOS_PAYLOAD_TYPES = ("AcceptRequest", "Accepted", "Learn", "Prepare", "Promise")


def paxos_fault_opportunities(
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
        for payload_type in _PAXOS_PAYLOAD_TYPES
    )


@dataclass(frozen=True, slots=True)
class PaxosTrialArtifact:
    """Persisted exact-replay evidence for one deterministic Paxos campaign trial."""

    seed: int
    proposals: SeededPaxosProposalSchedule
    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    node_ids: tuple[str, ...]
    final_event_budget: int
    outcome: PaxosScenarioOutcome
    chosen: LearnedDecision | None
    violation: str | None
    trace_json: str

    @classmethod
    def capture(
        cls,
        proposals: SeededPaxosProposalSchedule,
        faults: SeededFaultSchedule,
        result: PaxosScenarioResult,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        final_event_budget: int = 128,
    ) -> PaxosTrialArtifact:
        lifecycle = lifecycle or SeededLifecycleSchedule.empty(proposals.seed)
        if proposals.seed != faults.seed or proposals.seed != lifecycle.seed:
            raise ValueError("Paxos trial schedules must use the same seed")
        return cls(
            seed=proposals.seed,
            proposals=proposals,
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
        chosen = None
        if self.chosen is not None:
            chosen = {
                "proposal": {
                    "round": self.chosen.proposal.round,
                    "proposer_id": self.chosen.proposal.proposer_id,
                },
                "value": self.chosen.value,
            }
        payload = {
            "version": 1,
            "seed": self.seed,
            "proposals": json.loads(self.proposals.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "lifecycle": json.loads(self.lifecycle.to_json()),
            "node_ids": list(self.node_ids),
            "final_event_budget": self.final_event_budget,
            "outcome": self.outcome.value,
            "chosen": chosen,
            "violation": self.violation,
            "trace": json.loads(self.trace_json),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> PaxosTrialArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported Paxos trial artifact format")
        seed = raw.get("seed")
        node_ids = raw.get("node_ids")
        final_event_budget = raw.get("final_event_budget")
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
        if not isinstance(trace, list):
            raise ValueError("trace must be a list")
        if violation is not None and not isinstance(violation, str):
            raise ValueError("violation must be a string or null")
        try:
            outcome = PaxosScenarioOutcome(raw["outcome"])
            proposals = SeededPaxosProposalSchedule.from_json(
                canonical_json(raw["proposals"])
            )
            faults = SeededFaultSchedule.from_json(canonical_json(raw["faults"]))
            lifecycle = SeededLifecycleSchedule.from_json(
                canonical_json(raw["lifecycle"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Paxos trial artifact") from exc
        if proposals.seed != seed or faults.seed != seed or lifecycle.seed != seed:
            raise ValueError("artifact seed must match all Paxos schedules")

        chosen_raw = raw.get("chosen")
        chosen: LearnedDecision | None = None
        if chosen_raw is not None:
            if not isinstance(chosen_raw, dict):
                raise ValueError("chosen must be an object or null")
            proposal_raw = chosen_raw.get("proposal")
            if not isinstance(proposal_raw, dict):
                raise ValueError("chosen proposal must be an object")
            try:
                proposal = ProposalNumber(
                    proposal_raw["round"],
                    proposal_raw["proposer_id"],
                )
                value = chosen_raw["value"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid chosen Paxos decision") from exc
            if not isinstance(value, str):
                raise ValueError("chosen Paxos campaign value must be a string")
            chosen = LearnedDecision(proposal, value)

        return cls(
            seed=seed,
            proposals=proposals,
            faults=faults,
            lifecycle=lifecycle,
            node_ids=tuple(node_ids),
            final_event_budget=final_event_budget,
            outcome=outcome,
            chosen=chosen,
            violation=violation,
            trace_json=canonical_json(trace),
        )

    def replay(self) -> PaxosScenarioResult:
        result = PaxosScenarioRunner(
            self.proposals,
            self.faults,
            lifecycle=self.lifecycle,
            node_ids=self.node_ids,
            final_event_budget=self.final_event_budget,
        ).run()
        if result.outcome is not self.outcome:
            raise PaxosReplayMismatch(
                f"Paxos outcome changed from {self.outcome.value!r} "
                f"to {result.outcome.value!r}"
            )
        if result.chosen != self.chosen:
            raise PaxosReplayMismatch("Paxos chosen evidence changed during replay")
        if result.violation != self.violation:
            raise PaxosReplayMismatch("Paxos safety violation changed during replay")
        if encode_trace(result.trace) != self.trace_json:
            raise PaxosReplayMismatch("Paxos trace did not replay exactly")
        return result


@dataclass(frozen=True, slots=True)
class PaxosCampaignResult:
    attempted_seeds: tuple[int, ...]
    trials: tuple[PaxosTrialArtifact, ...]
    safety_failure: PaxosTrialArtifact | None


@dataclass(frozen=True, slots=True)
class SeededPaxosCampaign:
    proposal_generator: SeededPaxosProposalGenerator
    fault_generator: SeededFaultGenerator
    proposal_count: int
    lifecycle_generator: SeededLifecycleGenerator | None = None
    node_ids: tuple[str, ...] = ("n1", "n2", "n3")
    max_message_ordinal: int = 12
    final_event_budget: int = 128

    def __post_init__(self) -> None:
        if self.proposal_generator.node_ids != self.node_ids:
            raise ValueError("proposal generator node_ids must match campaign node_ids")
        if (
            self.lifecycle_generator is not None
            and tuple(sorted(self.lifecycle_generator.nodes)) != tuple(sorted(self.node_ids))
        ):
            raise ValueError("lifecycle generator nodes must match campaign node_ids")
        if not isinstance(self.proposal_count, int) or isinstance(self.proposal_count, bool):
            raise ValueError("proposal_count must be an integer")
        if self.proposal_count <= 0:
            raise ValueError("proposal_count must be positive")
        paxos_fault_opportunities(
            self.node_ids,
            max_message_ordinal=self.max_message_ordinal,
        )
        if not isinstance(self.final_event_budget, int) or isinstance(
            self.final_event_budget, bool
        ):
            raise ValueError("final_event_budget must be an integer")
        if self.final_event_budget < 0:
            raise ValueError("final_event_budget must be non-negative")

    def run(self, seeds: tuple[int, ...]) -> PaxosCampaignResult:
        attempted: list[int] = []
        trials: list[PaxosTrialArtifact] = []
        opportunities = paxos_fault_opportunities(
            self.node_ids,
            max_message_ordinal=self.max_message_ordinal,
        )

        for seed in seeds:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError("campaign seeds must be integers")
            proposals = self.proposal_generator.compile(seed, self.proposal_count)
            faults = self.fault_generator.compile(seed, opportunities)
            lifecycle = (
                self.lifecycle_generator.compile(seed, len(proposals.actions))
                if self.lifecycle_generator is not None
                else SeededLifecycleSchedule.empty(seed)
            )
            result = PaxosScenarioRunner(
                proposals,
                faults,
                lifecycle=lifecycle,
                node_ids=self.node_ids,
                final_event_budget=self.final_event_budget,
            ).run()
            artifact = PaxosTrialArtifact.capture(
                proposals,
                faults,
                result,
                lifecycle=lifecycle,
                node_ids=self.node_ids,
                final_event_budget=self.final_event_budget,
            )
            attempted.append(seed)
            trials.append(artifact)
            if result.outcome is PaxosScenarioOutcome.SAFETY_VIOLATION:
                return PaxosCampaignResult(
                    attempted_seeds=tuple(attempted),
                    trials=tuple(trials),
                    safety_failure=artifact,
                )

        return PaxosCampaignResult(
            attempted_seeds=tuple(attempted),
            trials=tuple(trials),
            safety_failure=None,
        )
