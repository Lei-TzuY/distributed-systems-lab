"""Opt-in application-aware Multi-Paxos fault campaign.

The legacy consensus-only campaign and its v1 artifacts are not modified.
This runner uses existing compiled action/fault/lifecycle schedules and captures
actual application, client history, and exact deterministic replay evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .kv import Delete, Put
from .lifecycle import SeededLifecycleSchedule
from .linearizability import Get, SingleKeyKVLinearizabilityChecker
from .multipaxos import MultiPaxosCluster
from .multipaxos_campaign import (
    MultiPaxosActionKind,
    MultiPaxosScenarioOutcome,
    MultiPaxosScenarioRunner,
    SeededMultiPaxosActionSchedule,
)
from .multipaxos_client import MultiPaxosClient
from .multipaxos_state_machine import Append, MultiPaxosRequest, MultiPaxosStateMachine
from .paxos import PaxosError, PaxosSafetyViolation
from .randomized_faults import SeededFaultSchedule
from .replay_codec import canonical_json, encode_trace
from .simulator import Simulator


def _operation(encoded: str) -> Put | Delete | Get | Append:
    kind, separator, rest = encoded.partition(":")
    if not separator:
        raise ValueError(
            "application action must be put:key:value, delete:key, "
            "get:key, or append:key:suffix"
        )
    if kind in ("put", "append"):
        key, separator, value = rest.partition(":")
        if not separator or not key:
            raise ValueError("put/append requires a nonempty key and a value")
        return Put(key, value) if kind == "put" else Append(key, value)
    if kind in ("get", "delete") and rest:
        return Get(rest) if kind == "get" else Delete(rest)
    raise ValueError("unsupported application command")


@dataclass(frozen=True, slots=True)
class MultiPaxosApplicationResult:
    outcome: MultiPaxosScenarioOutcome
    violation: str | None
    applied_slots: tuple[tuple[str, tuple[int, ...]], ...]
    kv: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    completed: tuple[tuple[str, str | None], ...]
    pending: tuple[str, ...]
    linearizable: bool | None
    trace_json: str

    def to_json(self) -> str:
        """Canonical evidence, including protocol/application trace, not wall-clock time."""
        return canonical_json({
            "version": 1,
            "outcome": self.outcome.value,
            "violation": self.violation,
            "applied_slots": self.applied_slots,
            "kv": self.kv,
            "completed": self.completed,
            "pending": self.pending,
            "linearizable": self.linearizable,
            "trace": json.loads(self.trace_json),
        })


class MultiPaxosApplicationScenarioRunner(MultiPaxosScenarioRunner):
    """Replay existing schedules with real learned/application/client semantics.

    PROPOSE values use put:key:value, delete:key, get:key, or append:key:suffix.
    Each action ID identifies one client operation. No automatically generated
    responses are inserted for lost acknowledgements. Append is excluded from
    the existing Put/Delete/Get history oracle, not declared linearizable.
    """

    def __init__(
        self,
        actions: SeededMultiPaxosActionSchedule,
        faults: SeededFaultSchedule,
        *,
        withheld_acks: tuple[str, ...] = (),
        application_rounds: int = 8,
        events_per_round: int = 128,
        **kwargs: object,
    ) -> None:
        super().__init__(actions, faults, **kwargs)
        if type(application_rounds) is not int or application_rounds < 0:
            raise ValueError("application_rounds must be non-negative integer")
        if type(events_per_round) is not int or events_per_round <= 0:
            raise ValueError("events_per_round must be a positive integer")
        ids = {a.action_id for a in actions.actions if a.kind is MultiPaxosActionKind.PROPOSE}
        if len(set(withheld_acks)) != len(withheld_acks) or not set(withheld_acks) <= ids:
            raise ValueError("withheld_acks must be unique proposal action IDs")
        if any(not isinstance(aid, str) for aid in withheld_acks):
            raise ValueError("withheld_acks must contain strings")
        self.requests: dict[str, MultiPaxosRequest] = {}
        for index, action in enumerate(actions.actions, start=1):
            if action.kind is MultiPaxosActionKind.PROPOSE:
                assert action.value is not None
                self.requests[action.action_id] = MultiPaxosRequest(
                    action.node_id, index, action.action_id, _operation(action.value)
                )
        keys = {request.operation.key for request in self.requests.values()}
        if len(keys) > 1:
            raise ValueError("application campaign currently supports single-key schedules")
        self.withheld_acks = frozenset(withheld_acks)
        self.application_rounds = application_rounds
        self.events_per_round = events_per_round

    def run(self) -> MultiPaxosApplicationResult:
        sim = Simulator(fault_plan=self.faults.to_fault_plan())
        cluster = MultiPaxosCluster(sim, self.node_ids, learn_decisions=True)
        machine = MultiPaxosStateMachine(cluster)
        client = MultiPaxosClient(machine)
        position = 0
        violation = None

        def observe() -> None:
            for node_id in self.node_ids:
                if sim.is_alive(node_id):
                    machine.apply_ready(node_id)
            for action in self.actions.actions:
                request = self.requests.get(action.action_id)
                if request is not None and action.action_id in client.journal.requests:
                    if sim.is_alive(action.node_id):
                        client.poll(
                            action.node_id,
                            request,
                            acknowledge=action.action_id not in self.withheld_acks,
                        )
            machine.assert_safety()

        try:
            for boundary, action in enumerate(self.actions.actions):
                position = self._apply_lifecycle_boundary(
                    boundary, position, sim, cluster
                )
                if action.kind is MultiPaxosActionKind.PREPARE:
                    super()._execute_action(action, sim, cluster)
                elif sim.is_alive(action.node_id):
                    request = self.requests[action.action_id]
                    try:
                        result = client.submit(action.node_id, request, slot=action.slot)
                    except PaxosError as exc:
                        sim._record(
                            "scenario-multipaxos-client-rejected",
                            action_id=action.action_id,
                            reason=str(exc),
                        )
                    else:
                        sim._record(
                            "scenario-multipaxos-client-submit",
                            action_id=action.action_id,
                            status=result.status.value,
                            slot=result.slot,
                        )
                else:
                    sim._record(
                        "scenario-multipaxos-action-skipped",
                        action_id=action.action_id,
                        action=action.kind.value,
                        node=action.node_id,
                        reason="node-crashed",
                    )
                if action.events_after:
                    sim.run(max_events=action.events_after)
                observe()

            position = self._apply_lifecycle_boundary(
                len(self.actions.actions), position, sim, cluster
            )
            if position != len(self.lifecycle.actions):
                raise AssertionError("lifecycle schedule was not fully consumed")
            sim.run(max_events=self.final_event_budget)
            observe()
            for _ in range(self.application_rounds):
                sim.run(max_events=self.events_per_round)
                observe()
        except PaxosSafetyViolation as exc:
            violation = str(exc)

        applied = tuple(
            (node_id, tuple(item.slot for item in machine.applied(node_id)))
            for node_id in self.node_ids
        )
        states = tuple(
            (
                node_id,
                tuple(sorted(MultiPaxosStateMachine.replay(machine.applied(node_id)).items())),
            )
            for node_id in self.node_ids
        )
        completed = tuple(
            (item.operation_id, item.completion.result)
            for item in client.history.completed()
        )
        pending = tuple(item.operation_id for item in client.history.pending())
        has_append = any(
            isinstance(request.operation, Append) for request in self.requests.values()
        )
        linearizable = None
        if not has_append and violation is None:
            linearizable = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(
                client.history
            ).linearizable
        if violation is not None:
            outcome = MultiPaxosScenarioOutcome.SAFETY_VIOLATION
        elif cluster.chosen_slots():
            outcome = MultiPaxosScenarioOutcome.PROGRESS
        else:
            outcome = MultiPaxosScenarioOutcome.INCOMPLETE
        return MultiPaxosApplicationResult(
            outcome=outcome,
            violation=violation,
            applied_slots=applied,
            kv=states,
            completed=completed,
            pending=pending,
            linearizable=linearizable,
            trace_json=encode_trace(tuple(sim.trace)),
        )


@dataclass(frozen=True, slots=True)
class MultiPaxosApplicationArtifact:
    """Self-contained opt-in replay artifact; legacy campaign v1 is unchanged."""

    actions: SeededMultiPaxosActionSchedule
    faults: SeededFaultSchedule
    lifecycle: SeededLifecycleSchedule
    node_ids: tuple[str, ...]
    final_event_budget: int
    withheld_acks: tuple[str, ...]
    application_rounds: int
    events_per_round: int
    report_json: str

    @classmethod
    def capture(
        cls,
        runner: MultiPaxosApplicationScenarioRunner,
        result: MultiPaxosApplicationResult,
    ) -> MultiPaxosApplicationArtifact:
        return cls(
            actions=runner.actions,
            faults=runner.faults,
            lifecycle=runner.lifecycle,
            node_ids=runner.node_ids,
            final_event_budget=runner.final_event_budget,
            withheld_acks=tuple(sorted(runner.withheld_acks)),
            application_rounds=runner.application_rounds,
            events_per_round=runner.events_per_round,
            report_json=result.to_json(),
        )

    def to_json(self) -> str:
        return canonical_json({
            "version": 1,
            "actions": json.loads(self.actions.to_json()),
            "faults": json.loads(self.faults.to_json()),
            "lifecycle": json.loads(self.lifecycle.to_json()),
            "node_ids": self.node_ids,
            "final_event_budget": self.final_event_budget,
            "withheld_acks": self.withheld_acks,
            "application_rounds": self.application_rounds,
            "events_per_round": self.events_per_round,
            "report": json.loads(self.report_json),
        })

    @classmethod
    def from_json(cls, encoded: str) -> MultiPaxosApplicationArtifact:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported Multi-Paxos application artifact")
        try:
            actions = SeededMultiPaxosActionSchedule.from_json(
                canonical_json(raw["actions"])
            )
            faults = SeededFaultSchedule.from_json(canonical_json(raw["faults"]))
            lifecycle = SeededLifecycleSchedule.from_json(
                canonical_json(raw["lifecycle"])
            )
            node_ids = raw["node_ids"]
            withheld = raw["withheld_acks"]
            if not isinstance(node_ids, list) or not all(
                isinstance(node, str) and node for node in node_ids
            ):
                raise ValueError("invalid artifact nodes")
            if not isinstance(withheld, list) or not all(
                isinstance(aid, str) for aid in withheld
            ):
                raise ValueError("invalid withheld acknowledgement identities")
            runner = MultiPaxosApplicationScenarioRunner(
                actions,
                faults,
                lifecycle=lifecycle,
                node_ids=tuple(node_ids),
                final_event_budget=raw["final_event_budget"],
                withheld_acks=tuple(withheld),
                application_rounds=raw["application_rounds"],
                events_per_round=raw["events_per_round"],
            )
            report = raw["report"]
            if not isinstance(report, dict) or report.get("version") != 1:
                raise ValueError("invalid application evidence report")
            return cls(
                actions=runner.actions,
                faults=runner.faults,
                lifecycle=runner.lifecycle,
                node_ids=runner.node_ids,
                final_event_budget=runner.final_event_budget,
                withheld_acks=tuple(sorted(runner.withheld_acks)),
                application_rounds=runner.application_rounds,
                events_per_round=runner.events_per_round,
                report_json=canonical_json(report),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Multi-Paxos application artifact") from exc

    def replay(self) -> MultiPaxosApplicationResult:
        from .multipaxos_campaign import MultiPaxosReplayMismatch

        result = MultiPaxosApplicationScenarioRunner(
            self.actions,
            self.faults,
            lifecycle=self.lifecycle,
            node_ids=self.node_ids,
            final_event_budget=self.final_event_budget,
            withheld_acks=self.withheld_acks,
            application_rounds=self.application_rounds,
            events_per_round=self.events_per_round,
        ).run()
        if result.to_json() != self.report_json:
            raise MultiPaxosReplayMismatch(
                "application campaign result, trace, or client history changed"
            )
        return result
