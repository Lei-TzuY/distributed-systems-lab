from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .kv import ClientRequestConflict, Delete, Put
from .linearizability import Get
from .multipaxos import MultiPaxosCluster
from .paxos import PaxosError, PaxosSafetyViolation
from .paxos_log import SlotLearnedDecision


@dataclass(frozen=True, slots=True)
class Append:
    """Non-idempotent command: applying it twice is visible in the KV state."""

    key: str
    suffix: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("KV key must be non-empty")


@dataclass(frozen=True, slots=True)
class MultiPaxosRequest:
    """One logical client invocation, including its stable history identity."""

    client_id: str
    request_id: int
    operation_id: str
    operation: Put | Delete | Get | Append

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id:
            raise ValueError("client_id must be a non-empty string")
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("operation_id must be a non-empty string")
        if (
            not isinstance(self.request_id, int)
            or isinstance(self.request_id, bool)
            or self.request_id < 0
        ):
            raise ValueError("request_id must be a non-negative integer")
        if not isinstance(self.operation, (Put, Delete, Get, Append)):
            raise TypeError("client operation must be Put, Delete, Get, or Append")
        if not isinstance(self.operation.key, str) or not self.operation.key:
            raise ValueError("client key must be a non-empty string")
        if isinstance(self.operation, Put) and not isinstance(self.operation.value, str):
            raise TypeError("Put value must be a string")
        if isinstance(self.operation, Append) and not isinstance(self.operation.suffix, str):
            raise TypeError("Append suffix must be a string")

    @property
    def identity(self) -> tuple[str, int]:
        return self.client_id, self.request_id


@dataclass(frozen=True, slots=True)
class AppliedRequest:
    request: MultiPaxosRequest
    slot: int
    result: str | None


def _apply_decision(
    state: dict[str, str],
    requests: dict[tuple[str, int], AppliedRequest],
    decision: SlotLearnedDecision,
) -> None:
    command = decision.value
    if not isinstance(command, MultiPaxosRequest):
        apply_multipaxos_command(state, command)
        return
    previous = requests.get(command.identity)
    if previous is not None:
        if previous.request != command:
            raise ClientRequestConflict("conflicting Multi-Paxos request identity")
        return
    if any(r.request.operation_id == command.operation_id for r in requests.values()):
        raise ClientRequestConflict("conflicting Multi-Paxos operation_id")
    if isinstance(command.operation, Get):
        result = state.get(command.operation.key)
    else:
        apply_multipaxos_command(state, command.operation)
        result = None
    requests[command.identity] = AppliedRequest(command, decision.slot, result)


def apply_multipaxos_command(state: dict[str, str], value: Any) -> None:
    """Deterministic KV transition; any other chosen value is an ordered no-op."""
    if isinstance(value, Put):
        state[value.key] = value.value
    elif isinstance(value, Delete):
        state.pop(value.key, None)
    elif isinstance(value, Append):
        state[value.key] = state.get(value.key, "") + value.suffix


class MultiPaxosStateMachine:
    """Apply learned Multi-Paxos decisions in contiguous slot order exactly once.

    Each node's applied prefix is durable: slots ``1..last_applied`` with the
    learned decision applied at each slot. KV state is volatile, kept in the
    simulator's per-node volatile state and rebuilt by replaying the durable
    prefix after a crash or full reconstruction. Application stops at the first
    unlearned slot, and every applied slot must match both the node's durable
    learned decision and the cluster's objective chosen evidence.
    """

    _APPLIED = "multipaxos_applied"
    _VOLATILE = "multipaxos_state_machine"

    def __init__(self, cluster: MultiPaxosCluster) -> None:
        if not cluster.learn_decisions:
            raise ValueError("Multi-Paxos application requires learn_decisions=True")
        self.cluster = cluster
        self.sim = cluster.sim
        for node_id in cluster.node_ids:
            self.sim.persistent_state[node_id].setdefault(self._APPLIED, ())
        self.assert_safety()

    def last_applied(self, node_id: str) -> int:
        return len(self.applied(node_id))

    def applied(self, node_id: str) -> tuple[SlotLearnedDecision, ...]:
        self._require_node(node_id)
        return self._durable_applied(node_id)

    def blocked_on(self, node_id: str) -> int | None:
        """Return the missing slot that holds back already-learned later slots."""
        next_slot = self.last_applied(node_id) + 1
        node = self.cluster.node(node_id)
        if node.learned_decision(next_slot) is not None:
            return None
        if any(decision.slot > next_slot for decision in node.learned_decisions):
            return next_slot
        return None

    def state(self, node_id: str) -> Mapping[str, str]:
        self._require_alive(node_id)
        return MappingProxyType(dict(self._machine(node_id)["kv"]))

    def client_requests(self, node_id: str) -> Mapping[tuple[str, int], AppliedRequest]:
        """Dedup/results reconstructed solely from this node's durable applied prefix."""
        self._require_alive(node_id)
        return MappingProxyType(dict(self._machine(node_id)["requests"]))

    def apply_ready(self, node_id: str) -> tuple[SlotLearnedDecision, ...]:
        """Apply the contiguous prefix, then probe/retry the first unlearned window.

        The application driver keeps calling this and processing simulator work.
        A probe also discovers a lost suffix when no later slot was learned yet.
        """
        self._require_alive(node_id)
        machine = self._machine(node_id)
        node = self.cluster.node(node_id)
        persistent = self.sim.persistent_state[node_id]
        applied = self._durable_applied(node_id)
        applied_now: list[SlotLearnedDecision] = []
        while (decision := node.learned_decision(len(applied) + 1)) is not None:
            self._require_chosen(node_id, decision)
            _apply_decision(machine["kv"], machine["requests"], decision)
            applied = (*applied, decision)
            persistent[self._APPLIED] = applied
            machine["applied_through"] = decision.slot
            applied_now.append(decision)
            self.sim._record(
                "multipaxos-apply",
                node=node_id,
                slot=decision.slot,
                ballot=decision.proposal,
                value=decision.value,
            )
        gap = self.blocked_on(node_id)
        if gap is not None:
            self.sim._record(
                "multipaxos-apply-gap",
                node=node_id,
                missing_slot=gap,
                learned_beyond=tuple(
                    decision.slot
                    for decision in node.learned_decisions
                    if decision.slot > gap
                ),
            )
        node.recover_decisions(len(applied) + 1)
        return tuple(applied_now)

    @staticmethod
    def replay(applied: tuple[SlotLearnedDecision, ...]) -> dict[str, str]:
        return MultiPaxosStateMachine._replay(applied)[0]

    @staticmethod
    def _replay(
        applied: tuple[SlotLearnedDecision, ...],
    ) -> tuple[dict[str, str], dict[tuple[str, int], AppliedRequest]]:
        state: dict[str, str] = {}
        requests: dict[tuple[str, int], AppliedRequest] = {}
        for decision in applied:
            _apply_decision(state, requests, decision)
        return state, requests

    def assert_safety(self) -> None:
        """Re-check durable applied prefixes against learned and quorum evidence."""
        self.cluster.assert_safety()
        reference: dict[int, tuple[str, Any]] = {}
        for node_id in self.cluster.node_ids:
            applied = self._durable_applied(node_id)
            for decision in applied:
                self._require_chosen(node_id, decision)
                canonical = reference.get(decision.slot)
                if canonical is not None and canonical[1] != decision.value:
                    raise PaxosSafetyViolation(
                        f"Multi-Paxos applied slot {decision.slot} diverged: "
                        f"{canonical[0]!r} applied {canonical[1]!r}, but {node_id!r} "
                        f"applied {decision.value!r}"
                    )
                reference.setdefault(decision.slot, (node_id, decision.value))
            machine = self.sim.volatile_state[node_id].get(self._VOLATILE)
            expected_state, expected_requests = self._replay(applied)
            if machine is not None and (
                machine["applied_through"] != len(applied)
                or machine["kv"] != expected_state
                or machine["requests"] != expected_requests
            ):
                raise PaxosSafetyViolation(
                    f"Multi-Paxos state machine for {node_id!r} diverged from its "
                    "durable applied prefix"
                )

    def _machine(self, node_id: str) -> dict[str, Any]:
        volatile = self.sim.volatile_state[node_id]
        applied = self._durable_applied(node_id)
        machine = volatile.get(self._VOLATILE)
        if machine is None:
            for decision in applied:
                self._require_chosen(node_id, decision)
            state, requests = self._replay(applied)
            machine = {"applied_through": len(applied), "kv": state, "requests": requests}
            volatile[self._VOLATILE] = machine
            self.sim._record(
                "multipaxos-state-machine-load",
                node=node_id,
                last_applied=len(applied),
            )
        elif machine["applied_through"] != len(applied):
            raise PaxosSafetyViolation(
                f"Multi-Paxos state machine for {node_id!r} diverged from its "
                "durable applied prefix"
            )
        return machine

    def _durable_applied(self, node_id: str) -> tuple[SlotLearnedDecision, ...]:
        value = self.sim.persistent_state[node_id].get(self._APPLIED, ())
        if not isinstance(value, tuple) or not all(
            isinstance(item, SlotLearnedDecision) for item in value
        ):
            raise PaxosSafetyViolation(
                f"durable Multi-Paxos applied history for {node_id!r} is invalid"
            )
        for expected, item in enumerate(value, start=1):
            if item.slot < expected:
                raise PaxosSafetyViolation(
                    f"durable Multi-Paxos applied history for {node_id!r} applies "
                    f"slot {item.slot} twice"
                )
            if item.slot > expected:
                raise PaxosSafetyViolation(
                    f"durable Multi-Paxos applied history for {node_id!r} skips "
                    f"slot {expected}"
                )
        return value

    def _require_chosen(self, node_id: str, decision: SlotLearnedDecision) -> None:
        learned = self.cluster.node(node_id).learned_decision(decision.slot)
        if learned is None or learned.value != decision.value:
            raise PaxosSafetyViolation(
                f"node {node_id!r} applied Multi-Paxos slot {decision.slot} without "
                "a matching durable learned decision"
            )
        chosen = self.cluster.chosen(decision.slot)
        if chosen is None or chosen.value != decision.value:
            raise PaxosSafetyViolation(
                f"node {node_id!r} applied Multi-Paxos slot {decision.slot} without "
                "matching chosen quorum evidence"
            )

    def _require_alive(self, node_id: str) -> None:
        self._require_node(node_id)
        if not self.sim.is_alive(node_id):
            raise PaxosError(f"crashed node {node_id!r} cannot run its state machine")

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.cluster.nodes:
            raise ValueError(f"unknown Multi-Paxos node {node_id!r}")
