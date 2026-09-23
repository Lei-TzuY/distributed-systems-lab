from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .paxos import PaxosError, PaxosSafetyViolation, ProposalNumber
from .paxos_log import SlotAcceptedValue, SlotLearnedDecision
from .simulator import Message, Simulator


@dataclass(frozen=True, slots=True)
class LeaderPrepare:
    ballot: ProposalNumber


@dataclass(frozen=True, slots=True)
class LeaderPromise:
    ballot: ProposalNumber
    acceptor_id: str
    accepted: tuple[SlotAcceptedValue, ...]


@dataclass(frozen=True, slots=True)
class LeaderAccept:
    slot: int
    ballot: ProposalNumber
    value: Any


@dataclass(frozen=True, slots=True)
class LeaderAccepted:
    slot: int
    ballot: ProposalNumber
    acceptor_id: str
    value: Any


class MultiPaxosCluster:
    """Stable-leader Multi-Paxos with one durable Phase-1 promise per ballot."""

    def __init__(self, sim: Simulator, node_ids: tuple[str, ...]) -> None:
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("Multi-Paxos requires unique non-empty node ids")
        self.sim = sim
        self.node_ids = node_ids
        self.nodes = {
            node_id: MultiPaxosNode(self, node_id, tuple(n for n in node_ids if n != node_id))
            for node_id in node_ids
        }
        self._chosen: dict[int, SlotLearnedDecision] = {}
        self._recover_chosen()
        for node_id, node in self.nodes.items():
            sim.register(
                node_id,
                node.handle_message,
                crash_handler=node.handle_crash,
                restart_handler=node.handle_restart,
            )

    @property
    def quorum_size(self) -> int:
        return len(self.node_ids) // 2 + 1

    def node(self, node_id: str) -> MultiPaxosNode:
        return self.nodes[node_id]

    def chosen(self, slot: int) -> SlotLearnedDecision | None:
        return self._chosen.get(slot)

    def _recover_chosen(self) -> None:
        by_slot: dict[int, list[tuple[ProposalNumber, Any, str]]] = {}
        for node_id, node in self.nodes.items():
            for accepted in node.accept_history:
                by_slot.setdefault(accepted.slot, []).append(
                    (accepted.proposal, accepted.value, node_id)
                )
        for slot, records in by_slot.items():
            quorum_evidence: list[tuple[ProposalNumber, Any]] = []
            for ballot, value, _ in records:
                acceptors = {
                    node_id
                    for candidate_ballot, candidate_value, node_id in records
                    if candidate_ballot == ballot and candidate_value == value
                }
                evidence = (ballot, value)
                if len(acceptors) >= self.quorum_size and evidence not in quorum_evidence:
                    quorum_evidence.append(evidence)
            if not quorum_evidence:
                continue
            first_value = quorum_evidence[0][1]
            if any(value != first_value for _, value in quorum_evidence[1:]):
                raise PaxosSafetyViolation(
                    f"Multi-Paxos slot {slot} has conflicting durable quorums"
                )
            ballot = max(ballot for ballot, _ in quorum_evidence)
            self._chosen[slot] = SlotLearnedDecision(slot, ballot, first_value)
            self.sim._record(
                "multipaxos-chosen-recovered",
                slot=slot,
                ballot=ballot,
                value=first_value,
            )
        self.assert_safety()

    def _record_chosen(self, slot: int, ballot: ProposalNumber, value: Any) -> None:
        current = self._chosen.get(slot)
        if current is not None and current.value != value:
            raise PaxosSafetyViolation(f"Multi-Paxos slot {slot} chose conflicting values")
        if current is None:
            self._chosen[slot] = SlotLearnedDecision(slot, ballot, value)
            self.sim._record("multipaxos-chosen", slot=slot, ballot=ballot, value=value)

    def assert_safety(self) -> None:
        by_slot: dict[int, list[tuple[ProposalNumber, Any, str]]] = {}
        for node_id, node in self.nodes.items():
            for accepted in node.accept_history:
                by_slot.setdefault(accepted.slot, []).append(
                    (accepted.proposal, accepted.value, node_id)
                )
        for slot, records in by_slot.items():
            candidates: list[Any] = []
            for ballot, value, _ in records:
                acceptors = {
                    node
                    for candidate_ballot, candidate_value, node in records
                    if candidate_ballot == ballot and candidate_value == value
                }
                if len(acceptors) >= self.quorum_size:
                    candidates.append(value)
            if candidates and any(value != candidates[0] for value in candidates[1:]):
                raise PaxosSafetyViolation(
                    f"Multi-Paxos slot {slot} has conflicting quorums"
                )
            chosen = self._chosen.get(slot)
            if chosen is not None and candidates and chosen.value != candidates[0]:
                raise PaxosSafetyViolation(
                    f"Multi-Paxos slot {slot} runtime/durable mismatch"
                )


class MultiPaxosNode:
    _PROMISED = "multipaxos_promised_ballot"
    _ACCEPTED = "multipaxos_accepted"
    _HISTORY = "multipaxos_accept_history"
    _LAST_ROUND = "multipaxos_last_round"

    def __init__(
        self,
        cluster: MultiPaxosCluster,
        node_id: str,
        peers: tuple[str, ...],
    ) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
        state = self.sim.persistent_state[node_id]
        state.setdefault(self._PROMISED, None)
        state.setdefault(self._ACCEPTED, {})
        state.setdefault(self._HISTORY, ())
        state.setdefault(self._LAST_ROUND, 0)
        self._leader_ballot: ProposalNumber | None = None
        self._promises: dict[str, LeaderPromise] = {}
        self._pending: dict[int, Any] = {}
        self._accepted_by: dict[int, set[str]] = {}

    @property
    def promised_ballot(self) -> ProposalNumber | None:
        value = self.sim.persistent_state[self.node_id][self._PROMISED]
        assert value is None or isinstance(value, ProposalNumber)
        return value

    @property
    def accept_history(self) -> tuple[SlotAcceptedValue, ...]:
        value = self.sim.persistent_state[self.node_id][self._HISTORY]
        assert isinstance(value, tuple)
        return value

    @property
    def has_leader_authority(self) -> bool:
        return (
            self._leader_ballot is not None
            and len(self._promises) >= self.cluster.quorum_size
        )

    def prepare_leadership(self, *, round_number: int | None = None) -> ProposalNumber:
        if not self.sim.is_alive(self.node_id):
            raise PaxosError("crashed node cannot prepare Multi-Paxos leadership")
        state = self.sim.persistent_state[self.node_id]
        last = int(state[self._LAST_ROUND])
        promised = self.promised_ballot
        if round_number is None:
            round_number = max(last, promised.round if promised is not None else 0) + 1
        if round_number <= last:
            raise ValueError("leader ballot round must increase monotonically")
        ballot = ProposalNumber(round_number, self.node_id)
        state[self._LAST_ROUND] = round_number
        self._leader_ballot = ballot
        self._promises.clear()
        self._pending.clear()
        self._accepted_by.clear()
        request = LeaderPrepare(ballot)
        self._receive_prepare(request)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)
        self.sim._record("multipaxos-phase1-start", leader=self.node_id, ballot=ballot)
        return ballot

    def propose(self, slot: int, value: Any) -> None:
        if slot <= 0:
            raise ValueError("Multi-Paxos slot must be positive")
        if not self.has_leader_authority or self._leader_ballot is None:
            raise PaxosError("Multi-Paxos proposal requires Phase-1 quorum authority")
        if slot in self._pending:
            raise PaxosError(f"slot {slot} already has an active proposal")
        adopted = self._adopted_value(slot)
        chosen_value = value if adopted is None else adopted
        self._pending[slot] = chosen_value
        self._accepted_by[slot] = set()
        request = LeaderAccept(slot, self._leader_ballot, chosen_value)
        self._receive_accept(request)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)
        self.sim._record(
            "multipaxos-phase2-start",
            slot=slot,
            leader=self.node_id,
            ballot=self._leader_ballot,
            value=chosen_value,
        )

    def _adopted_value(self, slot: int) -> Any | None:
        accepted = [
            item
            for promise in self._promises.values()
            for item in promise.accepted
            if item.slot == slot
        ]
        if not accepted:
            return None
        highest = max(item.proposal for item in accepted)
        values = [item.value for item in accepted if item.proposal == highest]
        first = values[0]
        if any(value != first for value in values[1:]):
            raise PaxosSafetyViolation(
                f"slot {slot} has conflicting accepted values at {highest!r}"
            )
        return first

    def handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim or message.dst != self.node_id:
            raise ValueError("invalid Multi-Paxos message delivery")
        payload = message.payload
        if isinstance(payload, LeaderPrepare):
            if message.src != payload.ballot.proposer_id:
                return
            self._receive_prepare(payload)
        elif isinstance(payload, LeaderPromise):
            if message.src != payload.acceptor_id:
                return
            self._receive_promise(payload)
        elif isinstance(payload, LeaderAccept):
            if message.src != payload.ballot.proposer_id:
                return
            self._receive_accept(payload)
        elif isinstance(payload, LeaderAccepted):
            if message.src != payload.acceptor_id:
                return
            self._receive_accepted(payload)
        else:
            raise TypeError(f"unsupported Multi-Paxos message {type(payload).__name__}")

    def _receive_prepare(self, request: LeaderPrepare) -> None:
        promised = self.promised_ballot
        if promised is not None and request.ballot < promised:
            return
        self.sim.persistent_state[self.node_id][self._PROMISED] = request.ballot
        accepted = self.sim.persistent_state[self.node_id][self._ACCEPTED]
        promise = LeaderPromise(request.ballot, self.node_id, tuple(accepted.values()))
        if request.ballot.proposer_id == self.node_id:
            self._receive_promise(promise)
        else:
            self.sim.send(self.node_id, request.ballot.proposer_id, promise)

    def _receive_promise(self, response: LeaderPromise) -> None:
        if (
            response.ballot != self._leader_ballot
            or response.acceptor_id not in self.cluster.nodes
        ):
            return
        self._promises.setdefault(response.acceptor_id, response)
        if len(self._promises) == self.cluster.quorum_size:
            self.sim._record(
                "multipaxos-phase1-complete",
                leader=self.node_id,
                ballot=response.ballot,
                promises=tuple(sorted(self._promises)),
            )

    def _receive_accept(self, request: LeaderAccept) -> None:
        promised = self.promised_ballot
        if promised is not None and request.ballot < promised:
            self.sim._record(
                "multipaxos-accept-rejected",
                slot=request.slot,
                acceptor=self.node_id,
                ballot=request.ballot,
                promised=promised,
            )
            return
        state = self.sim.persistent_state[self.node_id]
        state[self._PROMISED] = request.ballot
        current = state[self._ACCEPTED].get(request.slot)
        if (
            current is not None
            and current.proposal == request.ballot
            and current.value != request.value
        ):
            raise PaxosSafetyViolation(
                "same Multi-Paxos ballot accepted conflicting values"
            )
        accepted = SlotAcceptedValue(request.slot, request.ballot, request.value)
        state[self._ACCEPTED][request.slot] = accepted
        if accepted not in state[self._HISTORY]:
            state[self._HISTORY] = (*state[self._HISTORY], accepted)
        response = LeaderAccepted(request.slot, request.ballot, self.node_id, request.value)
        if request.ballot.proposer_id == self.node_id:
            self._receive_accepted(response)
        else:
            self.sim.send(self.node_id, request.ballot.proposer_id, response)

    def _receive_accepted(self, response: LeaderAccepted) -> None:
        if response.ballot != self._leader_ballot or response.slot not in self._pending:
            return
        if response.value != self._pending[response.slot]:
            raise PaxosSafetyViolation(
                "Multi-Paxos accepted response changed proposed value"
            )
        accepted_by = self._accepted_by[response.slot]
        accepted_by.add(response.acceptor_id)
        if len(accepted_by) >= self.cluster.quorum_size:
            self.cluster._record_chosen(response.slot, response.ballot, response.value)
            self._pending.pop(response.slot, None)
            self._accepted_by.pop(response.slot, None)

    def handle_crash(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Multi-Paxos crash from different simulator")
        self._leader_ballot = None
        self._promises.clear()
        self._pending.clear()
        self._accepted_by.clear()

    def handle_restart(self, sim: Simulator) -> None:
        self.handle_crash(sim)
        self.sim._record("multipaxos-node-restart", node=self.node_id)
