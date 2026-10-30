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


@dataclass(frozen=True, slots=True)
class LeaderDecision:
    slot: int
    ballot: ProposalNumber
    value: Any


RECOVERY_WINDOW = 16
RECOVERY_ATTEMPTS = 3
RECOVERY_RETRY_DELAY = 4  # Simulator ticks, never wall-clock time.


@dataclass(frozen=True, slots=True)
class DecisionRecoveryRequest:
    start_slot: int


@dataclass(frozen=True, slots=True)
class DecisionRecoveryResponse:
    start_slot: int
    decisions: tuple[SlotLearnedDecision, ...]


@dataclass(frozen=True, slots=True)
class _DecisionRecoveryRetry:
    generation: int
    attempt: int


def _valid_slot(slot: object) -> bool:
    return isinstance(slot, int) and not isinstance(slot, bool) and slot > 0


class MultiPaxosCluster:
    """Stable-leader Multi-Paxos with one durable Phase-1 promise per ballot.

    With ``learn_decisions`` enabled, the leader that observes a Phase-2 quorum
    durably learns the chosen slot and broadcasts a ``LeaderDecision`` so every
    node keeps a durable, possibly gapped, learned-decision map for ordered
    application. It is disabled by default so existing campaign traces and
    event budgets stay byte-identical.
    """

    def __init__(
        self,
        sim: Simulator,
        node_ids: tuple[str, ...],
        *,
        learn_decisions: bool = False,
    ) -> None:
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("Multi-Paxos requires unique non-empty node ids")
        self.sim = sim
        self.node_ids = node_ids
        self.learn_decisions = learn_decisions
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

    def chosen_slots(self) -> tuple[SlotLearnedDecision, ...]:
        return tuple(self._chosen[slot] for slot in sorted(self._chosen))

    def _validate_ballot(self, ballot: object) -> None:
        if (
            not isinstance(ballot, ProposalNumber)
            or not _valid_slot(ballot.round)
            or not isinstance(ballot.proposer_id, str)
        ):
            raise PaxosSafetyViolation("Multi-Paxos evidence has invalid ballot")
        if ballot.proposer_id not in self.nodes:
            raise PaxosSafetyViolation(
                f"Multi-Paxos evidence has non-member proposer {ballot.proposer_id!r}"
            )

    def _require_decision_evidence(self, decision: SlotLearnedDecision) -> None:
        # This is a harness safety check, not a source of peer-local decisions.
        # In particular, an accepted minority or a peer's claim is insufficient.
        if not isinstance(decision, SlotLearnedDecision) or not _valid_slot(decision.slot):
            raise PaxosSafetyViolation("Multi-Paxos decision has invalid slot or record")
        self._validate_ballot(decision.proposal)
        acceptors = set()
        for node_id, node in self.nodes.items():
            for accepted in node.accept_history:
                if (
                    accepted.slot == decision.slot
                    and accepted.proposal == decision.proposal
                    and accepted.value == decision.value
                ):
                    self._validate_durable_accepted(node, accepted)
                    acceptors.add(node_id)
                    break
        if len(acceptors) < self.quorum_size:
            raise PaxosSafetyViolation(
                f"Multi-Paxos slot {decision.slot} lacks matching durable quorum evidence "
                f"for ballot {decision.proposal!r}"
            )

    def _validate_durable_accepted(
        self, node: MultiPaxosNode, accepted: SlotAcceptedValue
    ) -> None:
        if not _valid_slot(accepted.slot):
            raise PaxosSafetyViolation(
                f"Multi-Paxos durable history has invalid slot {accepted.slot}"
            )
        self._validate_ballot(accepted.proposal)
        promised = node.promised_ballot
        if promised is None or accepted.proposal > promised:
            raise PaxosSafetyViolation(
                "Multi-Paxos durable accepted proposal exceeds promised ballot"
            )

    def _validate_durable_history(self, node: MultiPaxosNode) -> None:
        node._validate_durable_learned()
        by_slot_ballot: dict[tuple[int, ProposalNumber], Any] = {}
        for accepted in node.accept_history:
            self._validate_durable_accepted(node, accepted)
            key = (accepted.slot, accepted.proposal)
            if key in by_slot_ballot and by_slot_ballot[key] != accepted.value:
                raise PaxosSafetyViolation(
                    "same Multi-Paxos ballot has conflicting durable accepted values"
                )
            by_slot_ballot[key] = accepted.value

    def _recover_chosen(self) -> None:
        by_slot: dict[int, list[tuple[ProposalNumber, Any, str]]] = {}
        for node_id, node in self.nodes.items():
            self._validate_durable_history(node)
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
            self._validate_durable_history(node)
            for accepted in node.accept_history:
                by_slot.setdefault(accepted.slot, []).append(
                    (accepted.proposal, accepted.value, node_id)
                )
        quorum_values: dict[int, Any] = {}
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
            if candidates:
                quorum_values[slot] = candidates[0]
        for node_id, node in self.nodes.items():
            for decision in node.learned_decisions:
                if (
                    decision.slot not in quorum_values
                    or quorum_values[decision.slot] != decision.value
                ):
                    raise PaxosSafetyViolation(
                        f"node {node_id!r} learned Multi-Paxos slot {decision.slot} "
                        "without matching durable quorum evidence"
                    )
                self._require_decision_evidence(decision)


class MultiPaxosNode:
    _PROMISED = "multipaxos_promised_ballot"
    _ACCEPTED = "multipaxos_accepted"
    _HISTORY = "multipaxos_accept_history"
    _LAST_ROUND = "multipaxos_last_round"
    _LEARNED = "multipaxos_learned"

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
        if cluster.learn_decisions:
            state.setdefault(self._LEARNED, {})
        self._leader_ballot: ProposalNumber | None = None
        self._promises: dict[str, LeaderPromise] = {}
        self._pending: dict[int, Any] = {}
        self._accepted_by: dict[int, set[str]] = {}
        self._recovery: DecisionRecoveryRequest | None = None
        self._recovery_generation = 0
        self._recovery_attempt = 0

    @property
    def promised_ballot(self) -> ProposalNumber | None:
        value = self.sim.persistent_state[self.node_id][self._PROMISED]
        assert value is None or isinstance(value, ProposalNumber)
        if value is not None and value.proposer_id not in self.cluster.nodes:
            raise PaxosSafetyViolation(
                "Multi-Paxos durable promise has non-member proposer "
                f"{value.proposer_id!r}"
            )
        return value

    @property
    def accept_history(self) -> tuple[SlotAcceptedValue, ...]:
        value = self.sim.persistent_state[self.node_id][self._HISTORY]
        assert isinstance(value, tuple)
        return value

    @property
    def learned_decisions(self) -> tuple[SlotLearnedDecision, ...]:
        learned = self.sim.persistent_state[self.node_id].get(self._LEARNED, {})
        assert isinstance(learned, dict)
        return tuple(learned[slot] for slot in sorted(learned))

    def learned_decision(self, slot: int) -> SlotLearnedDecision | None:
        learned = self.sim.persistent_state[self.node_id].get(self._LEARNED, {})
        assert isinstance(learned, dict)
        return learned.get(slot)

    def recover_decisions(self, start_slot: int) -> None:
        """Probe one bounded window, retrying at most three times this episode.

        Normal application passes call this at the first unlearned slot, including
        the unknown tail. Repeated passes coalesce while an episode is active and
        can start another after exhaustion. No request state needs to be durable.
        """
        if not _valid_slot(start_slot):
            raise ValueError("recovery start slot must be a positive integer")
        if not self.cluster.learn_decisions or not self.peers:
            return
        if not self.sim.is_alive(self.node_id):
            raise PaxosError("crashed node cannot recover Multi-Paxos decisions")
        request = DecisionRecoveryRequest(start_slot)
        if self._recovery == request:
            return
        self._recovery_generation += 1
        self._recovery_attempt = 0
        self._recovery = request
        self._retry_recovery()

    def _retry_recovery(self) -> None:
        request = self._recovery
        if request is None:
            return
        if self.learned_decision(request.start_slot) is not None:
            self._recovery = None
            return
        if self._recovery_attempt == RECOVERY_ATTEMPTS:
            self._recovery = None
            self.sim._record(
                "multipaxos-recovery-exhausted",
                node=self.node_id,
                start_slot=request.start_slot,
                attempts=self._recovery_attempt,
                known_gap=any(d.slot > request.start_slot for d in self.learned_decisions),
            )
            return
        self._recovery_attempt += 1
        for peer in sorted(self.peers):
            self.sim.send(self.node_id, peer, request)
        # Like other simulator-local timers, retry work is not a network packet.
        # Requests and responses above/below use the fault-injected transport.
        self.sim._schedule(
            Message(
                self.node_id,
                self.node_id,
                _DecisionRecoveryRetry(self._recovery_generation, self._recovery_attempt),
                ordinal=0,
            ),
            RECOVERY_RETRY_DELAY,
        )

    @property
    def leader_ballot(self) -> ProposalNumber | None:
        """This node's volatile Phase-1 ballot; authority must be checked separately."""
        return self._leader_ballot

    @property
    def leadership_accepted(self) -> tuple[SlotAcceptedValue, ...]:
        """Accepted records received locally in this ballot's Phase-1 promises."""
        return tuple(
            item for peer in sorted(self._promises) for item in self._promises[peer].accepted
        )

    @property
    def pending_proposals(self) -> tuple[LeaderAccept, ...]:
        if self._leader_ballot is None:
            return ()
        return tuple(
            LeaderAccept(slot, self._leader_ballot, self._pending[slot])
            for slot in sorted(self._pending)
        )

    def retry_proposal(self, slot: int) -> None:
        """Retransmit an active Phase-2 proposal without changing its ballot/value."""
        if (
            not self.sim.is_alive(self.node_id)
            or not self.has_leader_authority
            or self._leader_ballot != self.promised_ballot
        ):
            raise PaxosError("Multi-Paxos retry requires live Phase-1 authority")
        if slot not in self._pending:
            raise PaxosError("Multi-Paxos retry requires an active proposal")
        request = LeaderAccept(slot, self._leader_ballot, self._pending[slot])
        self._receive_accept(request)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)

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
        chosen_value = value if adopted is None else adopted.value
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

    def _adopted_value(self, slot: int) -> SlotAcceptedValue | None:
        """Return the accepted record so a null payload still counts as evidence."""
        accepted = [
            item
            for promise in self._promises.values()
            for item in promise.accepted
            if item.slot == slot
        ]
        if not accepted:
            return None
        highest = max(accepted, key=lambda item: item.proposal)
        values = [item.value for item in accepted if item.proposal == highest.proposal]
        first = values[0]
        if any(value != first for value in values[1:]):
            raise PaxosSafetyViolation(
                f"slot {slot} has conflicting accepted values at {highest.proposal!r}"
            )
        return highest

    def handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim or message.dst != self.node_id:
            raise ValueError("invalid Multi-Paxos message delivery")
        payload = message.payload
        if isinstance(payload, LeaderPrepare):
            if (
                message.src != payload.ballot.proposer_id
                or payload.ballot.proposer_id not in self.cluster.nodes
            ):
                return
            self._receive_prepare(payload)
        elif isinstance(payload, LeaderPromise):
            if message.src != payload.acceptor_id:
                return
            self._receive_promise(payload)
        elif isinstance(payload, LeaderAccept):
            if (
                message.src != payload.ballot.proposer_id
                or payload.ballot.proposer_id not in self.cluster.nodes
            ):
                return
            self._receive_accept(payload)
        elif isinstance(payload, LeaderAccepted):
            if message.src != payload.acceptor_id:
                return
            self._receive_accepted(payload)
        elif isinstance(payload, LeaderDecision):
            if (
                not isinstance(payload.ballot, ProposalNumber)
                or message.src != payload.ballot.proposer_id
                or payload.ballot.proposer_id not in self.cluster.nodes
            ):
                return
            self._receive_decision(payload)
        elif isinstance(payload, (DecisionRecoveryRequest, DecisionRecoveryResponse)):
            if not self.cluster.learn_decisions or message.src not in self.peers:
                return
            if isinstance(payload, DecisionRecoveryRequest):
                self._receive_recovery_request(message.src, payload)
            else:
                self._receive_recovery_response(payload)
        elif isinstance(payload, _DecisionRecoveryRetry):
            if (
                message.src == self.node_id
                and payload.generation == self._recovery_generation
                and payload.attempt == self._recovery_attempt
            ):
                self._retry_recovery()
        else:
            raise TypeError(f"unsupported Multi-Paxos message {type(payload).__name__}")

    def _receive_recovery_request(self, requester: str, request: DecisionRecoveryRequest) -> None:
        if not _valid_slot(request.start_slot):
            return
        # Only this peer's durable learned records may populate the response.
        # Never use cluster.chosen() or another node's state to fill the window.
        decisions = tuple(
            decision
            for slot in range(request.start_slot, request.start_slot + RECOVERY_WINDOW)
            if (decision := self.learned_decision(slot)) is not None
        )
        for decision in decisions:
            self.cluster._require_decision_evidence(decision)
        self.sim.send(
            self.node_id, requester, DecisionRecoveryResponse(request.start_slot, decisions)
        )

    def _receive_recovery_response(self, response: DecisionRecoveryResponse) -> None:
        if (
            not _valid_slot(response.start_slot)
            or not isinstance(response.decisions, tuple)
            or len(response.decisions) > RECOVERY_WINDOW
        ):
            return
        previous = response.start_slot - 1
        for decision in response.decisions:
            if (
                not isinstance(decision, SlotLearnedDecision)
                or not _valid_slot(decision.slot)
                or not previous < decision.slot < response.start_slot + RECOVERY_WINDOW
                or not isinstance(decision.proposal, ProposalNumber)
                or not _valid_slot(decision.proposal.round)
                or decision.proposal.proposer_id not in (*self.peers, self.node_id)
            ):
                return
            previous = decision.slot
        # Validate the entire response before writing any learned record. Delayed
        # responses need no live request or current leader ballot to remain valid.
        for decision in response.decisions:
            current = self.learned_decision(decision.slot)
            if current is not None and current.value != decision.value:
                raise PaxosSafetyViolation(
                    f"Multi-Paxos recovery observed conflicting decisions for slot {decision.slot}"
                )
            self.cluster._require_decision_evidence(decision)
        for decision in response.decisions:
            self._receive_decision(
                LeaderDecision(decision.slot, decision.proposal, decision.value)
            )

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
        for accepted in response.accepted:
            if (
                accepted.slot <= 0
                or accepted.proposal.proposer_id not in self.cluster.nodes
                or accepted.proposal > response.ballot
            ):
                self.sim._record(
                    "multipaxos-promise-invalid-evidence",
                    leader=self.node_id,
                    acceptor=response.acceptor_id,
                    ballot=response.ballot,
                    accepted=accepted,
                )
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
        if request.slot <= 0:
            self.sim._record(
                "multipaxos-accept-invalid-slot",
                slot=request.slot,
                acceptor=self.node_id,
                ballot=request.ballot,
            )
            return
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
        if response.slot <= 0:
            return
        if (
            response.ballot != self._leader_ballot
            or response.slot not in self._pending
            or response.acceptor_id not in self.cluster.nodes
        ):
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
            if self.cluster.learn_decisions:
                decision = LeaderDecision(response.slot, response.ballot, response.value)
                self._receive_decision(decision)
                for peer in self.peers:
                    self.sim.send(self.node_id, peer, decision)

    def _receive_decision(self, decision: LeaderDecision) -> None:
        learned = self.sim.persistent_state[self.node_id].get(self._LEARNED)
        if not _valid_slot(decision.slot) or learned is None:
            self.sim._record(
                "multipaxos-decision-ignored",
                node=self.node_id,
                slot=decision.slot,
                ballot=decision.ballot,
            )
            return
        current = learned.get(decision.slot)
        if current is not None and current.value != decision.value:
            raise PaxosSafetyViolation(
                f"Multi-Paxos learner {self.node_id!r} observed conflicting "
                f"decisions for slot {decision.slot}"
            )
        record = SlotLearnedDecision(decision.slot, decision.ballot, decision.value)
        self.cluster._require_decision_evidence(record)
        if current is not None:
            self.sim._record(
                "multipaxos-decision-duplicate",
                node=self.node_id,
                slot=decision.slot,
                ballot=decision.ballot,
            )
            return
        learned[decision.slot] = record
        self.sim._record(
            "multipaxos-learned",
            node=self.node_id,
            slot=decision.slot,
            ballot=decision.ballot,
            value=decision.value,
        )

    def _validate_durable_learned(self) -> None:
        state = self.sim.persistent_state[self.node_id]
        if self._LEARNED not in state:
            return
        learned = state[self._LEARNED]
        if not isinstance(learned, dict):
            raise PaxosSafetyViolation("Multi-Paxos durable learned decisions are invalid")
        for slot, decision in learned.items():
            if (
                not _valid_slot(slot)
                or not isinstance(decision, SlotLearnedDecision)
                or not _valid_slot(decision.slot)
                or decision.slot != slot
            ):
                raise PaxosSafetyViolation(
                    f"Multi-Paxos durable learned decision for slot {slot!r} is invalid"
                )
            self.cluster._validate_ballot(decision.proposal)

    def handle_crash(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Multi-Paxos crash from different simulator")
        self._leader_ballot = None
        self._promises.clear()
        self._pending.clear()
        self._accepted_by.clear()
        self._recovery = None
        self._recovery_generation += 1
        self._recovery_attempt = 0

    def handle_restart(self, sim: Simulator) -> None:
        self.handle_crash(sim)
        self.sim._record("multipaxos-node-restart", node=self.node_id)
