from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .paxos import PaxosError, PaxosProposalActive, PaxosSafetyViolation, ProposalNumber
from .simulator import Message, Simulator


@dataclass(frozen=True, slots=True)
class SlotAcceptedValue:
    slot: int
    proposal: ProposalNumber
    value: Any

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("Paxos log slot must be positive")


@dataclass(frozen=True, slots=True)
class SlotLearnedDecision:
    slot: int
    proposal: ProposalNumber
    value: Any

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("Paxos log slot must be positive")


@dataclass(frozen=True, slots=True)
class SlotPrepare:
    slot: int
    proposal: ProposalNumber
    proposer_id: str

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("Paxos log slot must be positive")
        if self.proposer_id != self.proposal.proposer_id:
            raise ValueError("prepare proposer must match proposal number owner")


@dataclass(frozen=True, slots=True)
class SlotPromise:
    slot: int
    proposal: ProposalNumber
    acceptor_id: str
    accepted: SlotAcceptedValue | None


@dataclass(frozen=True, slots=True)
class SlotAcceptRequest:
    slot: int
    proposal: ProposalNumber
    proposer_id: str
    value: Any

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("Paxos log slot must be positive")
        if self.proposer_id != self.proposal.proposer_id:
            raise ValueError("accept proposer must match proposal number owner")


@dataclass(frozen=True, slots=True)
class SlotAccepted:
    slot: int
    proposal: ProposalNumber
    acceptor_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class SlotLearn:
    slot: int
    proposal: ProposalNumber
    proposer_id: str
    value: Any


@dataclass(slots=True)
class _SlotProposalState:
    slot: int
    proposal: ProposalNumber
    original_value: Any
    promises: dict[str, SlotPromise]
    accepted_by: set[str]
    chosen_value: Any = None
    accept_phase_started: bool = False


class PaxosLogCluster:
    """Independent Paxos decrees composed into an ordered replicated log."""

    def __init__(self, sim: Simulator, node_ids: tuple[str, ...]) -> None:
        if not node_ids:
            raise ValueError("Paxos log cluster requires at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Paxos log node ids must be unique")
        self.sim = sim
        self.node_ids = node_ids
        self.nodes = {
            node_id: PaxosLogNode(
                cluster=self,
                node_id=node_id,
                peers=tuple(peer for peer in node_ids if peer != node_id),
            )
            for node_id in node_ids
        }
        self._chosen: dict[int, SlotLearnedDecision] = {}
        self._chosen_acceptors: dict[int, tuple[str, ...]] = {}
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

    def node(self, node_id: str) -> PaxosLogNode:
        return self.nodes[node_id]

    def chosen(self, slot: int) -> SlotLearnedDecision | None:
        return self._chosen.get(slot)

    def chosen_acceptors(self, slot: int) -> tuple[str, ...]:
        return self._chosen_acceptors.get(slot, ())

    def chosen_prefix(self) -> tuple[SlotLearnedDecision, ...]:
        prefix: list[SlotLearnedDecision] = []
        slot = 1
        while (decision := self._chosen.get(slot)) is not None:
            prefix.append(decision)
            slot += 1
        return tuple(prefix)

    def _record_chosen(
        self,
        slot: int,
        proposal: ProposalNumber,
        value: Any,
        acceptors: set[str],
    ) -> None:
        if len(acceptors) < self.quorum_size:
            raise PaxosSafetyViolation("Paxos log decision requires a majority")
        current = self._chosen.get(slot)
        if current is not None and current.value != value:
            raise PaxosSafetyViolation(
                f"Paxos log slot {slot} chose conflicting values "
                f"{current.value!r} and {value!r}"
            )
        if current is None:
            self._chosen[slot] = SlotLearnedDecision(slot, proposal, value)
            self._chosen_acceptors[slot] = tuple(sorted(acceptors))
            self.sim._record(
                "paxos-log-chosen",
                slot=slot,
                proposal=proposal,
                value=value,
                acceptors=self._chosen_acceptors[slot],
            )

    def assert_safety(self) -> None:
        PaxosLogSafetyHarness(self).checkpoint()


class PaxosLogNode:
    _PROMISED = "paxos_log_promised"
    _ACCEPTED = "paxos_log_accepted"
    _ACCEPT_HISTORY = "paxos_log_accept_history"
    _LEARNED = "paxos_log_learned"
    _LAST_PROPOSAL_ROUND = "paxos_log_last_proposal_round"

    def __init__(
        self,
        *,
        cluster: PaxosLogCluster,
        node_id: str,
        peers: tuple[str, ...],
    ) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
        self._proposals: dict[int, _SlotProposalState] = {}
        state = self.sim.persistent_state[node_id]
        state.setdefault(self._PROMISED, {})
        state.setdefault(self._ACCEPTED, {})
        state.setdefault(self._ACCEPT_HISTORY, ())
        state.setdefault(self._LEARNED, {})
        state.setdefault(self._LAST_PROPOSAL_ROUND, 0)
        self._validate_durable_state()

    def promised_proposal(self, slot: int) -> ProposalNumber | None:
        self._require_slot(slot)
        promised = self.sim.persistent_state[self.node_id][self._PROMISED]
        assert isinstance(promised, dict)
        value = promised.get(slot)
        assert value is None or isinstance(value, ProposalNumber)
        return value

    def accepted_value(self, slot: int) -> SlotAcceptedValue | None:
        self._require_slot(slot)
        accepted = self.sim.persistent_state[self.node_id][self._ACCEPTED]
        assert isinstance(accepted, dict)
        value = accepted.get(slot)
        assert value is None or isinstance(value, SlotAcceptedValue)
        return value

    def learned_decision(self, slot: int) -> SlotLearnedDecision | None:
        self._require_slot(slot)
        learned = self.sim.persistent_state[self.node_id][self._LEARNED]
        assert isinstance(learned, dict)
        value = learned.get(slot)
        assert value is None or isinstance(value, SlotLearnedDecision)
        return value

    @property
    def accept_history(self) -> tuple[SlotAcceptedValue, ...]:
        value = self.sim.persistent_state[self.node_id][self._ACCEPT_HISTORY]
        assert isinstance(value, tuple)
        return value

    def learned_prefix(self) -> tuple[SlotLearnedDecision, ...]:
        prefix: list[SlotLearnedDecision] = []
        slot = 1
        while (decision := self.learned_decision(slot)) is not None:
            prefix.append(decision)
            slot += 1
        return tuple(prefix)

    def start_proposal(
        self,
        slot: int,
        value: Any,
        *,
        round_number: int | None = None,
    ) -> ProposalNumber:
        self._require_slot(slot)
        if not self.sim.is_alive(self.node_id):
            raise PaxosError(f"crashed proposer {self.node_id!r} cannot start Paxos log")
        if slot in self._proposals:
            raise PaxosProposalActive(
                f"proposer {self.node_id!r} already has active proposal for slot {slot}"
            )

        persistent = self.sim.persistent_state[self.node_id]
        last_round = int(persistent[self._LAST_PROPOSAL_ROUND])
        if round_number is None:
            known_rounds = [last_round]
            for proposal in persistent[self._PROMISED].values():
                known_rounds.append(proposal.round)
            for accepted in persistent[self._ACCEPTED].values():
                known_rounds.append(accepted.proposal.round)
            for learned in persistent[self._LEARNED].values():
                known_rounds.append(learned.proposal.round)
            round_number = max(known_rounds) + 1
        if round_number <= last_round:
            raise ValueError(
                f"proposer round {round_number} must exceed durable local round {last_round}"
            )

        proposal = ProposalNumber(round_number, self.node_id)
        persistent[self._LAST_PROPOSAL_ROUND] = round_number
        self._proposals[slot] = _SlotProposalState(
            slot=slot,
            proposal=proposal,
            original_value=value,
            promises={},
            accepted_by=set(),
        )
        self.sim._record(
            "paxos-log-proposal-start",
            slot=slot,
            proposer=self.node_id,
            proposal=proposal,
            value=value,
        )
        prepare = SlotPrepare(slot, proposal, self.node_id)
        self._receive_prepare(prepare)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, prepare)
        return proposal

    def handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos log node invoked by a different simulator")
        payload = message.payload
        if message.dst != self.node_id:
            self._reject_source(message, expected=None, reason="destination-identity-mismatch")
            return

        expected: str
        if isinstance(payload, SlotPrepare):
            expected = payload.proposer_id
            if expected not in self.cluster.nodes:
                self._reject_source(message, expected, reason="unknown-proposer")
                return
            if message.src != expected:
                self._reject_source(message, expected)
                return
            self._receive_prepare(payload)
        elif isinstance(payload, SlotPromise):
            expected = payload.acceptor_id
            if expected not in self.cluster.nodes:
                self._reject_source(message, expected, reason="unknown-acceptor")
                return
            if message.src != expected:
                self._reject_source(message, expected)
                return
            self._receive_promise(payload)
        elif isinstance(payload, SlotAcceptRequest):
            expected = payload.proposer_id
            if expected not in self.cluster.nodes:
                self._reject_source(message, expected, reason="unknown-proposer")
                return
            if message.src != expected:
                self._reject_source(message, expected)
                return
            self._receive_accept_request(payload)
        elif isinstance(payload, SlotAccepted):
            expected = payload.acceptor_id
            if expected not in self.cluster.nodes:
                self._reject_source(message, expected, reason="unknown-acceptor")
                return
            if message.src != expected:
                self._reject_source(message, expected)
                return
            self._receive_accepted(payload)
        elif isinstance(payload, SlotLearn):
            expected = payload.proposer_id
            if expected not in self.cluster.nodes:
                self._reject_source(message, expected, reason="unknown-proposer")
                return
            if message.src != expected:
                self._reject_source(message, expected)
                return
            self._receive_learn(payload)
        else:
            raise TypeError(f"unsupported Paxos log message {type(payload).__name__}")

    def handle_crash(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos log crash invoked by a different simulator")
        active = tuple(
            (slot, state.proposal)
            for slot, state in sorted(self._proposals.items())
        )
        self._proposals.clear()
        self.sim._record("paxos-log-node-crash", node=self.node_id, active=active)

    def handle_restart(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos log restart invoked by a different simulator")
        self._proposals.clear()
        self._validate_durable_state()
        self.sim._record(
            "paxos-log-node-restart",
            node=self.node_id,
            learned_prefix=self.learned_prefix(),
        )

    def _receive_prepare(self, request: SlotPrepare) -> None:
        promised = self.promised_proposal(request.slot)
        if promised is not None and request.proposal < promised:
            self.sim._record(
                "paxos-log-prepare-rejected",
                slot=request.slot,
                acceptor=self.node_id,
                proposal=request.proposal,
                promised=promised,
            )
            return

        state = self.sim.persistent_state[self.node_id]
        if promised is None or request.proposal > promised:
            state[self._PROMISED][request.slot] = request.proposal
        promise = SlotPromise(
            request.slot,
            request.proposal,
            self.node_id,
            self.accepted_value(request.slot),
        )
        self.sim._record(
            "paxos-log-promise",
            slot=request.slot,
            acceptor=self.node_id,
            proposer=request.proposer_id,
            proposal=request.proposal,
            accepted=promise.accepted,
        )
        if request.proposer_id == self.node_id:
            self._receive_promise(promise)
        else:
            self.sim.send(self.node_id, request.proposer_id, promise)

    def _receive_promise(self, response: SlotPromise) -> None:
        state = self._proposals.get(response.slot)
        if state is None or response.proposal != state.proposal:
            self.sim._record(
                "paxos-log-stale-promise",
                slot=response.slot,
                proposer=self.node_id,
                acceptor=response.acceptor_id,
                proposal=response.proposal,
            )
            return
        if response.acceptor_id not in self.cluster.nodes:
            raise PaxosSafetyViolation("Paxos log promise references unknown acceptor")
        state.promises.setdefault(response.acceptor_id, response)
        if state.accept_phase_started or len(state.promises) < self.cluster.quorum_size:
            return

        accepted = [
            promise.accepted
            for promise in state.promises.values()
            if promise.accepted is not None
        ]
        if accepted:
            highest = max(item.proposal for item in accepted)
            highest_values = [item.value for item in accepted if item.proposal == highest]
            first = highest_values[0]
            if any(value != first for value in highest_values[1:]):
                raise PaxosSafetyViolation(
                    f"slot {response.slot} proposal {highest!r} has conflicting values"
                )
            chosen_value = first
        else:
            chosen_value = state.original_value

        state.chosen_value = chosen_value
        state.accept_phase_started = True
        self.sim._record(
            "paxos-log-accept-phase-start",
            slot=state.slot,
            proposer=self.node_id,
            proposal=state.proposal,
            value=chosen_value,
            promises=tuple(sorted(state.promises)),
        )
        request = SlotAcceptRequest(state.slot, state.proposal, self.node_id, chosen_value)
        self._receive_accept_request(request)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)

    def _receive_accept_request(self, request: SlotAcceptRequest) -> None:
        promised = self.promised_proposal(request.slot)
        if promised is not None and request.proposal < promised:
            self.sim._record(
                "paxos-log-accept-rejected",
                slot=request.slot,
                acceptor=self.node_id,
                proposal=request.proposal,
                promised=promised,
            )
            return

        current = self.accepted_value(request.slot)
        if (
            current is not None
            and current.proposal == request.proposal
            and current.value != request.value
        ):
            raise PaxosSafetyViolation(
                f"slot {request.slot} proposal {request.proposal!r} "
                "attempted conflicting values"
            )

        persistent = self.sim.persistent_state[self.node_id]
        persistent[self._PROMISED][request.slot] = request.proposal
        accepted = SlotAcceptedValue(request.slot, request.proposal, request.value)
        persistent[self._ACCEPTED][request.slot] = accepted
        if accepted not in self.accept_history:
            persistent[self._ACCEPT_HISTORY] = (*self.accept_history, accepted)
        self.sim._record(
            "paxos-log-accepted",
            slot=request.slot,
            acceptor=self.node_id,
            proposer=request.proposer_id,
            proposal=request.proposal,
            value=request.value,
        )
        response = SlotAccepted(
            request.slot,
            request.proposal,
            self.node_id,
            request.value,
        )
        if request.proposer_id == self.node_id:
            self._receive_accepted(response)
        else:
            self.sim.send(self.node_id, request.proposer_id, response)

    def _receive_accepted(self, response: SlotAccepted) -> None:
        state = self._proposals.get(response.slot)
        if (
            state is None
            or response.proposal != state.proposal
            or not state.accept_phase_started
        ):
            self.sim._record(
                "paxos-log-stale-accepted",
                slot=response.slot,
                proposer=self.node_id,
                acceptor=response.acceptor_id,
                proposal=response.proposal,
            )
            return
        if response.value != state.chosen_value:
            raise PaxosSafetyViolation("Paxos log accepted response changed value")
        state.accepted_by.add(response.acceptor_id)
        if len(state.accepted_by) < self.cluster.quorum_size:
            return

        slot = state.slot
        proposal = state.proposal
        value = state.chosen_value
        acceptors = set(state.accepted_by)
        self.cluster._record_chosen(slot, proposal, value, acceptors)
        learn = SlotLearn(slot, proposal, self.node_id, value)
        self._receive_learn(learn)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, learn)
        self._proposals.pop(slot, None)

    def _receive_learn(self, message: SlotLearn) -> None:
        current = self.learned_decision(message.slot)
        if current is not None and current.value != message.value:
            raise PaxosSafetyViolation(
                f"learner {self.node_id!r} observed conflicting values "
                f"for slot {message.slot}"
            )
        if current is None:
            decision = SlotLearnedDecision(
                message.slot,
                message.proposal,
                message.value,
            )
            self.sim.persistent_state[self.node_id][self._LEARNED][
                message.slot
            ] = decision
            self.sim._record(
                "paxos-log-learned",
                slot=message.slot,
                learner=self.node_id,
                proposal=message.proposal,
                value=message.value,
                prefix_length=len(self.learned_prefix()),
            )

    def _reject_source(
        self,
        message: Message,
        expected: str | None,
        *,
        reason: str = "source-identity-mismatch",
    ) -> None:
        self.sim._record(
            "paxos-log-envelope-rejected",
            node=self.node_id,
            src=message.src,
            dst=message.dst,
            expected_src=expected,
            reason=reason,
            payload=message.payload,
        )

    def _validate_durable_state(self) -> None:
        state = self.sim.persistent_state[self.node_id]
        promised = state[self._PROMISED]
        accepted = state[self._ACCEPTED]
        history = state[self._ACCEPT_HISTORY]
        learned = state[self._LEARNED]
        last_round = state[self._LAST_PROPOSAL_ROUND]
        if not isinstance(promised, dict) or not all(
            isinstance(slot, int)
            and slot > 0
            and isinstance(proposal, ProposalNumber)
            for slot, proposal in promised.items()
        ):
            raise TypeError("durable Paxos log promises must map slots to ProposalNumber")
        if not isinstance(accepted, dict) or not all(
            isinstance(slot, int)
            and slot > 0
            and isinstance(value, SlotAcceptedValue)
            and value.slot == slot
            for slot, value in accepted.items()
        ):
            raise TypeError("durable Paxos log accepted state is invalid")
        if not isinstance(history, tuple) or not all(
            isinstance(item, SlotAcceptedValue) for item in history
        ):
            raise TypeError("durable Paxos log accept history is invalid")
        if not isinstance(learned, dict) or not all(
            isinstance(slot, int)
            and slot > 0
            and isinstance(value, SlotLearnedDecision)
            and value.slot == slot
            for slot, value in learned.items()
        ):
            raise TypeError("durable Paxos log learned state is invalid")
        if not isinstance(last_round, int) or last_round < 0:
            raise TypeError("durable Paxos log proposer round must be non-negative")

        history_by_pair: dict[tuple[int, ProposalNumber], Any] = {}
        for item in history:
            key = (item.slot, item.proposal)
            if key in history_by_pair and history_by_pair[key] != item.value:
                raise PaxosSafetyViolation(
                    f"durable Paxos log history conflicts at slot {item.slot} "
                    f"proposal {item.proposal!r}"
                )
            history_by_pair[key] = item.value
        for slot, current in accepted.items():
            if current not in history:
                raise PaxosSafetyViolation(
                    f"current accepted value for slot {slot} missing from durable history"
                )
            promised_value = promised.get(slot)
            if promised_value is not None and current.proposal > promised_value:
                raise PaxosSafetyViolation(
                    f"accepted proposal for slot {slot} exceeds durable promise"
                )

    @staticmethod
    def _require_slot(slot: int) -> None:
        if not isinstance(slot, int) or isinstance(slot, bool) or slot <= 0:
            raise ValueError("Paxos log slot must be a positive integer")


class PaxosLogSafetyHarness:
    """Reconstruct per-slot chosen evidence and ordered learned-prefix safety."""

    def __init__(self, cluster: PaxosLogCluster) -> None:
        self.cluster = cluster

    def checkpoint(self) -> None:
        quorum = self.cluster.quorum_size
        evidence: list[tuple[int, ProposalNumber, Any, set[str]]] = []
        proposal_values: list[tuple[int, ProposalNumber, Any]] = []

        for node_id in self.cluster.node_ids:
            node = self.cluster.node(node_id)
            node._validate_durable_state()
            for accepted in node.accept_history:
                matching_proposals = [
                    value
                    for slot, proposal, value in proposal_values
                    if slot == accepted.slot and proposal == accepted.proposal
                ]
                if matching_proposals and matching_proposals[0] != accepted.value:
                    raise PaxosSafetyViolation(
                        f"slot {accepted.slot} proposal {accepted.proposal!r} "
                        "has conflicting accepted values"
                    )
                if not matching_proposals:
                    proposal_values.append(
                        (accepted.slot, accepted.proposal, accepted.value)
                    )

                for item in evidence:
                    slot, proposal, value, acceptors = item
                    if (
                        slot == accepted.slot
                        and proposal == accepted.proposal
                        and value == accepted.value
                    ):
                        acceptors.add(node_id)
                        break
                else:
                    evidence.append(
                        (
                            accepted.slot,
                            accepted.proposal,
                            accepted.value,
                            {node_id},
                        )
                    )

        chosen_by_slot: dict[
            int, list[tuple[ProposalNumber, Any, tuple[str, ...]]]
        ] = {}
        for slot, proposal, value, acceptors in evidence:
            if len(acceptors) >= quorum:
                chosen_by_slot.setdefault(slot, []).append(
                    (proposal, value, tuple(sorted(acceptors)))
                )

        for slot, chosen in chosen_by_slot.items():
            first_value = chosen[0][1]
            if any(value != first_value for _, value, _ in chosen[1:]):
                raise PaxosSafetyViolation(
                    f"durable quorum evidence chose multiple values for slot {slot}: "
                    f"{chosen!r}"
                )

        for node_id in self.cluster.node_ids:
            node = self.cluster.node(node_id)
            learned = self.cluster.sim.persistent_state[node_id][node._LEARNED]
            assert isinstance(learned, dict)
            for slot, decision in learned.items():
                chosen = chosen_by_slot.get(slot, [])
                if not chosen or chosen[0][1] != decision.value:
                    raise PaxosSafetyViolation(
                        f"node {node_id!r} learned slot {slot} without matching "
                        "durable majority acceptance"
                    )

        for slot, objective in self.cluster._chosen.items():
            chosen = chosen_by_slot.get(slot, [])
            if not chosen or chosen[0][1] != objective.value:
                raise PaxosSafetyViolation(
                    f"objective chosen slot {slot} lacks durable majority evidence"
                )

        prefixes = {
            node_id: tuple(
                decision.value
                for decision in self.cluster.node(node_id).learned_prefix()
            )
            for node_id in self.cluster.node_ids
        }
        for left_id, left in prefixes.items():
            for right_id, right in prefixes.items():
                common = min(len(left), len(right))
                if left[:common] != right[:common]:
                    raise PaxosSafetyViolation(
                        "learned Paxos log prefixes diverged between "
                        f"{left_id!r} and {right_id!r}"
                    )
