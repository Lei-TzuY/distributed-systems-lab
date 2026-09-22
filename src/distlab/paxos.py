from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .simulator import Message, Simulator


class PaxosError(RuntimeError):
    """Base error for deterministic single-decree Paxos failures."""


class PaxosProposalActive(PaxosError):
    """Raised when one proposer tries to overlap two local proposals."""


class PaxosSafetyViolation(AssertionError):
    """Raised when durable Paxos evidence permits two chosen values."""


@dataclass(frozen=True, order=True, slots=True)
class ProposalNumber:
    round: int
    proposer_id: str

    def __post_init__(self) -> None:
        if self.round <= 0:
            raise ValueError("Paxos proposal round must be positive")
        if not self.proposer_id:
            raise ValueError("Paxos proposer_id must be non-empty")


@dataclass(frozen=True, slots=True)
class AcceptedValue:
    proposal: ProposalNumber
    value: Any


@dataclass(frozen=True, slots=True)
class LearnedDecision:
    proposal: ProposalNumber
    value: Any


@dataclass(frozen=True, slots=True)
class Prepare:
    proposal: ProposalNumber
    proposer_id: str

    def __post_init__(self) -> None:
        if self.proposer_id != self.proposal.proposer_id:
            raise ValueError("prepare proposer must match proposal number owner")


@dataclass(frozen=True, slots=True)
class Promise:
    proposal: ProposalNumber
    acceptor_id: str
    accepted: AcceptedValue | None


@dataclass(frozen=True, slots=True)
class AcceptRequest:
    proposal: ProposalNumber
    proposer_id: str
    value: Any

    def __post_init__(self) -> None:
        if self.proposer_id != self.proposal.proposer_id:
            raise ValueError("accept proposer must match proposal number owner")


@dataclass(frozen=True, slots=True)
class Accepted:
    proposal: ProposalNumber
    acceptor_id: str
    value: Any


@dataclass(frozen=True, slots=True)
class Learn:
    proposal: ProposalNumber
    proposer_id: str
    value: Any


@dataclass(slots=True)
class _ProposalState:
    proposal: ProposalNumber
    original_value: Any
    promises: dict[str, Promise]
    accepted_by: set[str]
    chosen_value: Any = None
    accept_phase_started: bool = False


class PaxosCluster:
    """One deterministic single-decree Paxos instance over the shared simulator."""

    def __init__(self, sim: Simulator, node_ids: tuple[str, ...]) -> None:
        if not node_ids:
            raise ValueError("Paxos cluster requires at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Paxos node ids must be unique")
        self.sim = sim
        self.node_ids = node_ids
        self.nodes = {
            node_id: PaxosNode(
                cluster=self,
                node_id=node_id,
                peers=tuple(peer for peer in node_ids if peer != node_id),
            )
            for node_id in node_ids
        }
        self._chosen: LearnedDecision | None = None
        self._chosen_acceptors: tuple[str, ...] = ()
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

    @property
    def chosen(self) -> LearnedDecision | None:
        return self._chosen

    @property
    def chosen_acceptors(self) -> tuple[str, ...]:
        return self._chosen_acceptors

    def node(self, node_id: str) -> PaxosNode:
        return self.nodes[node_id]

    def _record_chosen(
        self,
        proposal: ProposalNumber,
        value: Any,
        acceptors: set[str],
    ) -> None:
        if len(acceptors) < self.quorum_size:
            raise PaxosSafetyViolation("Paxos decision requires a majority of acceptors")
        current = self._chosen
        if current is not None and current.value != value:
            raise PaxosSafetyViolation(
                f"Paxos chose conflicting values {current.value!r} and {value!r}"
            )
        if current is None:
            self._chosen = LearnedDecision(proposal, value)
            self._chosen_acceptors = tuple(sorted(acceptors))
            self.sim._record(
                "paxos-chosen",
                proposal=proposal,
                value=value,
                acceptors=self._chosen_acceptors,
            )

    def assert_safety(self) -> None:
        PaxosSafetyHarness(self).checkpoint()


class PaxosNode:
    """Crash-recoverable proposer, acceptor, and learner for one Paxos decree."""

    _PROMISED = "paxos_promised"
    _ACCEPTED = "paxos_accepted"
    _ACCEPT_HISTORY = "paxos_accept_history"
    _LEARNED = "paxos_learned"
    _LAST_PROPOSAL_ROUND = "paxos_last_proposal_round"

    def __init__(
        self,
        *,
        cluster: PaxosCluster,
        node_id: str,
        peers: tuple[str, ...],
    ) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
        self._proposal: _ProposalState | None = None
        state = self.sim.persistent_state[node_id]
        state.setdefault(self._PROMISED, None)
        state.setdefault(self._ACCEPTED, None)
        state.setdefault(self._ACCEPT_HISTORY, ())
        state.setdefault(self._LEARNED, None)
        state.setdefault(self._LAST_PROPOSAL_ROUND, 0)
        self._validate_durable_state()

    @property
    def promised_proposal(self) -> ProposalNumber | None:
        value = self.sim.persistent_state[self.node_id][self._PROMISED]
        assert value is None or isinstance(value, ProposalNumber)
        return value

    @property
    def accepted_value(self) -> AcceptedValue | None:
        value = self.sim.persistent_state[self.node_id][self._ACCEPTED]
        assert value is None or isinstance(value, AcceptedValue)
        return value

    @property
    def learned_decision(self) -> LearnedDecision | None:
        value = self.sim.persistent_state[self.node_id][self._LEARNED]
        assert value is None or isinstance(value, LearnedDecision)
        return value

    @property
    def accept_history(self) -> tuple[AcceptedValue, ...]:
        value = self.sim.persistent_state[self.node_id][self._ACCEPT_HISTORY]
        assert isinstance(value, tuple)
        return value

    def start_proposal(self, value: Any, *, round_number: int | None = None) -> ProposalNumber:
        if not self.sim.is_alive(self.node_id):
            raise PaxosError(f"crashed proposer {self.node_id!r} cannot start Paxos")
        if self._proposal is not None:
            raise PaxosProposalActive(
                f"proposer {self.node_id!r} already has proposal {self._proposal.proposal!r}"
            )

        persistent = self.sim.persistent_state[self.node_id]
        last_round = int(persistent[self._LAST_PROPOSAL_ROUND])
        if round_number is None:
            known_rounds = [last_round]
            if self.promised_proposal is not None:
                known_rounds.append(self.promised_proposal.round)
            if self.accepted_value is not None:
                known_rounds.append(self.accepted_value.proposal.round)
            if self.learned_decision is not None:
                known_rounds.append(self.learned_decision.proposal.round)
            round_number = max(known_rounds) + 1
        if round_number <= last_round:
            raise ValueError(
                f"proposer round {round_number} must exceed durable local round {last_round}"
            )

        proposal = ProposalNumber(round_number, self.node_id)
        persistent[self._LAST_PROPOSAL_ROUND] = round_number
        self._proposal = _ProposalState(
            proposal=proposal,
            original_value=value,
            promises={},
            accepted_by=set(),
        )
        self.sim._record(
            "paxos-proposal-start",
            proposer=self.node_id,
            proposal=proposal,
            value=value,
        )

        prepare = Prepare(proposal, self.node_id)
        self._receive_prepare(prepare)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, prepare)
        return proposal

    def handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos node invoked by a different simulator")
        payload = message.payload
        if message.dst != self.node_id:
            self.sim._record(
                "paxos-envelope-rejected",
                node=self.node_id,
                src=message.src,
                dst=message.dst,
                reason="destination-identity-mismatch",
                payload=payload,
            )
            return

        if isinstance(payload, Prepare):
            if message.src != payload.proposer_id:
                self._reject_source(message, payload.proposer_id)
                return
            self._receive_prepare(payload)
            return
        if isinstance(payload, Promise):
            if message.src != payload.acceptor_id:
                self._reject_source(message, payload.acceptor_id)
                return
            self._receive_promise(payload)
            return
        if isinstance(payload, AcceptRequest):
            if message.src != payload.proposer_id:
                self._reject_source(message, payload.proposer_id)
                return
            self._receive_accept_request(payload)
            return
        if isinstance(payload, Accepted):
            if message.src != payload.acceptor_id:
                self._reject_source(message, payload.acceptor_id)
                return
            self._receive_accepted(payload)
            return
        if isinstance(payload, Learn):
            if message.src != payload.proposer_id:
                self._reject_source(message, payload.proposer_id)
                return
            self._receive_learn(payload)
            return
        raise TypeError(f"unsupported Paxos message {type(payload).__name__}")

    def handle_crash(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos crash invoked by a different simulator")
        active = self._proposal.proposal if self._proposal is not None else None
        self._proposal = None
        self.sim._record("paxos-node-crash", node=self.node_id, active_proposal=active)

    def handle_restart(self, sim: Simulator) -> None:
        if sim is not self.sim:
            raise ValueError("Paxos restart invoked by a different simulator")
        self._proposal = None
        self._validate_durable_state()
        self.sim._record(
            "paxos-node-restart",
            node=self.node_id,
            promised=self.promised_proposal,
            accepted=self.accepted_value,
            learned=self.learned_decision,
        )

    def _receive_prepare(self, request: Prepare) -> None:
        promised = self.promised_proposal
        if promised is not None and request.proposal < promised:
            self.sim._record(
                "paxos-prepare-rejected",
                acceptor=self.node_id,
                proposal=request.proposal,
                promised=promised,
            )
            return

        state = self.sim.persistent_state[self.node_id]
        if promised is None or request.proposal > promised:
            state[self._PROMISED] = request.proposal
        promise = Promise(
            proposal=request.proposal,
            acceptor_id=self.node_id,
            accepted=self.accepted_value,
        )
        self.sim._record(
            "paxos-promise",
            acceptor=self.node_id,
            proposer=request.proposer_id,
            proposal=request.proposal,
            accepted=promise.accepted,
        )
        if request.proposer_id == self.node_id:
            self._receive_promise(promise)
        else:
            self.sim.send(self.node_id, request.proposer_id, promise)

    def _receive_promise(self, response: Promise) -> None:
        state = self._proposal
        if state is None or response.proposal != state.proposal:
            self.sim._record(
                "paxos-stale-promise",
                proposer=self.node_id,
                acceptor=response.acceptor_id,
                proposal=response.proposal,
            )
            return
        if response.acceptor_id not in self.cluster.nodes:
            raise PaxosSafetyViolation("promise references unknown acceptor")
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
                    f"proposal {highest!r} was accepted with conflicting values"
                )
            chosen_value = first
        else:
            chosen_value = state.original_value

        state.chosen_value = chosen_value
        state.accept_phase_started = True
        self.sim._record(
            "paxos-accept-phase-start",
            proposer=self.node_id,
            proposal=state.proposal,
            value=chosen_value,
            promises=tuple(sorted(state.promises)),
        )
        request = AcceptRequest(state.proposal, self.node_id, chosen_value)
        self._receive_accept_request(request)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)

    def _receive_accept_request(self, request: AcceptRequest) -> None:
        promised = self.promised_proposal
        if promised is not None and request.proposal < promised:
            self.sim._record(
                "paxos-accept-rejected",
                acceptor=self.node_id,
                proposal=request.proposal,
                promised=promised,
            )
            return

        current = self.accepted_value
        if (
            current is not None
            and current.proposal == request.proposal
            and current.value != request.value
        ):
            raise PaxosSafetyViolation(
                f"proposal {request.proposal!r} attempted conflicting accepted values"
            )

        state = self.sim.persistent_state[self.node_id]
        state[self._PROMISED] = request.proposal
        accepted = AcceptedValue(request.proposal, request.value)
        state[self._ACCEPTED] = accepted
        history = self.accept_history
        if accepted not in history:
            state[self._ACCEPT_HISTORY] = (*history, accepted)
        self.sim._record(
            "paxos-accepted",
            acceptor=self.node_id,
            proposer=request.proposer_id,
            proposal=request.proposal,
            value=request.value,
        )

        response = Accepted(request.proposal, self.node_id, request.value)
        if request.proposer_id == self.node_id:
            self._receive_accepted(response)
        else:
            self.sim.send(self.node_id, request.proposer_id, response)

    def _receive_accepted(self, response: Accepted) -> None:
        state = self._proposal
        if (
            state is None
            or response.proposal != state.proposal
            or not state.accept_phase_started
        ):
            self.sim._record(
                "paxos-stale-accepted",
                proposer=self.node_id,
                acceptor=response.acceptor_id,
                proposal=response.proposal,
            )
            return
        if response.value != state.chosen_value:
            raise PaxosSafetyViolation("accepted response changed the proposal value")
        state.accepted_by.add(response.acceptor_id)
        if len(state.accepted_by) < self.cluster.quorum_size:
            return

        proposal = state.proposal
        value = state.chosen_value
        acceptors = set(state.accepted_by)
        self.cluster._record_chosen(proposal, value, acceptors)
        learn = Learn(proposal, self.node_id, value)
        self._receive_learn(learn)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, learn)
        self._proposal = None

    def _receive_learn(self, message: Learn) -> None:
        current = self.learned_decision
        if current is not None and current.value != message.value:
            raise PaxosSafetyViolation(
                f"learner {self.node_id!r} observed conflicting Paxos decisions"
            )
        if current is None:
            decision = LearnedDecision(message.proposal, message.value)
            self.sim.persistent_state[self.node_id][self._LEARNED] = decision
            self.sim._record(
                "paxos-learned",
                learner=self.node_id,
                proposal=message.proposal,
                value=message.value,
            )

    def _reject_source(self, message: Message, expected: str) -> None:
        self.sim._record(
            "paxos-envelope-rejected",
            node=self.node_id,
            src=message.src,
            dst=message.dst,
            expected_src=expected,
            reason="source-identity-mismatch",
            payload=message.payload,
        )

    def _validate_durable_state(self) -> None:
        state = self.sim.persistent_state[self.node_id]
        promised = state[self._PROMISED]
        accepted = state[self._ACCEPTED]
        history = state[self._ACCEPT_HISTORY]
        learned = state[self._LEARNED]
        last_round = state[self._LAST_PROPOSAL_ROUND]
        if promised is not None and not isinstance(promised, ProposalNumber):
            raise TypeError("durable Paxos promise must be a ProposalNumber or None")
        if accepted is not None and not isinstance(accepted, AcceptedValue):
            raise TypeError("durable Paxos accepted state must be AcceptedValue or None")
        if not isinstance(history, tuple) or not all(
            isinstance(item, AcceptedValue) for item in history
        ):
            raise TypeError("durable Paxos accept history must contain AcceptedValue entries")
        if learned is not None and not isinstance(learned, LearnedDecision):
            raise TypeError("durable Paxos learned state must be LearnedDecision or None")
        if not isinstance(last_round, int) or last_round < 0:
            raise TypeError("durable Paxos proposer round must be a non-negative integer")
        if accepted is not None and promised is not None and accepted.proposal > promised:
            raise PaxosSafetyViolation("accepted proposal cannot exceed durable promise")
        by_proposal: dict[ProposalNumber, Any] = {}
        for item in history:
            existing = by_proposal.get(item.proposal)
            if item.proposal in by_proposal and existing != item.value:
                raise PaxosSafetyViolation(
                    f"durable accept history conflicts at proposal {item.proposal!r}"
                )
            by_proposal[item.proposal] = item.value
        if accepted is not None and accepted not in history:
            raise PaxosSafetyViolation("current accepted value must appear in durable history")


class PaxosSafetyHarness:
    """Reconstruct single-decree Paxos safety from durable acceptor evidence."""

    def __init__(self, cluster: PaxosCluster) -> None:
        self.cluster = cluster

    def checkpoint(self) -> None:
        quorum = self.cluster.quorum_size
        acceptors_by_pair: list[tuple[ProposalNumber, Any, set[str]]] = []
        for node_id in self.cluster.node_ids:
            node = self.cluster.node(node_id)
            node._validate_durable_state()
            for accepted in node.accept_history:
                for index, (proposal, value, acceptors) in enumerate(acceptors_by_pair):
                    if proposal == accepted.proposal and value == accepted.value:
                        acceptors.add(node_id)
                        acceptors_by_pair[index] = (proposal, value, acceptors)
                        break
                else:
                    acceptors_by_pair.append(
                        (accepted.proposal, accepted.value, {node_id})
                    )

        values_by_proposal: list[tuple[ProposalNumber, Any]] = []
        for proposal, value, _ in acceptors_by_pair:
            existing = [item for item in values_by_proposal if item[0] == proposal]
            if existing and existing[0][1] != value:
                raise PaxosSafetyViolation(
                    f"proposal {proposal!r} has conflicting accepted values"
                )
            if not existing:
                values_by_proposal.append((proposal, value))

        chosen = [
            (proposal, value, tuple(sorted(acceptors)))
            for proposal, value, acceptors in acceptors_by_pair
            if len(acceptors) >= quorum
        ]
        if chosen:
            first_value = chosen[0][1]
            if any(value != first_value for _, value, _ in chosen[1:]):
                raise PaxosSafetyViolation(
                    f"durable quorum evidence chose multiple Paxos values: {chosen!r}"
                )

        learned = [
            decision
            for node_id in self.cluster.node_ids
            if (decision := self.cluster.node(node_id).learned_decision) is not None
        ]
        if learned:
            first = learned[0].value
            if any(decision.value != first for decision in learned[1:]):
                raise PaxosSafetyViolation("learners contain conflicting Paxos decisions")
            if not chosen or chosen[0][1] != first:
                raise PaxosSafetyViolation(
                    "learned Paxos decision lacks matching durable majority acceptance"
                )

        objective = self.cluster.chosen
        if objective is not None:
            if not chosen or chosen[0][1] != objective.value:
                raise PaxosSafetyViolation(
                    "cluster chosen value lacks matching durable majority acceptance"
                )
            if learned and learned[0].value != objective.value:
                raise PaxosSafetyViolation("objective and learned Paxos decisions diverged")
