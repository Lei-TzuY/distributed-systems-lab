from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .kv import ClientRequestConflict
from .linearizability import OperationHistory
from .multipaxos import MultiPaxosNode
from .multipaxos_state_machine import (
    Append,
    MultiPaxosRequest,
    MultiPaxosStateMachine,
)
from .paxos import ProposalNumber


class ClientStatus(StrEnum):
    SUBMITTED = "submitted"
    PENDING = "pending"
    CHOSEN = "chosen_not_applied"
    APPLIED = "applied_not_acknowledged"
    COMPLETED = "completed"
    CONFLICT = "conflict"
    LEADERSHIP_UNAVAILABLE = "leadership_unavailable"
    QUORUM_UNAVAILABLE = "quorum_unavailable"


@dataclass(frozen=True, slots=True)
class ClientResult:
    status: ClientStatus
    request: MultiPaxosRequest
    node_id: str
    slot: int | None = None
    ballot: ProposalNumber | None = None
    value: str | None = None
    authority_ballot: ProposalNumber | None = None


@dataclass
class ClientJournal:
    """Client-owned retry/history state, retained across server reconstruction.

    Server deduplication never trusts this journal: it is replayed from applied
    decisions. Append is an exactly-once test extension, outside the KV checker.
    """

    history: OperationHistory = field(default_factory=OperationHistory)
    requests: dict[str, MultiPaxosRequest] = field(default_factory=dict)

    def register(self, request: MultiPaxosRequest) -> None:
        for previous in self.requests.values():
            if (
                previous.identity == request.identity
                or previous.operation_id == request.operation_id
            ):
                if previous != request:
                    raise ClientRequestConflict(
                        "request/history identity reused with different content"
                    )
                return
        if not isinstance(request.operation, Append):
            self.history.invoke(request.operation_id, request.client_id, request.operation)
        self.requests[request.operation_id] = request


class MultiPaxosClient:
    """Bounded deterministic client adapter over the existing Multi-Paxos protocol.

    submit/poll expose asynchronous states; execute adds a finite simulator pump.
    Success requires a matching local durable applied request and live local
    Phase-1 authority. Get uses the result captured at its first applied slot.
    No service decision is sourced from the cluster's objective chosen map.
    """

    def __init__(
        self,
        machine: MultiPaxosStateMachine,
        journal: ClientJournal | None = None,
    ) -> None:
        self.machine = machine
        self.cluster = machine.cluster
        self.sim = machine.sim
        self.journal = journal if journal is not None else ClientJournal()

    @property
    def history(self) -> OperationHistory:
        return self.journal.history

    def submit(
        self,
        node_id: str,
        request: MultiPaxosRequest,
        *,
        slot: int | None = None,
    ) -> ClientResult:
        """Submit/retry one identity. Optional slot is a proposal-placement hint.

        Phase 1 may adopt a different command at that slot. Completion always
        identifies the actual request, independently of this placement hint.
        """
        if not isinstance(request, MultiPaxosRequest):
            raise TypeError("expected MultiPaxosRequest")
        if slot is not None and (type(slot) is not int or slot <= 0):
            raise ValueError("slot must be a positive integer")
        node = self.cluster.node(node_id)
        try:
            self.journal.register(request)
            self._check_local_identity(node, request)
        except ClientRequestConflict:
            return self._result(ClientStatus.CONFLICT, node, request)
        if not self._authorized(node):
            return self._result(ClientStatus.LEADERSHIP_UNAVAILABLE, node, request)
        observed = self.poll(node_id, request, apply=False, acknowledge=False)
        if observed.status in (ClientStatus.APPLIED, ClientStatus.CHOSEN):
            return observed
        for pending in node.pending_proposals:
            if pending.value == request:
                node.retry_proposal(pending.slot)
                return self._result(ClientStatus.PENDING, node, request, pending.slot)
        proposed_slot = slot if slot is not None else self._high_water(node) + 1
        if any(p.slot == proposed_slot for p in node.pending_proposals):
            return self._result(ClientStatus.PENDING, node, request, proposed_slot)
        node.propose(proposed_slot, request)
        return self._result(ClientStatus.SUBMITTED, node, request, proposed_slot)

    def poll(
        self,
        node_id: str,
        request: MultiPaxosRequest,
        *,
        expected_ballot: ProposalNumber | None = None,
        apply: bool = True,
        acknowledge: bool = True,
    ) -> ClientResult:
        """Observe exact request evidence; optionally apply and publish its response."""
        if self.journal.requests.get(request.operation_id) != request:
            raise ValueError("poll requires the original registered request")
        node = self.cluster.node(node_id)
        if not self._authorized(node, expected_ballot):
            return self._result(ClientStatus.LEADERSHIP_UNAVAILABLE, node, request)
        try:
            self._check_local_identity(node, request)
        except ClientRequestConflict:
            return self._result(ClientStatus.CONFLICT, node, request)
        if apply:
            self.machine.apply_ready(node_id)
        record = self.machine.client_requests(node_id).get(request.identity)
        if record is not None:
            if record.request != request:
                return self._result(ClientStatus.CONFLICT, node, request)
            decision = self.machine.applied(node_id)[record.slot - 1]
            if decision.value != request:
                raise AssertionError("client completion lacks matching applied request")
            if acknowledge and not isinstance(request.operation, Append):
                completed = {r.operation_id: r for r in self.history.completed()}
                if request.operation_id not in completed:
                    self.history.respond(request.operation_id, record.result)
                elif completed[request.operation_id].completion.result != record.result:
                    raise AssertionError("reconstructed client result changed")
            return self._result(
                ClientStatus.COMPLETED if acknowledge else ClientStatus.APPLIED,
                node,
                request,
                record.slot,
                record.result,
                decision.proposal,
            )
        for decision in node.learned_decisions:
            if decision.value == request:
                return self._result(
                    ClientStatus.CHOSEN,
                    node,
                    request,
                    decision.slot,
                    ballot=decision.proposal,
                )
        return self._result(ClientStatus.PENDING, node, request)

    def execute(
        self,
        node_id: str,
        request: MultiPaxosRequest,
        *,
        max_rounds: int = 8,
        events_per_round: int = 64,
    ) -> ClientResult:
        """Submit/retry with bounded protocol/application work and explicit uncertainty."""
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if type(events_per_round) is not int or events_per_round <= 0:
            raise ValueError("events_per_round must be a positive integer")
        result = self.submit(node_id, request)
        for _ in range(max_rounds):
            if result.status in (ClientStatus.LEADERSHIP_UNAVAILABLE, ClientStatus.CONFLICT):
                return result
            node = self.cluster.node(node_id)
            self._repair_prefix(node)
            self.sim.run(max_events=events_per_round)
            for peer in self.cluster.node_ids:
                if self.sim.is_alive(peer):
                    self.machine.apply_ready(peer)
            result = self.poll(node_id, request)
            if result.status in (
                ClientStatus.COMPLETED,
                ClientStatus.CONFLICT,
                ClientStatus.LEADERSHIP_UNAVAILABLE,
            ):
                return result
            result = self.submit(node_id, request)
        # This is a bounded observation, not a claim that the operation failed.
        # A delayed decision may still arrive and an exact retry may complete it.
        if result.status in (ClientStatus.SUBMITTED, ClientStatus.PENDING):
            return self._result(
                ClientStatus.QUORUM_UNAVAILABLE,
                self.cluster.node(node_id),
                request,
                result.slot,
            )
        return result

    @staticmethod
    def _authorized(
        node: MultiPaxosNode,
        expected_ballot: ProposalNumber | None = None,
    ) -> bool:
        return (
            node.sim.is_alive(node.node_id)
            and node.has_leader_authority
            and node.leader_ballot == node.promised_ballot
            and (expected_ballot is None or node.leader_ballot == expected_ballot)
        )

    @staticmethod
    def _high_water(node: MultiPaxosNode) -> int:
        return max(
            (
                r.slot
                for r in (
                    *node.accept_history,
                    *node.leadership_accepted,
                    *node.learned_decisions,
                )
            ),
            default=0,
        )

    @staticmethod
    def _check_local_identity(node: MultiPaxosNode, request: MultiPaxosRequest) -> None:
        for record in (*node.accept_history, *node.leadership_accepted, *node.learned_decisions):
            command = record.value
            if (
                isinstance(command, MultiPaxosRequest)
                and (
                    command.identity == request.identity
                    or command.operation_id == request.operation_id
                )
                and command != request
            ):
                raise ClientRequestConflict("conflicting locally known request identity")

    def _repair_prefix(self, node: MultiPaxosNode) -> None:
        if not self._authorized(node):
            return
        start = self.machine.last_applied(node.node_id) + 1
        stop = min(self._high_water(node) + 1, start + 16)
        pending = {p.slot for p in node.pending_proposals}
        for slot in range(start, stop):
            if node.learned_decision(slot) is not None:
                continue
            if slot in pending:
                node.retry_proposal(slot)
            else:
                # Phase 1 still adopts the highest accepted record, including
                # accepted None. Only a genuinely unaccepted hole gets a no-op.
                node.propose(slot, None)

    def _result(
        self,
        status: ClientStatus,
        node: MultiPaxosNode,
        request: MultiPaxosRequest,
        slot: int | None = None,
        value: str | None = None,
        ballot: ProposalNumber | None = None,
    ) -> ClientResult:
        result = ClientResult(
            status,
            request,
            node.node_id,
            slot,
            ballot or node.leader_ballot,
            value,
            node.leader_ballot if self._authorized(node) else None,
        )
        self.sim._record(
            "multipaxos-client-result",
            status=status.value,
            node=node.node_id,
            client_id=request.client_id,
            request_id=request.request_id,
            operation_id=request.operation_id,
            slot=slot,
            ballot=result.ballot,
            authority_ballot=result.authority_ballot,
            result=value,
        )
        return result
