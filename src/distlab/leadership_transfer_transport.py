from __future__ import annotations

from dataclasses import dataclass

from .membership import ReconfigurableRaftCluster, VotingConfiguration
from .raft import RaftCluster, RaftNode, RaftRole
from .simulator import Message, Simulator


@dataclass(frozen=True, slots=True)
class TimeoutNow:
    term: int
    leader_id: str
    transferee_id: str
    attempt_id: int

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if not self.leader_id or not self.transferee_id:
            raise ValueError("leadership transfer endpoints must be non-empty")
        if self.leader_id == self.transferee_id:
            raise ValueError("leadership transfer endpoints must be distinct")
        if self.attempt_id <= 0:
            raise ValueError("leadership transfer attempt id must be positive")


class LeadershipTransferTransport:
    """Deterministic transport for the final leadership-transfer election trigger.

    TimeoutNow traffic keeps the logical Raft node-to-node link identity while
    using a dedicated delivery endpoint. It therefore participates in the same
    partition and drop/delay/duplicate semantics as ordinary Raft traffic, while
    validating the control-message authority again at delivery time.
    """

    _PREFIX = "__raft_leadership_transfer__"
    _CLUSTER_CACHE_ATTR = "_leadership_transfer_transport"

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self._next_attempt_id = 0
        self._active_attempts: dict[int, tuple[str, str, int]] = {}
        self._attempt_configurations: dict[int, VotingConfiguration | None] = {}
        for node_id in self.cluster.node_ids:
            self.sim.register(self.endpoint(node_id), self._handle_message)

    @classmethod
    def for_cluster(cls, cluster: RaftCluster) -> LeadershipTransferTransport:
        existing = getattr(cluster, cls._CLUSTER_CACHE_ATTR, None)
        if isinstance(existing, cls):
            return existing
        transport = cls(cluster)
        setattr(cluster, cls._CLUSTER_CACHE_ATTR, transport)
        return transport

    @classmethod
    def endpoint(cls, node_id: str) -> str:
        return f"{cls._PREFIX}:{node_id}"

    def send_timeout_now(self, leader_id: str, transferee_id: str, *, term: int) -> int:
        if leader_id not in self.cluster.nodes or transferee_id not in self.cluster.nodes:
            raise ValueError("leadership transfer transport requires known Raft nodes")
        if leader_id == transferee_id:
            raise ValueError("leadership transfer endpoints must be distinct")
        leader = self.cluster.node(leader_id)
        if not self.sim.is_alive(leader_id):
            raise RuntimeError("leadership transfer trigger requires a live source leader")
        if leader.role is not RaftRole.LEADER or leader.current_term != term:
            raise RuntimeError("leadership transfer trigger requires current leader authority")
        active_attempt = self._active_attempt_for(leader_id, term)
        if active_attempt is not None:
            active_attempt_id, active_transferee = active_attempt
            self.sim._record(
                "raft-timeout-now-request-rejected",
                leader=leader_id,
                transferee=transferee_id,
                term=term,
                active_attempt_id=active_attempt_id,
                active_transferee=active_transferee,
                reason="transfer-attempt-already-active",
            )
            raise RuntimeError(
                "leadership transfer trigger already active for current leader term"
            )
        self._next_attempt_id += 1
        attempt_id = self._next_attempt_id
        self._active_attempts[attempt_id] = (leader_id, transferee_id, term)
        self._attempt_configurations[attempt_id] = self._configuration_identity()
        self.sim._record(
            "raft-timeout-now-request",
            leader=leader_id,
            transferee=transferee_id,
            term=term,
            attempt_id=attempt_id,
        )
        self.sim.send(
            leader_id,
            transferee_id,
            TimeoutNow(
                term=term,
                leader_id=leader_id,
                transferee_id=transferee_id,
                attempt_id=attempt_id,
            ),
            delivery_dst=self.endpoint(transferee_id),
        )
        return attempt_id

    def cancel_timeout_now(self, attempt_id: int) -> bool:
        identity = self._active_attempts.pop(attempt_id, None)
        self._attempt_configurations.pop(attempt_id, None)
        if identity is None:
            return False
        leader_id, transferee_id, term = identity
        self.sim._record(
            "raft-timeout-now-cancelled",
            leader=leader_id,
            transferee=transferee_id,
            term=term,
            attempt_id=attempt_id,
        )
        return True

    def _active_attempt_for(self, leader_id: str, term: int) -> tuple[int, str] | None:
        for attempt_id, identity in self._active_attempts.items():
            active_leader, transferee_id, active_term = identity
            if active_leader == leader_id and active_term == term:
                return attempt_id, transferee_id
        return None

    def _configuration_identity(self) -> VotingConfiguration | None:
        if not isinstance(self.cluster, ReconfigurableRaftCluster):
            return None
        return self.cluster.voting_configuration

    def _handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim:
            raise ValueError("leadership transfer transport invoked by a different simulator")
        payload = message.payload
        if not isinstance(payload, TimeoutNow):
            raise TypeError(f"unsupported leadership transfer message {type(payload).__name__}")
        if self._reject_invalid_envelope(message, payload):
            return
        self._handle_timeout_now(payload)

    def _reject_invalid_envelope(self, message: Message, request: TimeoutNow) -> bool:
        expected_delivery_dst = self.endpoint(request.transferee_id)
        reason: str | None = None
        if request.leader_id not in self.cluster.nodes:
            reason = "unknown-source-principal"
        elif request.transferee_id not in self.cluster.nodes:
            reason = "unknown-destination-principal"
        elif message.src != request.leader_id:
            reason = "source-identity-mismatch"
        elif message.dst != request.transferee_id:
            reason = "destination-identity-mismatch"
        elif message.delivery_dst != expected_delivery_dst:
            reason = "delivery-endpoint-mismatch"
        if reason is None:
            return False

        self.sim._record(
            "raft-timeout-now-envelope-rejected",
            src=message.src,
            dst=message.dst,
            delivery_dst=message.delivery_dst,
            leader=request.leader_id,
            transferee=request.transferee_id,
            term=request.term,
            attempt_id=request.attempt_id,
            expected_delivery_dst=expected_delivery_dst,
            reason=reason,
        )
        return True

    def _handle_timeout_now(self, request: TimeoutNow) -> None:
        target = self.cluster.node(request.transferee_id)
        leader = self.cluster.node(request.leader_id)
        reason: str | None = None
        active_identity = self._active_attempts.get(request.attempt_id)
        if active_identity != (request.leader_id, request.transferee_id, request.term):
            reason = "stale-transfer-attempt"
        elif not self.sim.is_alive(request.transferee_id):
            reason = "transferee-crashed"
        elif not self.sim.is_alive(request.leader_id):
            reason = "source-leader-crashed"
        elif request.term != target.current_term:
            reason = "term-mismatch"
        elif self.cluster.leaders_by_term.get(request.term) != request.leader_id:
            reason = "source-not-recorded-leader"
        elif leader.role is not RaftRole.LEADER or leader.current_term != request.term:
            reason = "source-no-longer-current-leader"
        elif (
            target.last_log_index != leader.last_log_index
            or target.last_log_term != leader.last_log_term
            or target.commit_index < leader.commit_index
        ):
            reason = "transferee-not-caught-up"
        elif not self._retained_log_overlap_matches(leader, target):
            reason = "transferee-log-mismatch"
        elif isinstance(self.cluster, ReconfigurableRaftCluster):
            configuration = self.cluster.voting_configuration
            if not self.cluster.is_voter(request.transferee_id):
                reason = "transferee-not-voter"
            elif (
                configuration.new_voters is not None
                and request.transferee_id not in configuration.new_voters
            ):
                reason = "transferee-outgoing-only"
            elif self._attempt_configurations.get(request.attempt_id) is not configuration:
                reason = "membership-configuration-changed"

        if active_identity == (request.leader_id, request.transferee_id, request.term):
            self._active_attempts.pop(request.attempt_id, None)
            self._attempt_configurations.pop(request.attempt_id, None)

        if reason is not None:
            self.sim._record(
                "raft-timeout-now-rejected",
                leader=request.leader_id,
                transferee=request.transferee_id,
                term=request.term,
                attempt_id=request.attempt_id,
                target_term=target.current_term,
                reason=reason,
            )
            return

        self.sim._record(
            "raft-timeout-now-delivered",
            leader=request.leader_id,
            transferee=request.transferee_id,
            term=request.term,
            attempt_id=request.attempt_id,
        )
        target.start_election()

    @staticmethod
    def _retained_log_overlap_matches(leader: RaftNode, target: RaftNode) -> bool:
        """Verify every log position still retained by both transfer endpoints."""

        leader_log = leader.log_view
        target_log = target.log_view
        overlap_boundary = max(leader_log.base_index, target_log.base_index)
        try:
            if leader_log.term_at(overlap_boundary) != target_log.term_at(overlap_boundary):
                return False
            for index in range(overlap_boundary + 1, leader_log.last_index + 1):
                if leader_log.entry_at(index) != target_log.entry_at(index):
                    return False
        except IndexError:
            return False
        return True
