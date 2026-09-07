from __future__ import annotations

from dataclasses import dataclass

from .raft import RaftCluster, RaftRole
from .simulator import Message, Simulator


@dataclass(frozen=True, slots=True)
class TimeoutNow:
    term: int
    leader_id: str
    transferee_id: str

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if not self.leader_id or not self.transferee_id:
            raise ValueError("leadership transfer endpoints must be non-empty")
        if self.leader_id == self.transferee_id:
            raise ValueError("leadership transfer endpoints must be distinct")


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

    def send_timeout_now(self, leader_id: str, transferee_id: str, *, term: int) -> None:
        if leader_id not in self.cluster.nodes or transferee_id not in self.cluster.nodes:
            raise ValueError("leadership transfer transport requires known Raft nodes")
        leader = self.cluster.node(leader_id)
        if not self.sim.is_alive(leader_id):
            raise RuntimeError("leadership transfer trigger requires a live source leader")
        if leader.role is not RaftRole.LEADER or leader.current_term != term:
            raise RuntimeError("leadership transfer trigger requires current leader authority")
        self.sim._record(
            "raft-timeout-now-request",
            leader=leader_id,
            transferee=transferee_id,
            term=term,
        )
        self.sim.send(
            leader_id,
            transferee_id,
            TimeoutNow(term=term, leader_id=leader_id, transferee_id=transferee_id),
            delivery_dst=self.endpoint(transferee_id),
        )

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
            expected_delivery_dst=expected_delivery_dst,
            reason=reason,
        )
        return True

    def _handle_timeout_now(self, request: TimeoutNow) -> None:
        target = self.cluster.node(request.transferee_id)
        leader = self.cluster.node(request.leader_id)
        reason: str | None = None
        if not self.sim.is_alive(request.transferee_id):
            reason = "transferee-crashed"
        elif not self.sim.is_alive(request.leader_id):
            reason = "source-leader-crashed"
        elif request.term != target.current_term:
            reason = "term-mismatch"
        elif self.cluster.leaders_by_term.get(request.term) != request.leader_id:
            reason = "source-not-recorded-leader"
        elif leader.role is not RaftRole.LEADER or leader.current_term != request.term:
            reason = "source-no-longer-current-leader"

        if reason is not None:
            self.sim._record(
                "raft-timeout-now-rejected",
                leader=request.leader_id,
                transferee=request.transferee_id,
                term=request.term,
                target_term=target.current_term,
                reason=reason,
            )
            return

        self.sim._record(
            "raft-timeout-now-delivered",
            leader=request.leader_id,
            transferee=request.transferee_id,
            term=request.term,
        )
        target.start_election()
