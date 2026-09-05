from __future__ import annotations

from dataclasses import dataclass

from .raft import RaftRole
from .simulator import Message, Simulator
from .snapshot import KVSnapshot, KVSnapshotStore


@dataclass(frozen=True, slots=True)
class InstallSnapshotRequest:
    term: int
    leader_id: str
    follower_id: str
    snapshot: KVSnapshot
    request_id: int = 0

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if not self.leader_id or not self.follower_id:
            raise ValueError("snapshot endpoints must be non-empty")
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")


@dataclass(frozen=True, slots=True)
class InstallSnapshotResponse:
    term: int
    leader_id: str
    follower_id: str
    success: bool
    last_included_index: int
    requested_last_included_index: int
    request_id: int = 0

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if self.last_included_index < 0:
            raise ValueError("last_included_index must be non-negative")
        if self.requested_last_included_index < 0:
            raise ValueError("requested_last_included_index must be non-negative")
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")


class SnapshotTransport:
    """Deterministic simulator transport for durable InstallSnapshot exchange.

    Snapshot traffic uses dedicated per-node delivery handlers but shares the
    logical Raft node-to-node link identity. Existing partition and explicit
    drop/delay/duplicate schedules therefore apply to InstallSnapshot exactly as
    they do to ordinary Raft traffic. Receiver delivery is coupled to the durable
    KV snapshot installer, so a successful response means the follower's Raft
    boundary, applied boundary, KV state, and dedup state are durably installed.
    """

    _PREFIX = "__raft_snapshot__"

    def __init__(self, store: KVSnapshotStore) -> None:
        self.store = store
        self.cluster = store.cluster
        self.sim = store.sim
        for node_id in self.cluster.node_ids:
            self.sim.register(self.endpoint(node_id), self._handle_message)

    @classmethod
    def endpoint(cls, node_id: str) -> str:
        return f"{cls._PREFIX}:{node_id}"

    def send_install_snapshot(
        self,
        *,
        leader_id: str,
        follower_id: str,
        term: int,
        snapshot: KVSnapshot,
        request_id: int = 0,
    ) -> None:
        if leader_id not in self.cluster.nodes or follower_id not in self.cluster.nodes:
            raise ValueError("snapshot transport requires known Raft nodes")
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        self.sim._record(
            "raft-install-snapshot-request",
            leader=leader_id,
            follower=follower_id,
            term=term,
            last_included_index=snapshot.last_included_index,
            last_included_term=snapshot.last_included_term,
            request_id=request_id,
        )
        self.sim.send(
            leader_id,
            follower_id,
            InstallSnapshotRequest(
                term=term,
                leader_id=leader_id,
                follower_id=follower_id,
                snapshot=snapshot,
                request_id=request_id,
            ),
            delivery_dst=self.endpoint(follower_id),
        )

    def _handle_message(self, sim: Simulator, message: Message) -> None:
        if sim is not self.sim:
            raise ValueError("snapshot transport invoked by a different simulator")
        payload = message.payload
        if isinstance(payload, InstallSnapshotRequest):
            self._handle_request(payload)
            return
        if isinstance(payload, InstallSnapshotResponse):
            self._handle_response(payload)
            return
        raise TypeError(f"unsupported snapshot message {type(payload).__name__}")

    def _handle_request(self, request: InstallSnapshotRequest) -> None:
        follower = self.cluster.node(request.follower_id)
        if not self.sim.is_alive(request.follower_id):
            self.sim._record(
                "raft-install-snapshot-unavailable",
                leader=request.leader_id,
                follower=request.follower_id,
                term=request.term,
                reason="follower-crashed",
            )
            return

        if request.term < follower.current_term:
            success = False
            installed_index = follower.log_base_index
        else:
            if request.term > follower.current_term:
                follower._advance_term(request.term)
            volatile = self.sim.volatile_state[request.follower_id]
            volatile["role"] = RaftRole.FOLLOWER.value
            volatile["votes_received"] = set()
            follower.reset_election_timeout(reason="install-snapshot")
            latest = self.store.latest(request.follower_id)
            if (
                latest is not None
                and latest.last_included_index > request.snapshot.last_included_index
            ):
                installed_index = latest.last_included_index
                self.sim._record(
                    "raft-install-snapshot-stale",
                    leader=request.leader_id,
                    follower=request.follower_id,
                    term=request.term,
                    incoming_index=request.snapshot.last_included_index,
                    installed_index=installed_index,
                )
            else:
                self.store.install(
                    request.follower_id,
                    request.snapshot,
                    preserve_matching_suffix=True,
                )
                installed_index = request.snapshot.last_included_index
            success = True

        response = InstallSnapshotResponse(
            term=follower.current_term,
            leader_id=request.leader_id,
            follower_id=request.follower_id,
            success=success,
            last_included_index=installed_index,
            requested_last_included_index=request.snapshot.last_included_index,
            request_id=request.request_id,
        )
        self.sim.send(
            request.follower_id,
            request.leader_id,
            response,
            delivery_dst=self.endpoint(request.leader_id),
        )

    def _handle_response(self, response: InstallSnapshotResponse) -> None:
        leader = self.cluster.node(response.leader_id)
        if not self.sim.is_alive(response.leader_id):
            return
        if response.term > leader.current_term:
            leader._advance_term(response.term)
        self.sim._record(
            "raft-install-snapshot-response",
            leader=response.leader_id,
            follower=response.follower_id,
            term=response.term,
            success=response.success,
            last_included_index=response.last_included_index,
            requested_last_included_index=response.requested_last_included_index,
            request_id=response.request_id,
        )
