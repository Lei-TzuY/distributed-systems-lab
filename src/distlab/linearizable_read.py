from __future__ import annotations

from .kv import ReplicatedKV
from .raft import RaftRole
from .replication import LeaderReplicator, ReplicationError, ReplicationResponseMissing


class LinearizableReadError(RuntimeError):
    """Base error for linearizable leader-read failures."""


class CurrentTermCommitRequired(LinearizableReadError):
    """Raised until the leader has committed an entry from its current term."""


class ReadQuorumUnavailable(LinearizableReadError):
    """Raised when the leader cannot confirm authority with a majority."""


class LinearizableKVReader:
    """Serve KV reads only after a deterministic Raft quorum confirmation.

    The reader is intentionally conservative. A leader must first have a
    current-term committed entry, then obtain successful AppendEntries
    acknowledgements from enough peers to form a majority in the same term.
    Only after that barrier succeeds are newly committed entries applied to the
    local state machine and the requested key returned.
    """

    def __init__(self, kv: ReplicatedKV, replicator: LeaderReplicator) -> None:
        self.kv = kv
        self.replicator = replicator
        self.leader = replicator.leader
        if kv.cluster is not self.leader.cluster:
            raise ValueError("KV state and leader replicator must belong to the same cluster")

    def get(self, key: str, *, max_attempts_per_peer: int = 1) -> str | None:
        if not key:
            raise ValueError("KV key must be non-empty")
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")

        self._require_current_leader()
        self._require_current_term_commit()

        cluster_size = len(self.leader.cluster.node_ids)
        majority = cluster_size // 2 + 1
        acknowledgements = 1
        acknowledged_peers: list[str] = []

        for peer in self.leader.peers:
            if acknowledgements >= majority:
                break
            try:
                if self.replicator.replicate(peer, max_attempts=max_attempts_per_peer):
                    acknowledgements += 1
                    acknowledged_peers.append(peer)
            except ReplicationResponseMissing:
                continue

        self._require_current_leader()
        if acknowledgements < majority:
            self.leader.sim._record(
                "raft-linearizable-read-quorum-failed",
                leader=self.leader.node_id,
                term=self.leader.current_term,
                acknowledgements=acknowledgements,
                majority=majority,
            )
            raise ReadQuorumUnavailable(
                f"leader {self.leader.node_id!r} confirmed only {acknowledgements} "
                f"of {majority} required replicas"
            )

        self.kv.apply_committed(self.leader.node_id)
        value = self.kv.get(self.leader.node_id, key)
        self.leader.sim._record(
            "raft-linearizable-read",
            leader=self.leader.node_id,
            term=self.leader.current_term,
            commit_index=self.leader.commit_index,
            key=key,
            value=value,
            acknowledged_peers=tuple(acknowledged_peers),
            majority=majority,
        )
        return value

    def _require_current_term_commit(self) -> None:
        commit_index = self.leader.commit_index
        if commit_index == 0 or self.leader.log[commit_index - 1].term != self.leader.current_term:
            raise CurrentTermCommitRequired(
                "linearizable reads require a committed entry from the leader's current term"
            )

    def _require_current_leader(self) -> None:
        if not self.leader.sim.is_alive(self.leader.node_id):
            raise LinearizableReadError("linearizable reads require a live leader")
        if self.leader.role is not RaftRole.LEADER:
            raise LinearizableReadError("linearizable reads require leader role")
        if self.leader.current_term != self.replicator.term:
            raise LinearizableReadError("linearizable read replicator term is stale")
        try:
            self.replicator.advance_commit_index()
        except ReplicationError as exc:
            raise LinearizableReadError(str(exc)) from exc
