from __future__ import annotations

from dataclasses import dataclass

from .raft import RaftNode, RaftRole


class ReplicationError(RuntimeError):
    """Base error for deterministic leader replication failures."""


class ReplicationResponseMissing(ReplicationError):
    """Raised when a probe produces no matching AppendEntries response."""


@dataclass(frozen=True, slots=True)
class PeerReplicationProgress:
    """Leader-side replication progress for one follower."""

    next_index: int
    match_index: int


class LeaderReplicator:
    """Deterministically drive Raft leader replication and commit advancement.

    The replicator owns the leader-side ``nextIndex``/``matchIndex`` state used
    by AppendEntries retry plus the leader's volatile ``commitIndex``. A newly
    elected leader starts every follower at ``last_log_index + 1``. Rejections
    decrement ``nextIndex`` one index at a time (never below 1); successful
    responses advance ``matchIndex`` monotonically.

    After every successful response, commit advancement scans backward for the
    highest index replicated on a majority. Following Raft's safety rule, an
    index is advanced by replica counting only when the entry at that index is
    from the leader's current term. Committing such an entry implicitly commits
    all preceding entries in the log.
    """

    def __init__(self, leader: RaftNode) -> None:
        self.leader = leader
        self.sim = leader.sim
        self._term = leader.current_term
        self._commit_index = 0
        self._progress = {
            peer: PeerReplicationProgress(
                next_index=leader.last_log_index + 1,
                match_index=0,
            )
            for peer in leader.peers
        }
        self._require_current_leader()

    @property
    def term(self) -> int:
        return self._term

    @property
    def commit_index(self) -> int:
        return self._commit_index

    def progress(self, peer: str) -> PeerReplicationProgress:
        self._require_peer(peer)
        return self._progress[peer]

    def replicate(self, peer: str, *, max_attempts: int | None = None) -> bool:
        """Replicate the leader's current log to ``peer`` using deterministic retries.

        Returns ``True`` after a successful AppendEntries response. If
        ``max_attempts`` is reached first, returns ``False`` while preserving the
        latest backtracked progress for a later call.
        """

        self._require_peer(peer)
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive when provided")

        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            self._require_current_leader()
            attempts += 1
            progress = self._progress[peer]
            next_index = progress.next_index
            prev_log_index = next_index - 1
            entries = self.leader.log[prev_log_index:]
            trace_start = len(self.sim.trace)

            self.sim._record(
                "raft-replication-probe",
                leader=self.leader.node_id,
                follower=peer,
                term=self._term,
                next_index=next_index,
                prev_log_index=prev_log_index,
                entry_count=len(entries),
                attempt=attempts,
            )
            self.leader.send_append_entries(
                peer,
                prev_log_index=prev_log_index,
                entries=entries,
            )
            self.sim.run()

            response = self._matching_response(peer, trace_start)
            if response is None:
                raise ReplicationResponseMissing(
                    f"no AppendEntries response from {peer!r} for leader "
                    f"{self.leader.node_id!r} in term {self._term}"
                )

            if bool(response.details["success"]):
                match_index = int(response.details["match_index"])
                old_match_index = progress.match_index
                new_match_index = max(old_match_index, match_index)
                self._progress[peer] = PeerReplicationProgress(
                    next_index=new_match_index + 1,
                    match_index=new_match_index,
                )
                self.sim._record(
                    "raft-replication-advance",
                    leader=self.leader.node_id,
                    follower=peer,
                    term=self._term,
                    previous_match_index=old_match_index,
                    match_index=new_match_index,
                    next_index=new_match_index + 1,
                )
                self.advance_commit_index()
                return True

            old_next_index = progress.next_index
            new_next_index = max(1, old_next_index - 1)
            self._progress[peer] = PeerReplicationProgress(
                next_index=new_next_index,
                match_index=progress.match_index,
            )
            self.sim._record(
                "raft-replication-backtrack",
                leader=self.leader.node_id,
                follower=peer,
                term=self._term,
                previous_next_index=old_next_index,
                next_index=new_next_index,
            )

        return False

    def advance_commit_index(self) -> int:
        """Advance and return the leader's commit index when Raft permits it.

        The leader itself counts as one replica. The method chooses the highest
        index greater than the current commit index that is present on a
        majority and whose log entry belongs to this leader term. It never
        decreases ``commit_index`` and never commits an older-term entry merely
        because that older entry is replicated on a majority.
        """

        self._require_current_leader()
        majority = len(self.leader.cluster.node_ids) // 2 + 1
        previous = self._commit_index

        for index in range(self.leader.last_log_index, previous, -1):
            if self.leader.log[index - 1].term != self._term:
                continue
            replicas = 1 + sum(
                progress.match_index >= index for progress in self._progress.values()
            )
            if replicas < majority:
                continue

            self._commit_index = index
            self.sim._record(
                "raft-commit-advance",
                leader=self.leader.node_id,
                term=self._term,
                previous_commit_index=previous,
                commit_index=index,
                replicas=replicas,
                majority=majority,
            )
            break

        return self._commit_index

    def _matching_response(self, peer: str, trace_start: int):
        for record in reversed(self.sim.trace[trace_start:]):
            if record.kind != "raft-append-response":
                continue
            if record.details.get("leader") != self.leader.node_id:
                continue
            if record.details.get("follower") != peer:
                continue
            if int(record.details.get("term", -1)) != self._term:
                continue
            return record
        return None

    def _require_current_leader(self) -> None:
        if not self.sim.is_alive(self.leader.node_id):
            raise ReplicationError("replication requires a live leader")
        if self.leader.role is not RaftRole.LEADER:
            raise ReplicationError("replication requires leader role")
        if self.leader.current_term != self._term:
            raise ReplicationError("replicator term is stale")

    def _require_peer(self, peer: str) -> None:
        if peer not in self._progress:
            raise ValueError(f"unknown peer {peer!r}")
