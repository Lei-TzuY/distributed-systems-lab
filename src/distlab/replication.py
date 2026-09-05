from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .raft import RaftNode, RaftRole

if TYPE_CHECKING:
    from .snapshot_transport import SnapshotTransport


class ReplicationError(RuntimeError):
    """Base error for deterministic leader replication failures."""


class ReplicationResponseMissing(ReplicationError):
    """Raised when a replication probe produces no matching response."""


@dataclass(frozen=True, slots=True)
class PeerReplicationProgress:
    next_index: int
    match_index: int


class LeaderReplicator:
    """Deterministically drive leader replication and commit propagation."""

    def __init__(
        self,
        leader: RaftNode,
        *,
        snapshot_transport: SnapshotTransport | None = None,
    ) -> None:
        self.leader = leader
        self.sim = leader.sim
        self.snapshot_transport = snapshot_transport
        self._term = leader.current_term
        self._commit_index = leader.commit_index
        self._snapshot_request_id = 0
        self._progress = {
            peer: PeerReplicationProgress(
                next_index=leader.last_log_index + 1,
                match_index=0,
            )
            for peer in leader.peers
        }
        self._require_current_leader()
        if snapshot_transport is not None and snapshot_transport.cluster is not leader.cluster:
            raise ValueError("snapshot transport must belong to the leader cluster")

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
        self._require_peer(peer)
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive when provided")

        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            self._require_current_leader()
            attempts += 1
            progress = self._progress[peer]
            next_index = progress.next_index
            log = self.leader.log_view
            if next_index <= log.base_index:
                return self._replicate_snapshot(peer, progress, attempts)

            prev_log_index = next_index - 1
            entries = log.suffix_from(next_index)
            trace_start = len(self.sim.trace)

            self.sim._record(
                "raft-replication-probe",
                leader=self.leader.node_id,
                follower=peer,
                term=self._term,
                next_index=next_index,
                prev_log_index=prev_log_index,
                entry_count=len(entries),
                leader_commit=self._commit_index,
                attempt=attempts,
            )
            self.leader.send_append_entries(
                peer,
                prev_log_index=prev_log_index,
                entries=entries,
                leader_commit=self._commit_index,
            )
            self.sim.run()

            response = self._matching_response(peer, trace_start, kind="raft-append-response")
            if response is None:
                raise ReplicationResponseMissing(
                    f"no AppendEntries response from {peer!r} for leader "
                    f"{self.leader.node_id!r} in term {self._term}"
                )

            if bool(response.details["success"]):
                match_index = int(response.details["match_index"])
                self._advance_peer(peer, progress, match_index, source="append")
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

    def recover_peer(self, peer: str, *, max_attempts: int | None = None) -> bool:
        """Drive a restarted or stale follower to the leader's current durable prefix.

        Recovery is complete once the follower has acknowledged the leader's full
        current log prefix and has observed the leader's current commit index. The
        latter may require one extra heartbeat because a successful replication can
        advance the leader commit only after the follower has processed that probe.
        """
        self._require_peer(peer)
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive when provided")

        attempts = 0
        while True:
            self._require_current_leader()
            follower = self.leader.cluster.node(peer)
            progress = self._progress[peer]
            target_commit = min(self._commit_index, self.leader.last_log_index)
            if (
                progress.match_index >= self.leader.last_log_index
                and follower.commit_index >= target_commit
            ):
                self.sim._record(
                    "raft-peer-recovered",
                    leader=self.leader.node_id,
                    follower=peer,
                    term=self._term,
                    match_index=progress.match_index,
                    commit_index=follower.commit_index,
                )
                return True
            if max_attempts is not None and attempts >= max_attempts:
                return False

            attempts += 1
            self.replicate(peer, max_attempts=1)

    def advance_commit_index(self) -> int:
        self._require_current_leader()
        majority = len(self.leader.cluster.node_ids) // 2 + 1
        previous = self._commit_index
        log = self.leader.log_view
        scan_floor = max(previous, log.base_index)

        for index in range(log.last_index, scan_floor, -1):
            if log.term_at(index) != self._term:
                continue
            replicas = 1 + sum(
                progress.match_index >= index for progress in self._progress.values()
            )
            if replicas < majority:
                continue

            self._commit_index = index
            self.leader.advance_commit_index(index, source=self.leader.node_id)
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

    def _replicate_snapshot(
        self,
        peer: str,
        progress: PeerReplicationProgress,
        attempt: int,
    ) -> bool:
        transport = self.snapshot_transport
        if transport is None:
            raise ReplicationError(
                "replication crossed the compacted log boundary without snapshot transport"
            )
        snapshot = transport.store.latest(self.leader.node_id)
        if snapshot is None:
            raise ReplicationError("compacted leader has no durable snapshot to install")
        log = self.leader.log_view
        if (
            snapshot.last_included_index != log.base_index
            or snapshot.last_included_term != log.base_term
        ):
            raise ReplicationError("leader snapshot does not match compacted log boundary")

        self._snapshot_request_id += 1
        request_id = self._snapshot_request_id
        trace_start = len(self.sim.trace)
        self.sim._record(
            "raft-replication-snapshot",
            leader=self.leader.node_id,
            follower=peer,
            term=self._term,
            next_index=progress.next_index,
            last_included_index=snapshot.last_included_index,
            last_included_term=snapshot.last_included_term,
            attempt=attempt,
            request_id=request_id,
        )
        transport.send_install_snapshot(
            leader_id=self.leader.node_id,
            follower_id=peer,
            term=self._term,
            snapshot=snapshot,
            request_id=request_id,
        )
        self.sim.run()
        if self.leader.current_term != self._term:
            raise ReplicationError("replicator term is stale")

        response = self._matching_response(
            peer,
            trace_start,
            kind="raft-install-snapshot-response",
            requested_last_included_index=snapshot.last_included_index,
            request_id=request_id,
        )
        if response is None:
            raise ReplicationResponseMissing(
                f"no InstallSnapshot response from {peer!r} for leader "
                f"{self.leader.node_id!r} in term {self._term}"
            )
        if not bool(response.details["success"]):
            return False

        match_index = int(response.details["last_included_index"])
        self._advance_peer(peer, progress, match_index, source="snapshot")
        self.advance_commit_index()
        return True

    def _advance_peer(
        self,
        peer: str,
        progress: PeerReplicationProgress,
        match_index: int,
        *,
        source: str,
    ) -> None:
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
            source=source,
        )

    def _matching_response(
        self,
        peer: str,
        trace_start: int,
        *,
        kind: str,
        requested_last_included_index: int | None = None,
        request_id: int | None = None,
    ):
        for record in reversed(self.sim.trace[trace_start:]):
            if record.kind != kind:
                continue
            if record.details.get("leader") != self.leader.node_id:
                continue
            if record.details.get("follower") != peer:
                continue
            if int(record.details.get("term", -1)) != self._term:
                continue
            if (
                requested_last_included_index is not None
                and int(record.details.get("requested_last_included_index", -1))
                != requested_last_included_index
            ):
                continue
            if request_id is not None and int(record.details.get("request_id", -1)) != request_id:
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
