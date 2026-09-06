from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .membership import ReconfigurableRaftCluster
from .membership_replication import MembershipAwareLeaderReplicator
from .raft import RaftNode, RaftRole
from .replication import LeaderReplicator, ReplicationError

if TYPE_CHECKING:
    from .snapshot_transport import SnapshotTransport


class LeadershipTransferError(RuntimeError):
    """Base error for deterministic leadership-transfer failures."""


class InvalidLeadershipTransferTarget(LeadershipTransferError):
    """Raised when the requested transferee cannot become leader."""


class LeadershipTransferTargetUnavailable(LeadershipTransferError):
    """Raised when the requested transferee is currently crashed."""


class LeadershipTransferIncomplete(LeadershipTransferError):
    """Raised when catch-up or the transferee election cannot complete."""


@dataclass(frozen=True, slots=True)
class LeadershipTransferResult:
    previous_leader_id: str
    new_leader_id: str
    previous_term: int
    new_term: int
    commit_index: int


class LeadershipTransfer:
    """Move leadership to a caught-up peer through a deterministic Raft election.

    This controller deliberately reuses the normal replication and RequestVote paths:
    the requested transferee is first brought to the leader's complete log/commit
    prefix, then starts the next-term election. No role or term is assigned directly.
    """

    def __init__(
        self,
        leader: RaftNode,
        *,
        snapshot_transport: SnapshotTransport | None = None,
    ) -> None:
        self.leader = leader
        self.sim = leader.sim
        self.snapshot_transport = snapshot_transport

    def transfer(
        self,
        transferee_id: str,
        *,
        max_replication_attempts: int = 8,
    ) -> LeadershipTransferResult:
        if max_replication_attempts <= 0:
            raise ValueError("max_replication_attempts must be positive")
        self._require_current_leader()
        if transferee_id == self.leader.node_id or transferee_id not in self.leader.peers:
            raise InvalidLeadershipTransferTarget(
                "leadership transferee must be a distinct peer of the current leader"
            )
        cluster = self.leader.cluster
        if isinstance(cluster, ReconfigurableRaftCluster):
            configuration = cluster.voting_configuration
            if not cluster.is_voter(transferee_id):
                raise InvalidLeadershipTransferTarget(
                    f"leadership transferee {transferee_id!r} must be an active voter"
                )
            if (
                configuration.new_voters is not None
                and transferee_id not in configuration.new_voters
            ):
                raise InvalidLeadershipTransferTarget(
                    f"leadership transferee {transferee_id!r} must belong to the new voter "
                    "configuration during joint consensus"
                )
        if not self.sim.is_alive(transferee_id):
            raise LeadershipTransferTargetUnavailable(
                f"leadership transferee {transferee_id!r} is not live"
            )

        previous_term = self.leader.current_term
        target = cluster.node(transferee_id)
        self.sim._record(
            "raft-leadership-transfer-start",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=previous_term,
            leader_last_log_index=self.leader.last_log_index,
            leader_commit_index=self.leader.commit_index,
        )

        if isinstance(cluster, ReconfigurableRaftCluster):
            replicator = MembershipAwareLeaderReplicator(
                self.leader,
                snapshot_transport=self.snapshot_transport,
            )
        else:
            replicator = LeaderReplicator(
                self.leader,
                snapshot_transport=self.snapshot_transport,
            )
        try:
            recovered = replicator.recover_peer(
                transferee_id,
                max_attempts=max_replication_attempts,
            )
        except ReplicationError as exc:
            self._record_failure(transferee_id, previous_term, stage="catch-up", reason=str(exc))
            raise LeadershipTransferIncomplete(
                f"failed to catch up leadership transferee {transferee_id!r}"
            ) from exc
        if not recovered:
            self._record_failure(
                transferee_id,
                previous_term,
                stage="catch-up",
                reason="replication attempt budget exhausted",
            )
            raise LeadershipTransferIncomplete(
                f"leadership transferee {transferee_id!r} did not catch up"
            )

        if (
            target.last_log_index != self.leader.last_log_index
            or target.last_log_term != self.leader.last_log_term
            or target.commit_index < self.leader.commit_index
        ):
            self._record_failure(
                transferee_id,
                previous_term,
                stage="catch-up",
                reason="transferee durable prefix does not match leader",
            )
            raise LeadershipTransferIncomplete(
                "leadership transferee does not hold the leader's complete committed prefix"
            )

        self.sim._record(
            "raft-leadership-transfer-ready",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=previous_term,
            last_log_index=target.last_log_index,
            commit_index=target.commit_index,
        )

        target.start_election()
        event_budget = max(4, len(cluster.node_ids) * 4)
        for _ in range(event_budget):
            if self._transfer_elected(target, previous_term):
                break
            if self.sim.run(max_events=1) == 0:
                break

        if not self._transfer_elected(target, previous_term):
            self._record_failure(
                transferee_id,
                previous_term,
                stage="election",
                reason="transferee did not become leader and retire the source leader",
            )
            raise LeadershipTransferIncomplete(
                f"leadership transferee {transferee_id!r} did not win the next-term election"
            )

        result = LeadershipTransferResult(
            previous_leader_id=self.leader.node_id,
            new_leader_id=transferee_id,
            previous_term=previous_term,
            new_term=target.current_term,
            commit_index=target.commit_index,
        )
        self.sim._record(
            "raft-leadership-transfer-complete",
            previous_leader=result.previous_leader_id,
            new_leader=result.new_leader_id,
            previous_term=result.previous_term,
            new_term=result.new_term,
            commit_index=result.commit_index,
        )
        return result

    def _transfer_elected(self, target: RaftNode, previous_term: int) -> bool:
        return (
            target.role is RaftRole.LEADER
            and target.current_term > previous_term
            and self.leader.role is not RaftRole.LEADER
            and self.leader.current_term >= target.current_term
        )

    def _require_current_leader(self) -> None:
        if not self.sim.is_alive(self.leader.node_id):
            raise LeadershipTransferError("leadership transfer requires a live source leader")
        if self.leader.role is not RaftRole.LEADER:
            raise LeadershipTransferError("leadership transfer requires current leader role")

    def _record_failure(
        self,
        transferee_id: str,
        term: int,
        *,
        stage: str,
        reason: str,
    ) -> None:
        self.sim._record(
            "raft-leadership-transfer-failed",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=term,
            stage=stage,
            reason=reason,
        )
