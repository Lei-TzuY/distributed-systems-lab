from __future__ import annotations

from dataclasses import dataclass

from .membership import MembershipChangeError, ReconfigurableRaftCluster
from .membership_replication import MembershipAwareLeaderReplicator
from .raft import LogEntry, RaftNode, RaftRole


@dataclass(frozen=True, slots=True)
class JointConsensusCommand:
    """Durable Raft command proposing transition to a new voter set."""

    new_voters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.new_voters:
            raise ValueError("joint consensus requires at least one new voter")
        if len(set(self.new_voters)) != len(self.new_voters):
            raise ValueError("new voter ids must be unique")


class ReplicatedMembershipTransition:
    """Drive one bounded stable-to-joint configuration transition through Raft.

    The proposal is first appended as an ordinary durable current-term log entry.
    The cluster continues to use the current stable configuration until that entry is
    committed by the membership-aware replicator. Only then is joint quorum semantics
    activated. Finalization into the new stable configuration is a later slice.
    """

    def __init__(self, leader: RaftNode) -> None:
        if not isinstance(leader.cluster, ReconfigurableRaftCluster):
            raise TypeError("membership transition requires a reconfigurable Raft cluster")
        self.leader = leader
        self.cluster = leader.cluster
        self.sim = leader.sim
        self._term = leader.current_term
        self._pending_index: int | None = None
        self._pending_command: JointConsensusCommand | None = None
        self._require_current_leader()

    @property
    def pending_index(self) -> int | None:
        return self._pending_index

    def propose_joint(self, new_voters: tuple[str, ...]) -> int:
        self._require_current_leader()
        if self._pending_index is not None:
            raise MembershipChangeError("a membership proposal is already pending")
        if self.cluster.voting_configuration.is_joint:
            raise MembershipChangeError("joint consensus is already active")

        proposed = frozenset(new_voters)
        if not proposed:
            raise ValueError("new voter configuration cannot be empty")
        if not proposed <= frozenset(self.cluster.node_ids):
            raise ValueError("new voters must be pre-provisioned cluster nodes")

        command = JointConsensusCommand(tuple(sorted(proposed)))
        previous = self.leader.log
        entry = LogEntry(term=self._term, command=command)
        self.sim.persistent_state[self.leader.node_id]["log"] = (*previous, entry)
        index = self.leader.log_view.last_index
        self._pending_index = index
        self._pending_command = command
        self.sim._record(
            "raft-membership-proposed",
            leader=self.leader.node_id,
            term=self._term,
            index=index,
            old_voters=tuple(sorted(self.cluster.voting_configuration.old_voters)),
            new_voters=command.new_voters,
        )

        if self.leader.log[: len(previous)] != previous:
            raise AssertionError("Leader Append-Only violated while appending membership proposal")
        if self.leader.log_view.entry_at(index) != entry:
            raise AssertionError("membership proposal was not durably appended")
        return index

    def activate_if_committed(self, replicator: MembershipAwareLeaderReplicator) -> bool:
        self._require_current_leader()
        if replicator.leader is not self.leader:
            raise MembershipChangeError("replicator must belong to the transition leader")
        index = self._pending_index
        command = self._pending_command
        if index is None or command is None:
            raise MembershipChangeError("no membership proposal is pending")
        if replicator.commit_index < index:
            return False
        if self.leader.log_view.entry_at(index).command != command:
            raise MembershipChangeError("pending membership entry changed before activation")

        self.cluster.begin_joint_consensus(self.leader.node_id, command.new_voters)
        self.sim._record(
            "raft-membership-committed",
            leader=self.leader.node_id,
            term=self._term,
            index=index,
            new_voters=command.new_voters,
        )
        self._pending_index = None
        self._pending_command = None
        return True

    def replicate_and_activate(self, *, max_attempts_per_peer: int = 2) -> bool:
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")
        self._require_current_leader()
        if self._pending_index is None:
            raise MembershipChangeError("no membership proposal is pending")

        replicator = MembershipAwareLeaderReplicator(self.leader)
        replicator.advance_commit_index()
        if self.activate_if_committed(replicator):
            return True

        current_voters = self.cluster.voting_configuration.old_voters
        for peer in sorted(current_voters - {self.leader.node_id}):
            if not self.sim.is_alive(peer):
                continue
            replicator.replicate(peer, max_attempts=max_attempts_per_peer)
            if self.activate_if_committed(replicator):
                return True
        return False

    def _require_current_leader(self) -> None:
        if not self.sim.is_alive(self.leader.node_id):
            raise MembershipChangeError("membership transition requires a live current leader")
        if self.leader.role is not RaftRole.LEADER or self.leader.current_term != self._term:
            raise MembershipChangeError("membership transition leader is no longer current")
