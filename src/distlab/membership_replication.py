from __future__ import annotations

from .membership import MembershipChangeError, ReconfigurableRaftCluster, VotingConfiguration
from .replication import LeaderReplicator


class MembershipAwareLeaderReplicator(LeaderReplicator):
    """Leader replication whose commit rule follows the active voter configuration.

    Reconfigurable clusters may pre-provision learners that must not inflate the
    commit quorum. During joint consensus, a committed index must be acknowledged by
    independent majorities of both the old and new voter sets, matching the election
    quorum rule used by ``ReconfigurableRaftCluster``. A pending joint-consensus entry
    and any later entry whose commit would cover it use the proposed joint
    configuration, so the new voter majority has the transition before it becomes
    active.
    """

    def advance_commit_index(self) -> int:
        self._require_current_leader()
        cluster = self.leader.cluster
        if not isinstance(cluster, ReconfigurableRaftCluster):
            return super().advance_commit_index()

        previous = self._commit_index
        log = self.leader.log_view
        scan_floor = max(previous, log.base_index)

        for index in range(log.last_index, scan_floor, -1):
            if log.term_at(index) != self._term:
                continue
            acknowledged = {
                self.leader.node_id,
                *(
                    peer
                    for peer, progress in self._progress.items()
                    if progress.match_index >= index
                ),
            }
            configuration, quorum_mode = self._commit_configuration(index)
            if not configuration.has_quorum(acknowledged):
                continue

            self._commit_index = index
            self.leader.advance_commit_index(index, source=self.leader.node_id)
            self._persist_committed_membership(index)
            self.sim._record(
                "raft-commit-advance",
                leader=self.leader.node_id,
                term=self._term,
                previous_commit_index=previous,
                commit_index=index,
                replicas=len(acknowledged),
                acknowledged_voters=tuple(sorted(acknowledged & configuration.voters)),
                quorum_mode=quorum_mode,
            )
            break

        return self._commit_index

    def _commit_configuration(self, index: int) -> tuple[VotingConfiguration, str]:
        from .membership_log import JointConsensusCommand, StableConsensusCommand

        cluster = self.leader.cluster
        assert isinstance(cluster, ReconfigurableRaftCluster)
        active = cluster.voting_configuration
        log = self.leader.log_view
        first_uncommitted = max(self._commit_index + 1, log.first_retained_index)
        membership_commands = [
            log.entry_at(prefix_index).command
            for prefix_index in range(first_uncommitted, index + 1)
            if isinstance(
                log.entry_at(prefix_index).command,
                (JointConsensusCommand, StableConsensusCommand),
            )
        ]

        if not membership_commands:
            return active, "joint" if active.is_joint else "stable"
        if len(membership_commands) > 1:
            raise MembershipChangeError(
                "commit candidate covers multiple uncommitted membership commands"
            )

        command = membership_commands[0]
        node_ids = frozenset(cluster.node_ids)
        if isinstance(command, JointConsensusCommand):
            if active.is_joint:
                raise MembershipChangeError(
                    "cannot commit a second joint configuration while joint consensus is active"
                )
            new_voters = frozenset(command.new_voters)
            if not new_voters <= node_ids:
                raise MembershipChangeError("joint membership proposal references unknown nodes")
            return VotingConfiguration(active.old_voters, new_voters), "joint-proposal"

        if active.new_voters is None:
            raise MembershipChangeError(
                "cannot commit stable membership without active joint consensus"
            )
        voters = frozenset(command.voters)
        if not voters <= node_ids:
            raise MembershipChangeError("stable membership proposal references unknown nodes")
        if voters != active.new_voters:
            raise MembershipChangeError(
                "stable membership proposal does not match active joint new-voter set"
            )
        return active, "joint-finalize"

    def _persist_committed_membership(self, commit_index: int) -> None:
        """Durably record any membership command covered by this commit advance.

        Commit advancement and live configuration activation are intentionally
        separate operations. Persisting the watermark here closes the crash window
        between objective quorum commit and the transition controller observing it.
        """

        from .membership_log import (
            JointConsensusCommand,
            StableConsensusCommand,
            _persist_membership_commit_watermark,
        )

        cluster = self.leader.cluster
        assert isinstance(cluster, ReconfigurableRaftCluster)
        log = self.leader.log_view
        for index in range(max(1, log.first_retained_index), commit_index + 1):
            command = log.entry_at(index).command
            if not isinstance(command, (JointConsensusCommand, StableConsensusCommand)):
                continue
            _persist_membership_commit_watermark(cluster, index=index, command=command)
