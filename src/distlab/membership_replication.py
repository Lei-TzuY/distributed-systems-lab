from __future__ import annotations

from .membership import ReconfigurableRaftCluster
from .replication import LeaderReplicator


class MembershipAwareLeaderReplicator(LeaderReplicator):
    """Leader replication whose commit rule follows the active voter configuration.

    Reconfigurable clusters may pre-provision learners that must not inflate the
    commit quorum. During joint consensus, a committed index must be acknowledged by
    independent majorities of both the old and new voter sets, matching the election
    quorum rule used by ``ReconfigurableRaftCluster``.
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
            if not cluster.voting_configuration.has_quorum(acknowledged):
                continue

            self._commit_index = index
            self.leader.advance_commit_index(index, source=self.leader.node_id)
            self.sim._record(
                "raft-commit-advance",
                leader=self.leader.node_id,
                term=self._term,
                previous_commit_index=previous,
                commit_index=index,
                replicas=len(acknowledged),
                acknowledged_voters=tuple(
                    sorted(acknowledged & cluster.voting_configuration.voters)
                ),
                quorum_mode=(
                    "joint" if cluster.voting_configuration.is_joint else "stable"
                ),
            )
            break

        return self._commit_index
