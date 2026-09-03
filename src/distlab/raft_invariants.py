from __future__ import annotations

from dataclasses import dataclass

from .raft import LogEntry, RaftCluster, RaftNode, RaftRole


class LeaderCompletenessViolation(AssertionError):
    """Raised when a higher-term leader is missing an observed committed entry."""


@dataclass(frozen=True, slots=True)
class CommittedEntryObservation:
    index: int
    entry: LogEntry
    committed_in_term: int
    leader_id: str


class LeaderCompletenessChecker:
    """Executable Raft Leader Completeness assertion for deterministic tests.

    The checker records committed log positions at the moment a leader advances
    its commit index. Any leader in a strictly higher term must contain the same
    entry at every previously observed committed index.
    """

    def __init__(self) -> None:
        self._committed: dict[int, CommittedEntryObservation] = {}

    @property
    def committed_entries(self) -> tuple[CommittedEntryObservation, ...]:
        return tuple(self._committed[index] for index in sorted(self._committed))

    def observe_commit(
        self,
        leader: RaftNode,
        *,
        previous_commit_index: int = 0,
    ) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ValueError("committed entries must be observed from a leader")
        if previous_commit_index < 0:
            raise ValueError("previous_commit_index must be non-negative")
        if previous_commit_index > leader.commit_index:
            raise ValueError("previous_commit_index cannot exceed leader commit index")

        for index in range(previous_commit_index + 1, leader.commit_index + 1):
            entry = leader.log[index - 1]
            existing = self._committed.get(index)
            if existing is not None:
                if existing.entry != entry:
                    raise LeaderCompletenessViolation(
                        "committed entry changed at "
                        f"index {index}: observed {existing.entry!r}, now {entry!r}"
                    )
                continue
            self._committed[index] = CommittedEntryObservation(
                index=index,
                entry=entry,
                committed_in_term=leader.current_term,
                leader_id=leader.node_id,
            )

        self.assert_leader_node(leader)

    def assert_leader_node(self, leader: RaftNode) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ValueError("node must currently be a leader")
        self.assert_leader_log(
            term=leader.current_term,
            node_id=leader.node_id,
            log=leader.log,
        )

    def assert_leader_log(
        self,
        *,
        term: int,
        node_id: str,
        log: tuple[LogEntry, ...],
    ) -> None:
        if term < 0:
            raise ValueError("leader term must be non-negative")
        if not all(isinstance(entry, LogEntry) for entry in log):
            raise TypeError("leader log must contain only LogEntry values")

        for observation in self.committed_entries:
            if term <= observation.committed_in_term:
                continue
            if len(log) < observation.index:
                raise LeaderCompletenessViolation(
                    "Leader Completeness violated: "
                    f"leader {node_id!r} in term {term} is missing committed "
                    f"index {observation.index} from term {observation.committed_in_term}"
                )
            actual = log[observation.index - 1]
            if actual != observation.entry:
                raise LeaderCompletenessViolation(
                    "Leader Completeness violated: "
                    f"leader {node_id!r} in term {term} has {actual!r} at "
                    f"committed index {observation.index}, expected {observation.entry!r}"
                )

    def assert_recorded_leaders(self, cluster: RaftCluster) -> None:
        for term, node_id in sorted(cluster.leaders_by_term.items()):
            self.assert_leader_log(
                term=term,
                node_id=node_id,
                log=cluster.node(node_id).log,
            )


class RaftSafetyHarness:
    """Checkpoint core Raft safety properties across deterministic lifecycles.

    A checkpoint observes every currently committed leader prefix, validates all
    recorded leaders against Leader Completeness, and checks Log Matching. Tests
    and scenario runners should checkpoint after elections, replication/commit
    advancement, crash/restart boundaries, and leader replacement.
    """

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.leader_completeness = LeaderCompletenessChecker()
        self._observed_commit_index: dict[str, int] = {
            node_id: 0 for node_id in cluster.node_ids
        }

    def checkpoint(self) -> None:
        for node_id in self.cluster.node_ids:
            node = self.cluster.node(node_id)
            if node.role is not RaftRole.LEADER:
                continue
            previous_commit_index = self._observed_commit_index[node_id]
            if node.commit_index < previous_commit_index:
                previous_commit_index = 0
            self.leader_completeness.observe_commit(
                node,
                previous_commit_index=previous_commit_index,
            )
            self._observed_commit_index[node_id] = node.commit_index

        self.leader_completeness.assert_recorded_leaders(self.cluster)
        self.cluster.assert_log_matching()
