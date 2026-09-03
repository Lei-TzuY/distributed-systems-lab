from __future__ import annotations

from dataclasses import dataclass

from .raft import ElectionSafetyViolation, LogEntry, RaftCluster, RaftNode, RaftRole
from .state_machine import StateMachineApplier


class LeaderAppendOnlyViolation(AssertionError):
    """Raised when a leader deletes or overwrites an entry from its observed log."""


class LeaderCompletenessViolation(AssertionError):
    """Raised when a higher-term leader is missing an observed committed entry."""


@dataclass(frozen=True, slots=True)
class CommittedEntryObservation:
    index: int
    entry: LogEntry
    committed_in_term: int
    leader_id: str


@dataclass(frozen=True, slots=True)
class LeaderLogObservation:
    term: int
    leader_id: str
    log: tuple[LogEntry, ...]


class ElectionSafetyChecker:
    """Executable Election Safety assertion across deterministic lifecycle checkpoints.

    The checker independently remembers the first leader observed for every term.
    Each checkpoint validates both the cluster's recorded leaders and all nodes that
    currently expose the leader role. Seeing a different leader for an already
    observed term raises ``ElectionSafetyViolation`` immediately.
    """

    def __init__(self) -> None:
        self._leaders_by_term: dict[int, str] = {}

    @property
    def leaders_by_term(self) -> dict[int, str]:
        return dict(self._leaders_by_term)

    def observe_leader(self, *, term: int, node_id: str) -> None:
        if term < 0:
            raise ValueError("leader term must be non-negative")
        existing = self._leaders_by_term.get(term)
        if existing is not None and existing != node_id:
            raise ElectionSafetyViolation(
                f"Election Safety violated in term {term}: {existing!r} and {node_id!r}"
            )
        self._leaders_by_term[term] = node_id

    def assert_cluster(self, cluster: RaftCluster) -> None:
        for term, node_id in sorted(cluster.leaders_by_term.items()):
            self.observe_leader(term=term, node_id=node_id)

        for node_id in cluster.node_ids:
            node = cluster.node(node_id)
            if node.role is RaftRole.LEADER:
                self.observe_leader(term=node.current_term, node_id=node_id)


class LeaderAppendOnlyChecker:
    """Executable Leader Append-Only assertion across leader checkpoints.

    For each observed ``(term, leader)`` epoch, the checker remembers the longest
    log seen while that node exposes the leader role. Later checkpoints in the same
    leadership epoch must retain that entire prefix. Former leaders are not checked
    after stepping down because Raft followers may legitimately replace uncommitted
    suffixes through AppendEntries conflict resolution.
    """

    def __init__(self) -> None:
        self._logs_by_leadership: dict[tuple[int, str], tuple[LogEntry, ...]] = {}

    @property
    def observations(self) -> tuple[LeaderLogObservation, ...]:
        return tuple(
            LeaderLogObservation(term=term, leader_id=leader_id, log=log)
            for (term, leader_id), log in sorted(self._logs_by_leadership.items())
        )

    def observe_leader(self, leader: RaftNode) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ValueError("node must currently be a leader")
        key = (leader.current_term, leader.node_id)
        current = leader.log
        previous = self._logs_by_leadership.get(key)
        if previous is not None:
            if len(current) < len(previous):
                raise LeaderAppendOnlyViolation(
                    "Leader Append-Only violated: "
                    f"leader {leader.node_id!r} in term {leader.current_term} "
                    f"shrunk its log from {len(previous)} to {len(current)} entries"
                )
            if current[: len(previous)] != previous:
                raise LeaderAppendOnlyViolation(
                    "Leader Append-Only violated: "
                    f"leader {leader.node_id!r} in term {leader.current_term} "
                    "overwrote an entry in its previously observed log prefix"
                )
        self._logs_by_leadership[key] = current

    def assert_cluster(self, cluster: RaftCluster) -> None:
        for node_id in cluster.node_ids:
            node = cluster.node(node_id)
            if node.role is RaftRole.LEADER:
                self.observe_leader(node)


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

    A checkpoint validates Election Safety and Leader Append-Only, observes every
    currently committed leader prefix, validates all recorded leaders against
    Leader Completeness, checks Log Matching, and re-validates durable applied
    histories against State Machine Safety. Tests and scenario runners should
    checkpoint after elections, leader appends, replication/commit advancement,
    state-machine application, crash/restart boundaries, and leader replacement.
    """

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.election_safety = ElectionSafetyChecker()
        self.leader_append_only = LeaderAppendOnlyChecker()
        self.leader_completeness = LeaderCompletenessChecker()
        self.state_machine = StateMachineApplier(cluster)
        self._observed_commit_index: dict[str, int] = {
            node_id: 0 for node_id in cluster.node_ids
        }

    def checkpoint(self) -> None:
        self.election_safety.assert_cluster(self.cluster)
        self.leader_append_only.assert_cluster(self.cluster)

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
        self.state_machine.assert_state_machine_safety()
