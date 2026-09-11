from __future__ import annotations

from dataclasses import dataclass

from .log_index import RaftLogView
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
    base_index: int = 0
    base_term: int = 0


class ElectionSafetyChecker:
    """Executable Election Safety assertion across deterministic lifecycle checkpoints."""

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

    Observations use absolute-index ``RaftLogView`` semantics. A leader may discard
    a prefix through legitimate compaction, but its compacted boundary must advance
    monotonically, its absolute last index may not shrink, the boundary term must
    agree with any previously observed entry at that index, and every retained
    overlapping entry must remain identical within the same leadership epoch.
    """

    def __init__(self) -> None:
        self._logs_by_leadership: dict[tuple[int, str], RaftLogView] = {}

    @property
    def observations(self) -> tuple[LeaderLogObservation, ...]:
        return tuple(
            LeaderLogObservation(
                term=term,
                leader_id=leader_id,
                log=log.entries,
                base_index=log.base_index,
                base_term=log.base_term,
            )
            for (term, leader_id), log in sorted(self._logs_by_leadership.items())
        )

    def observe_leader(self, leader: RaftNode) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ValueError("node must currently be a leader")
        key = (leader.current_term, leader.node_id)
        current = leader.log_view
        previous = self._logs_by_leadership.get(key)
        if previous is not None:
            if current.base_index < previous.base_index:
                raise LeaderAppendOnlyViolation(
                    "Leader Append-Only violated: "
                    f"leader {leader.node_id!r} in term {leader.current_term} "
                    f"regressed compacted boundary from {previous.base_index} "
                    f"to {current.base_index}"
                )
            if current.last_index < previous.last_index:
                raise LeaderAppendOnlyViolation(
                    "Leader Append-Only violated: "
                    f"leader {leader.node_id!r} in term {leader.current_term} "
                    f"shrunk its log in absolute index space from {previous.last_index} "
                    f"to {current.last_index}"
                )
            if previous.base_index < current.base_index <= previous.last_index:
                expected_term = previous.term_at(current.base_index)
                if current.base_term != expected_term:
                    raise LeaderAppendOnlyViolation(
                        "Leader Append-Only violated: "
                        f"leader {leader.node_id!r} in term {leader.current_term} "
                        f"changed compacted boundary term at index {current.base_index} "
                        f"from {expected_term} to {current.base_term}"
                    )
            overlap_start = max(previous.first_retained_index, current.first_retained_index)
            for index in range(overlap_start, previous.last_index + 1):
                if current.entry_at(index) != previous.entry_at(index):
                    raise LeaderAppendOnlyViolation(
                        "Leader Append-Only violated: "
                        f"leader {leader.node_id!r} in term {leader.current_term} "
                        f"overwrote an entry previously observed at absolute index {index}"
                    )
        self._logs_by_leadership[key] = current

    def assert_cluster(self, cluster: RaftCluster) -> None:
        for node_id in cluster.node_ids:
            node = cluster.node(node_id)
            if node.role is RaftRole.LEADER:
                self.observe_leader(node)


class LeaderCompletenessChecker:
    """Executable Raft Leader Completeness assertion for deterministic tests."""

    def __init__(self) -> None:
        self._committed: dict[int, CommittedEntryObservation] = {}

    @property
    def committed_entries(self) -> tuple[CommittedEntryObservation, ...]:
        return tuple(self._committed[index] for index in sorted(self._committed))

    def observe_commit(self, leader: RaftNode, *, previous_commit_index: int = 0) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ValueError("committed entries must be observed from a leader")
        if previous_commit_index < 0:
            raise ValueError("previous_commit_index must be non-negative")
        if previous_commit_index > leader.commit_index:
            raise ValueError("previous_commit_index cannot exceed leader commit index")
        log = leader.log_view
        first_observable = max(previous_commit_index + 1, log.first_retained_index)
        for index in range(first_observable, leader.commit_index + 1):
            entry = log.entry_at(index)
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
        self.assert_leader_view(
            term=leader.current_term, node_id=leader.node_id, log=leader.log_view
        )

    def assert_leader_log(
        self, *, term: int, node_id: str, log: tuple[LogEntry, ...]
    ) -> None:
        self.assert_leader_view(
            term=term, node_id=node_id, log=RaftLogView.uncompacted(log)
        )

    def assert_leader_view(
        self, *, term: int, node_id: str, log: RaftLogView
    ) -> None:
        if term < 0:
            raise ValueError("leader term must be non-negative")
        for observation in self.committed_entries:
            if term <= observation.committed_in_term:
                continue
            if observation.index < log.base_index:
                continue
            if observation.index == log.base_index:
                if log.base_term != observation.entry.term:
                    raise LeaderCompletenessViolation(
                        "Leader Completeness violated: "
                        f"leader {node_id!r} in term {term} has compacted boundary "
                        f"term {log.base_term} at committed index {observation.index}, "
                        f"expected term {observation.entry.term}"
                    )
                continue
            if log.last_index < observation.index:
                raise LeaderCompletenessViolation(
                    "Leader Completeness violated: "
                    f"leader {node_id!r} in term {term} is missing committed "
                    f"index {observation.index} from term {observation.committed_in_term}"
                )
            actual = log.entry_at(observation.index)
            if actual != observation.entry:
                raise LeaderCompletenessViolation(
                    "Leader Completeness violated: "
                    f"leader {node_id!r} in term {term} has {actual!r} at "
                    f"committed index {observation.index}, expected {observation.entry!r}"
                )

    def assert_recorded_leaders(self, cluster: RaftCluster) -> None:
        for term, node_id in sorted(cluster.leaders_by_term.items()):
            self.assert_leader_view(
                term=term, node_id=node_id, log=cluster.node(node_id).log_view
            )


class RaftSafetyHarness:
    """Checkpoint core Raft safety properties across deterministic lifecycles."""

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
                node, previous_commit_index=previous_commit_index
            )
            self._observed_commit_index[node_id] = node.commit_index
        self.leader_completeness.assert_recorded_leaders(self.cluster)
        self.cluster.assert_log_matching()
        self.state_machine.assert_state_machine_safety()
