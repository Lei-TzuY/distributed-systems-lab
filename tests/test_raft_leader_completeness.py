import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import LeaderCompletenessChecker, LeaderCompletenessViolation
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _committed_leader() -> tuple[Simulator, RaftCluster, LeaderReplicator]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="old"),
        LogEntry(term=3, command="current"),
    )
    sim.persistent_state["n2"]["log"] = (LogEntry(term=1, command="old"),)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 3

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 2
    return sim, cluster, replicator


def test_observe_commit_records_entire_newly_committed_prefix() -> None:
    _, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()

    checker.observe_commit(cluster.node("n1"))

    assert [(record.index, record.entry.command, record.committed_in_term) for record in checker.committed_entries] == [
        (1, "old", 3),
        (2, "current", 3),
    ]


def test_higher_term_leader_with_committed_prefix_satisfies_invariant() -> None:
    _, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()
    checker.observe_commit(cluster.node("n1"))

    checker.assert_leader_log(
        term=4,
        node_id="n2",
        log=cluster.node("n2").log,
    )


def test_higher_term_leader_missing_committed_entry_is_rejected() -> None:
    _, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()
    checker.observe_commit(cluster.node("n1"))

    with pytest.raises(LeaderCompletenessViolation, match="missing committed index 2"):
        checker.assert_leader_log(
            term=4,
            node_id="stale",
            log=(LogEntry(term=1, command="old"),),
        )


def test_higher_term_leader_with_conflicting_committed_entry_is_rejected() -> None:
    _, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()
    checker.observe_commit(cluster.node("n1"))

    with pytest.raises(LeaderCompletenessViolation, match="committed index 2"):
        checker.assert_leader_log(
            term=4,
            node_id="corrupt",
            log=(
                LogEntry(term=1, command="old"),
                LogEntry(term=3, command="different"),
            ),
        )


def test_same_term_leader_is_not_constrained_by_future_term_rule() -> None:
    _, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()
    checker.observe_commit(cluster.node("n1"))

    checker.assert_leader_log(term=3, node_id="same-term", log=())


def test_reobserving_a_changed_committed_entry_is_rejected() -> None:
    sim, cluster, _ = _committed_leader()
    checker = LeaderCompletenessChecker()
    leader = cluster.node("n1")
    checker.observe_commit(leader)

    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="old"),
        LogEntry(term=3, command="corrupt"),
    )

    with pytest.raises(LeaderCompletenessViolation, match="committed entry changed"):
        checker.observe_commit(leader)
