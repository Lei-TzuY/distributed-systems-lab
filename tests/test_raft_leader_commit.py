import pytest

from distlab.raft import AppendEntries, LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _elect_leader_with_current_term_log() -> tuple[Simulator, RaftCluster, LeaderReplicator]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = (LogEntry(term=3, command="set x=1"),)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.current_term == 3
    assert leader.role is RaftRole.LEADER
    return sim, cluster, LeaderReplicator(leader)


def test_committed_entry_is_propagated_on_subsequent_append_entries() -> None:
    _, cluster, replicator = _elect_leader_with_current_term_log()

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 1
    assert cluster.node("n1").commit_index == 1
    assert cluster.node("n2").commit_index == 0

    assert replicator.replicate("n3") is True
    assert cluster.node("n3").commit_index == 1

    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == 1


def test_follower_commit_index_is_monotonic() -> None:
    sim = Simulator()
    log = (LogEntry(term=1, command="a"), LogEntry(term=2, command="b"))
    sim.persistent_state["n2"]["log"] = log
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=2,
            leader_id="n1",
            prev_log_index=2,
            prev_log_term=2,
            leader_commit=2,
        ),
    )
    sim.run(max_events=1)
    assert cluster.node("n2").commit_index == 2

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=2,
            leader_id="n1",
            prev_log_index=2,
            prev_log_term=2,
            leader_commit=1,
        ),
    )
    sim.run(max_events=1)
    assert cluster.node("n2").commit_index == 2


def test_rejected_append_entries_cannot_advance_commit_index() -> None:
    sim = Simulator()
    sim.persistent_state["n2"]["log"] = (LogEntry(term=1, command="a"),)
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=2,
            leader_id="n1",
            prev_log_index=2,
            prev_log_term=1,
            leader_commit=1,
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").commit_index == 0


def test_commit_is_bounded_by_matched_prefix() -> None:
    sim = Simulator()
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="stale"),
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=3,
            leader_id="n1",
            prev_log_index=1,
            prev_log_term=1,
            leader_commit=2,
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").commit_index == 1


def test_append_entries_rejects_negative_leader_commit() -> None:
    with pytest.raises(ValueError, match="leader_commit"):
        AppendEntries(term=1, leader_id="n1", leader_commit=-1)
