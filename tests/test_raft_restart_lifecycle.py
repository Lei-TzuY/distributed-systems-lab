import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.simulator import Simulator


def test_restart_immediately_reconstructs_follower_state_and_records_durable_boundary() -> None:
    sim = Simulator()
    sim.persistent_state["n1"].update(
        {
            "current_term": 7,
            "voted_for": "n2",
            "log": (LogEntry(term=3, command="a"), LogEntry(term=7, command="b")),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    sim.volatile_state["n1"].update(
        {"role": RaftRole.LEADER.value, "votes_received": {"n1", "n2"}, "commit_index": 2}
    )

    sim.crash("n1")
    sim.restart("n1")

    node = cluster.node("n1")
    assert node.role is RaftRole.FOLLOWER
    assert node.votes_received == frozenset()
    assert node.commit_index == 0
    assert node.current_term == 7
    assert node.voted_for == "n2"
    assert node.log == (LogEntry(term=3, command="a"), LogEntry(term=7, command="b"))

    restart = next(record for record in sim.trace if record.kind == "raft-restart")
    assert restart.details == {
        "node": "n1",
        "term": 7,
        "voted_for": "n2",
        "log_base_index": 0,
        "log_base_term": 0,
        "last_log_index": 2,
        "last_log_term": 7,
    }


def test_restart_invalidates_pre_crash_election_timeout_and_schedules_a_fresh_one() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 5, "n2": 50, "n3": 60},
    )

    sim.crash("n1")
    sim.restart("n1")
    sim.run(max_events=2)

    node = cluster.node("n1")
    assert node.current_term == 1
    assert node.role is RaftRole.CANDIDATE
    stale = [
        record
        for record in sim.trace
        if record.kind == "raft-election-timeout-stale" and record.details["node"] == "n1"
    ]
    assert len(stale) == 1
    starts = [
        record
        for record in sim.trace
        if record.kind == "raft-election-start" and record.details["node"] == "n1"
    ]
    assert len(starts) == 1
    restart_resets = [
        record
        for record in sim.trace
        if record.kind == "raft-election-timeout-reset"
        and record.details["node"] == "n1"
        and record.details["reason"] == "restart"
    ]
    assert len(restart_resets) == 1


def test_restart_validates_persistent_log_before_processing_new_messages() -> None:
    sim = Simulator()
    RaftCluster(sim, ("n1", "n2", "n3"))
    sim.crash("n1")
    sim.persistent_state["n1"]["log"] = ("corrupt",)

    with pytest.raises(TypeError, match="persistent Raft log must contain only LogEntry values"):
        sim.restart("n1")
