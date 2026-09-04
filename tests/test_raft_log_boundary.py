import pytest

from distlab.raft import AppendEntries, LogEntry, RaftCluster, RequestVote
from distlab.simulator import Simulator


def test_compacted_log_boundary_survives_restart_and_drives_absolute_tail() -> None:
    sim = Simulator()
    sim.persistent_state["n1"].update(
        {
            "current_term": 4,
            "log_base_index": 5,
            "log_base_term": 2,
            "log": (LogEntry(term=3, command="x"),),
        }
    )
    cluster = RaftCluster(sim, ("n1",))
    node = cluster.node("n1")

    assert node.log_base_index == 5
    assert node.log_base_term == 2
    assert node.last_log_index == 6
    assert node.last_log_term == 3
    assert node.log_view.term_at(5) == 2
    assert node.log_view.entry_at(6) == LogEntry(term=3, command="x")

    sim.crash("n1")
    sim.restart("n1")

    assert node.log_base_index == 5
    assert node.log_base_term == 2
    assert node.last_log_index == 6
    assert node.last_log_term == 3


def test_vote_freshness_uses_absolute_tail_after_compaction() -> None:
    sim = Simulator()
    sim.persistent_state["n2"].update(
        {
            "current_term": 4,
            "log_base_index": 5,
            "log_base_term": 2,
            "log": (LogEntry(term=3, command="x"),),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        RequestVote(
            term=5,
            candidate_id="n1",
            last_log_index=5,
            last_log_term=2,
        ),
    )
    sim.run(max_events=1)

    voter = cluster.node("n2")
    assert voter.current_term == 5
    assert voter.voted_for is None
    assert voter.last_log_index == 6
    assert voter.last_log_term == 3


def test_append_entries_merges_at_durable_compacted_boundary() -> None:
    sim = Simulator()
    full_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=1, command="b"),
        LogEntry(term=2, command="c"),
        LogEntry(term=2, command="d"),
        LogEntry(term=2, command="e"),
        LogEntry(term=3, command="x"),
    )
    sim.persistent_state["n1"].update({"current_term": 4, "log": full_log})
    sim.persistent_state["n2"].update(
        {
            "current_term": 4,
            "log_base_index": 5,
            "log_base_term": 2,
            "log": (LogEntry(term=3, command="x"),),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=4,
            leader_id="n1",
            prev_log_index=5,
            prev_log_term=2,
            entries=(
                LogEntry(term=3, command="x"),
                LogEntry(term=4, command="y"),
            ),
        ),
    )
    sim.run(max_events=1)

    follower = cluster.node("n2")
    assert follower.log_base_index == 5
    assert follower.log_base_term == 2
    assert follower.log == (
        LogEntry(term=3, command="x"),
        LogEntry(term=4, command="y"),
    )
    assert follower.last_log_index == 7
    assert follower.last_log_term == 4
    cluster.assert_log_matching()


def test_invalid_persistent_compaction_boundary_is_rejected() -> None:
    sim = Simulator()
    sim.persistent_state["n1"].update(
        {
            "log_base_index": -1,
            "log_base_term": 0,
            "log": (),
        }
    )

    with pytest.raises(ValueError, match="base index must be non-negative"):
        RaftCluster(sim, ("n1",))
