import pytest

from distlab.raft import (
    AppendEntries,
    LogEntry,
    LogMatchingViolation,
    RaftCluster,
    RaftRole,
)
from distlab.simulator import Simulator


def test_append_entries_rejects_missing_previous_entry_without_mutating_log() -> None:
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
            entries=(LogEntry(term=2, command="b"),),
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").log == (LogEntry(term=1, command="a"),)
    append = next(record for record in sim.trace if record.kind == "raft-append-entries")
    assert append.details["success"] is False
    assert append.details["match_index"] == 0


def test_append_entries_rejects_previous_term_mismatch_without_mutating_log() -> None:
    sim = Simulator()
    original = (LogEntry(term=1, command="a"), LogEntry(term=3, command="old"))
    sim.persistent_state["n2"]["log"] = original
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=4,
            leader_id="n1",
            prev_log_index=2,
            prev_log_term=2,
            entries=(LogEntry(term=4, command="new"),),
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").log == original
    append = next(record for record in sim.trace if record.kind == "raft-append-entries")
    assert append.details["success"] is False


def test_append_entries_truncates_conflicting_suffix_and_appends_leader_entries() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
        LogEntry(term=4, command="c"),
    )
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=3, command="stale-b"),
        LogEntry(term=3, command="stale-c"),
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=4,
            leader_id="n1",
            prev_log_index=1,
            prev_log_term=1,
            entries=(
                LogEntry(term=2, command="b"),
                LogEntry(term=4, command="c"),
            ),
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").log == cluster.node("n1").log
    append = next(record for record in sim.trace if record.kind == "raft-append-entries")
    assert append.details["success"] is True
    assert append.details["match_index"] == 3
    cluster.assert_log_matching()


def test_matching_entries_are_retained_and_only_missing_suffix_is_appended() -> None:
    sim = Simulator()
    prefix = (LogEntry(term=1, command="a"), LogEntry(term=2, command="b"))
    sim.persistent_state["n1"]["log"] = (*prefix, LogEntry(term=3, command="c"))
    sim.persistent_state["n2"]["log"] = prefix
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=3,
            leader_id="n1",
            prev_log_index=1,
            prev_log_term=1,
            entries=(LogEntry(term=2, command="b"), LogEntry(term=3, command="c")),
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").log == cluster.node("n1").log
    persisted = [record for record in sim.trace if record.kind == "raft-persist-log"]
    assert len(persisted) == 1


def test_same_term_append_entries_steps_candidate_down() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n2").start_election()
    assert cluster.node("n2").role is RaftRole.CANDIDATE

    sim.send("n1", "n2", AppendEntries(term=1, leader_id="n1"))
    sim.run(max_events=1)

    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n2").current_term == 1


def test_higher_term_append_entries_advances_term_and_clears_vote() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n2").start_election()
    assert cluster.node("n2").voted_for == "n2"

    sim.send("n1", "n2", AppendEntries(term=4, leader_id="n1"))
    sim.run(max_events=1)

    assert cluster.node("n2").current_term == 4
    assert cluster.node("n2").voted_for is None
    assert cluster.node("n2").role is RaftRole.FOLLOWER


def test_log_matching_invariant_detects_same_index_term_with_divergent_prefix() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="left"),
        LogEntry(term=2, command="same-index-term"),
    )
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="right"),
        LogEntry(term=2, command="same-index-term"),
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    with pytest.raises(LogMatchingViolation, match="Log Matching violated"):
        cluster.assert_log_matching()


def test_same_index_and_term_cannot_identify_different_entries_during_merge() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="leader"),
    )
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="follower"),
    )
    RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        AppendEntries(
            term=2,
            leader_id="n1",
            prev_log_index=1,
            prev_log_term=1,
            entries=(LogEntry(term=2, command="leader"),),
        ),
    )

    with pytest.raises(LogMatchingViolation, match="same index/term"):
        sim.run(max_events=1)


def test_leader_can_send_explicit_append_entries_probe() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command="a"),)
    sim.persistent_state["n2"]["log"] = (LogEntry(term=1, command="a"),)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER

    cluster.node("n1").send_append_entries("n2", prev_log_index=1)
    sim.run()

    responses = [record for record in sim.trace if record.kind == "raft-append-response"]
    assert responses[-1].details["success"] is True
    assert responses[-1].details["match_index"] == 1
