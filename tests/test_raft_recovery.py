from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.simulator import Simulator


def test_restart_preserves_persistent_term_vote_and_log_but_resets_volatile_state() -> None:
    sim = Simulator()
    sim.persistent_state["n1"].update(
        {
            "current_term": 4,
            "voted_for": "n2",
            "log": (LogEntry(term=2, command="a"), LogEntry(term=4, command="b")),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    node = cluster.node("n1")
    sim.volatile_state["n1"].update(
        {
            "role": RaftRole.CANDIDATE.value,
            "votes_received": {"n1", "n2"},
            "commit_index": 2,
        }
    )

    sim.crash("n1")
    sim.restart("n1")

    assert node.current_term == 4
    assert node.voted_for == "n2"
    assert node.log == (LogEntry(term=2, command="a"), LogEntry(term=4, command="b"))
    assert node.role is RaftRole.FOLLOWER
    assert node.votes_received == frozenset()
    assert node.commit_index == 0


def test_restarted_node_re_elects_from_persisted_term_and_log() -> None:
    sim = Simulator()
    sim.persistent_state["n1"].update(
        {
            "current_term": 3,
            "voted_for": "n1",
            "log": (LogEntry(term=1, command="x"), LogEntry(term=3, command="y")),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    sim.crash("n1")
    sim.restart("n1")
    cluster.node("n1").start_election()
    sim.run()

    leader = cluster.node("n1")
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 4
    assert leader.voted_for == "n1"
    assert leader.log == (LogEntry(term=1, command="x"), LogEntry(term=3, command="y"))
    assert cluster.leaders_by_term == {4: "n1"}

    election = next(
        record
        for record in sim.trace
        if record.kind == "raft-election-start" and record.details["node"] == "n1"
    )
    assert election.details["term"] == 4
    assert election.details["last_log_index"] == 2
    assert election.details["last_log_term"] == 3


def test_crash_discards_inflight_vote_response_and_restart_does_not_restore_votes() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    candidate = cluster.node("n1")

    candidate.start_election()
    sim.run(max_events=1)
    sim.crash("n1")
    sim.run()

    assert candidate.current_term == 1
    assert candidate.voted_for == "n1"
    assert candidate.votes_received == frozenset()

    sim.restart("n1")
    assert candidate.role is RaftRole.FOLLOWER
    assert candidate.votes_received == frozenset()
    assert candidate.commit_index == 0

    discarded = [
        record
        for record in sim.trace
        if record.kind == "discard-crashed" and record.details["dst"] == "n1"
    ]
    assert discarded


def test_restart_then_higher_term_election_never_reuses_old_vote() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    node = cluster.node("n1")

    node.start_election()
    sim.run()
    assert node.current_term == 1
    assert node.voted_for == "n1"

    sim.crash("n1")
    sim.restart("n1")
    node.start_election()
    sim.run()

    assert node.role is RaftRole.LEADER
    assert node.current_term == 2
    assert node.voted_for == "n1"
    assert cluster.leaders_by_term == {1: "n1", 2: "n1"}
