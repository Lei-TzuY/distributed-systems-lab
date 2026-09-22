from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_isolated_timeout_retries_pre_vote_without_advancing_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 50, "n2": 60, "n3": 5},
    )
    harness = RaftSafetyHarness(cluster)
    sim.partition(("n3",), ("n1", "n2"))

    sim.run(max_events=3)

    candidate = cluster.node("n3")
    assert candidate.current_term == 0
    assert candidate.voted_for is None
    assert candidate.role is RaftRole.FOLLOWER
    starts = [
        record
        for record in sim.trace
        if record.kind == "raft-pre-vote-start" and record.details["node"] == "n3"
    ]
    assert [record.details["prospective_term"] for record in starts] == [1, 1, 1]
    assert not any(
        record.kind == "raft-election-start" and record.details["node"] == "n3"
        for record in sim.trace
    )
    harness.checkpoint()

    sim.heal_partition(("n3",), ("n1", "n2"))
    sim.run(max_events=4)

    assert candidate.current_term == 1
    assert candidate.voted_for == "n3"
    assert candidate.role is RaftRole.CANDIDATE
    persisted_election = [
        record
        for record in sim.trace
        if record.kind == "raft-persist-term-vote"
        and record.details["node"] == "n3"
        and record.details["term"] == 1
        and record.details["voted_for"] == "n3"
    ]
    assert len(persisted_election) == 1
    harness.checkpoint()

    sim.run(max_events=4)

    assert candidate.role is RaftRole.LEADER
    assert cluster.leaders_by_term == {1: "n3"}
    harness.checkpoint()


def test_pre_vote_checks_log_freshness_without_mutating_receiver_term_or_vote() -> None:
    sim = Simulator()
    for node_id in ("n1", "n2"):
        sim.persistent_state[node_id].update(
            {
                "current_term": 3,
                "voted_for": "n2",
                "log": (LogEntry(term=3, command="committed"),),
            }
        )
    sim.persistent_state["n3"].update(
        {
            "current_term": 3,
            "voted_for": None,
            "log": (),
        }
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    harness = RaftSafetyHarness(cluster)

    cluster.node("n3").start_pre_vote()
    sim.run()

    assert cluster.node("n3").current_term == 3
    assert cluster.node("n3").voted_for is None
    assert cluster.node("n3").role is RaftRole.FOLLOWER
    assert cluster.node("n1").current_term == 3
    assert cluster.node("n1").voted_for == "n2"
    assert cluster.node("n2").current_term == 3
    assert cluster.node("n2").voted_for == "n2"
    votes = [
        record
        for record in sim.trace
        if record.kind == "raft-pre-vote"
        and record.details["candidate"] == "n3"
    ]
    assert {record.details["voter"] for record in votes} == {"n1", "n2"}
    assert all(record.details["granted"] is False for record in votes)
    assert all(record.details["log_up_to_date"] is False for record in votes)
    assert not any(
        record.kind == "raft-election-start" and record.details["node"] == "n3"
        for record in sim.trace
    )
    harness.checkpoint()
