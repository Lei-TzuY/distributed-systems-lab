from distlab.raft import AppendEntries, RaftCluster, RaftRole
from distlab.simulator import Simulator


def test_earliest_deterministic_timeout_starts_election() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 5, "n2": 10, "n3": 15},
    )

    sim.run(max_events=1)

    assert sim.time == 5
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n1").role is RaftRole.CANDIDATE
    assert cluster.node("n2").current_term == 0
    assert cluster.node("n3").current_term == 0


def test_timeout_driven_candidate_can_reach_leader() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 5, "n2": 20, "n3": 30},
    )

    sim.run(max_events=4)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.leaders_by_term == {1: "n1"}


def test_append_entries_resets_timeout_and_stale_deadline_is_ignored() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2"),
        election_timeouts={"n1": 100, "n2": 5},
    )

    sim.send("n1", "n2", AppendEntries(term=1, leader_id="n1"))
    sim.run(max_events=3)

    assert sim.time == 5
    assert cluster.node("n2").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    stale = [record for record in sim.trace if record.kind == "raft-election-timeout-stale"]
    assert stale
    assert stale[-1].details["node"] == "n2"

    sim.run(max_events=1)

    assert sim.time == 6
    assert cluster.node("n2").current_term == 2
    assert cluster.node("n2").role is RaftRole.CANDIDATE


def test_single_node_leader_invalidates_candidate_timeout() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",), election_timeouts={"n1": 5})

    sim.run()

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.leaders_by_term == {1: "n1"}
    stale = [record for record in sim.trace if record.kind == "raft-election-timeout-stale"]
    assert len(stale) == 1


def test_election_timeout_configuration_requires_every_node() -> None:
    sim = Simulator()

    try:
        RaftCluster(sim, ("n1", "n2"), election_timeouts={"n1": 5})
    except ValueError as exc:
        assert "every Raft node" in str(exc)
    else:
        raise AssertionError("partial election timeout configuration must be rejected")


def test_election_timeout_configuration_rejects_non_positive_values() -> None:
    sim = Simulator()

    try:
        RaftCluster(sim, ("n1",), election_timeouts={"n1": 0})
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("non-positive election timeout must be rejected")
