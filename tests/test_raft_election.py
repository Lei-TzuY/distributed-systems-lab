from distlab.raft import RaftCluster, RaftRole, RequestVote
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_three_node_election_reaches_single_leader() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    sim.run()

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.leaders_by_term == {1: "n1"}
    assert len(cluster.node("n1").votes_received) >= 2


def test_simultaneous_candidates_cannot_both_win_same_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    cluster.node("n2").start_election()
    sim.run()

    assert list(cluster.leaders_by_term) == [1]
    assert cluster.leaders_by_term[1] in {"n1", "n2"}
    assert sum(node.role is RaftRole.LEADER for node in cluster.nodes.values()) == 1


def test_vote_persists_across_crash_and_restart() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    sim.send("n1", "n3", RequestVote(term=1, candidate_id="n1"))
    sim.run(max_events=1)
    assert cluster.node("n3").voted_for == "n1"

    sim.crash("n3")
    sim.restart("n3")
    sim.send("n2", "n3", RequestVote(term=1, candidate_id="n2"))
    sim.run()

    assert cluster.node("n3").voted_for == "n1"
    votes = [
        record
        for record in sim.trace
        if record.kind == "raft-vote" and record.details["voter"] == "n3"
    ]
    assert [(record.details["candidate"], record.details["granted"]) for record in votes] == [
        ("n1", True),
        ("n2", False),
    ]


def test_higher_term_request_resets_old_vote_and_steps_down() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER

    sim.send("n2", "n1", RequestVote(term=2, candidate_id="n2"))
    sim.run()

    assert cluster.node("n1").current_term == 2
    assert cluster.node("n1").voted_for == "n2"
    assert cluster.node("n1").role is RaftRole.FOLLOWER


def test_duplicate_vote_responses_do_not_inflate_majority() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DUPLICATE,
                    src="n2",
                    dst="n1",
                    ordinal=1,
                    extra_delay=0,
                ),
                FaultRule(FaultAction.DROP, src="n1", dst="n3", ordinal=1),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3", "n4", "n5"))

    cluster.node("n1").start_election()
    sim.crash("n4")
    sim.crash("n5")
    sim.run()

    assert cluster.node("n1").role is RaftRole.CANDIDATE
    assert cluster.node("n1").votes_received == frozenset({"n1", "n2"})
    assert cluster.leaders_by_term == {}


def test_crashed_node_cannot_start_election() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    sim.crash("n1")

    try:
        cluster.node("n1").start_election()
    except RuntimeError as exc:
        assert "cannot start an election" in str(exc)
    else:
        raise AssertionError("crashed nodes must not start elections")
