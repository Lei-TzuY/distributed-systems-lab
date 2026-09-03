import pytest

from distlab.raft import ElectionSafetyViolation, RaftCluster, RaftRole
from distlab.raft_invariants import ElectionSafetyChecker, RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_safety_harness_tracks_leaders_across_replacement_terms() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    harness = RaftSafetyHarness(cluster)

    cluster.node("n1").start_election()
    sim.run()
    harness.checkpoint()

    sim.crash("n1")
    cluster.node("n2").start_election()
    sim.run()
    harness.checkpoint()

    assert harness.election_safety.leaders_by_term == {1: "n1", 2: "n2"}
    assert cluster.node("n2").role is RaftRole.LEADER


def test_safety_harness_rejects_two_visible_leaders_in_same_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    harness = RaftSafetyHarness(cluster)

    cluster.node("n1").start_election()
    sim.run()
    harness.checkpoint()

    sim.persistent_state["n2"]["current_term"] = 1
    sim.volatile_state["n2"]["role"] = RaftRole.LEADER.value

    with pytest.raises(ElectionSafetyViolation, match="term 1"):
        harness.checkpoint()


def test_duplicate_vote_response_fault_does_not_fool_lifecycle_checker() -> None:
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
    harness = RaftSafetyHarness(cluster)

    cluster.node("n1").start_election()
    sim.crash("n4")
    sim.crash("n5")
    sim.run()
    harness.checkpoint()

    assert cluster.node("n1").role is RaftRole.CANDIDATE
    assert cluster.node("n1").votes_received == frozenset({"n1", "n2"})
    assert harness.election_safety.leaders_by_term == {}


def test_election_safety_checker_rejects_conflicting_observation_history() -> None:
    checker = ElectionSafetyChecker()
    checker.observe_leader(term=7, node_id="n1")

    with pytest.raises(ElectionSafetyViolation, match=r"n1.*n2"):
        checker.observe_leader(term=7, node_id="n2")
