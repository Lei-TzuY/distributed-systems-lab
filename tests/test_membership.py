import pytest

from distlab.membership import (
    MembershipChangeError,
    NonVoterElectionError,
    ReconfigurableRaftCluster,
)
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _cluster_with_learners() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_joint_consensus_requires_majority_of_old_and_new_voters() -> None:
    sim, cluster = _cluster_with_learners()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    cluster.begin_joint_consensus("n1", ("n3", "n4", "n5"))
    sim.crash("n4")
    sim.crash("n5")

    cluster.node("n3").start_election()
    sim.run()

    assert cluster.node("n3").role is RaftRole.CANDIDATE
    assert cluster.node("n3").votes_received >= frozenset({"n1", "n3"})
    assert cluster.leaders_by_term.get(2) is None
    assert not cluster.has_election_quorum(cluster.node("n3").votes_received)

    sim.restart("n4")
    cluster.node("n3").start_election()
    sim.run()

    assert cluster.node("n3").role is RaftRole.LEADER
    assert cluster.node("n3").current_term == 3
    assert cluster.has_election_quorum(cluster.node("n3").votes_received)
    assert cluster.leaders_by_term[3] == "n3"
    harness.checkpoint()


def test_finalize_new_configuration_fences_removed_voters() -> None:
    sim, cluster = _cluster_with_learners()
    cluster.begin_joint_consensus("n1", ("n3", "n4", "n5"))

    cluster.node("n3").start_election()
    sim.run()
    assert cluster.node("n3").role is RaftRole.LEADER

    cluster.finalize_membership("n3")
    assert cluster.voting_configuration.old_voters == frozenset({"n3", "n4", "n5"})
    assert cluster.voting_configuration.new_voters is None

    with pytest.raises(NonVoterElectionError, match="non-voter"):
        cluster.node("n1").start_election()

    source_term = cluster.node("n4").current_term
    cluster.node("n1")._persist_term_and_vote(term=source_term + 1, voted_for=None)
    from distlab.raft import RequestVote

    sim.send("n1", "n4", RequestVote(term=source_term + 1, candidate_id="n1"))
    sim.run(max_events=1)
    vote = [
        record
        for record in sim.trace
        if record.kind == "raft-vote"
        and record.details["voter"] == "n4"
        and record.details["candidate"] == "n1"
    ][-1]
    assert vote.details["granted"] is False
    assert vote.details["candidate_eligible"] is False


def test_membership_transition_requires_current_leader() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4"),
        voters=("n1", "n2", "n3"),
    )

    with pytest.raises(MembershipChangeError, match="current leader"):
        cluster.begin_joint_consensus("n1", ("n2", "n3", "n4"))


def test_initial_learner_cannot_start_election() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4"),
        voters=("n1", "n2", "n3"),
    )

    with pytest.raises(NonVoterElectionError, match="non-voter"):
        cluster.node("n4").start_election()
