import pytest

from distlab.membership import (
    MembershipChangeError,
    NonVoterElectionError,
    ReconfigurableRaftCluster,
)
from distlab.membership_log import (
    JointConsensusCommand,
    ReplicatedMembershipTransition,
    StableConsensusCommand,
)
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
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


def test_joint_membership_activates_only_after_log_entry_commits() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)

    index = transition.propose_joint(("n3", "n4", "n5"))

    assert index == leader.log_view.last_index
    assert leader.commit_index < index
    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n2", "n3"})
    assert cluster.voting_configuration.new_voters is None
    assert leader.log_view.entry_at(index).command == JointConsensusCommand(("n3", "n4", "n5"))
    harness.checkpoint()

    assert transition.replicate_and_activate(max_attempts_per_peer=2)

    assert leader.commit_index == index
    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n2", "n3"})
    assert cluster.voting_configuration.new_voters == frozenset({"n3", "n4", "n5"})
    assert transition.pending_index is None
    assert any(
        record.kind == "raft-membership-committed"
        and record.details["index"] == index
        for record in sim.trace
    )
    harness.checkpoint()


def test_stable_finalization_waits_for_joint_quorum_commit() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)

    transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)
    joint_commit_index = leader.commit_index
    harness.checkpoint()

    final_index = transition.propose_finalize()

    assert final_index == joint_commit_index + 1
    assert leader.commit_index == joint_commit_index
    assert leader.log_view.entry_at(final_index).command == StableConsensusCommand(
        ("n1", "n4", "n5")
    )
    assert cluster.voting_configuration.is_joint
    assert cluster.is_voter("n2")

    replicator = MembershipAwareLeaderReplicator(leader)
    replicator.replicate("n2", max_attempts=2)

    assert replicator.commit_index == joint_commit_index
    assert not transition.finalize_if_committed(replicator)
    assert cluster.voting_configuration.is_joint
    assert cluster.is_voter("n2")
    assert not any(record.kind == "raft-membership-finalized" for record in sim.trace)
    harness.checkpoint()

    replicator.replicate("n4", max_attempts=3)
    assert replicator.commit_index == final_index
    assert transition.finalize_if_committed(replicator)

    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n4", "n5"})
    assert cluster.voting_configuration.new_voters is None
    assert not cluster.is_voter("n2")
    assert transition.pending_index is None
    assert any(
        record.kind == "raft-membership-finalized"
        and record.details["index"] == final_index
        and record.details["voters"] == ("n1", "n4", "n5")
        for record in sim.trace
    )
    with pytest.raises(NonVoterElectionError, match="non-voter"):
        cluster.node("n2").start_election()
    harness.checkpoint()


def test_finalize_requires_active_joint_configuration() -> None:
    _, cluster = _cluster()
    transition = ReplicatedMembershipTransition(cluster.node("n1"))

    with pytest.raises(MembershipChangeError, match="no joint consensus"):
        transition.propose_finalize()


def test_pending_membership_proposal_rejects_conflicting_second_proposal() -> None:
    _, cluster = _cluster()
    transition = ReplicatedMembershipTransition(cluster.node("n1"))
    index = transition.propose_joint(("n2", "n3", "n4"))

    with pytest.raises(MembershipChangeError, match="already pending"):
        transition.propose_joint(("n3", "n4", "n5"))

    assert transition.pending_index == index
    assert cluster.voting_configuration.new_voters is None
