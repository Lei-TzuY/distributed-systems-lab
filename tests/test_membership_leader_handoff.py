import pytest

from distlab.membership import MembershipChangeError, ReconfigurableRaftCluster
from distlab.leadership_transfer import LeadershipTransfer
from distlab.membership_log import ReplicatedMembershipTransition
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


NODE_IDS = ("n1", "n2", "n3", "n4", "n5")


def _cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        NODE_IDS,
        voters=("n1", "n2", "n3"),
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_joint_membership_becomes_live_at_objective_commit_boundary() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)
    index = transition.propose_joint(("n1", "n4", "n5"))
    replicator = MembershipAwareLeaderReplicator(leader)

    assert replicator.replicate("n2", max_attempts=2)
    assert leader.commit_index < index
    assert cluster.voting_configuration.new_voters is None

    assert replicator.replicate("n4", max_attempts=2)

    assert leader.commit_index == index
    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n2", "n3"})
    assert cluster.voting_configuration.new_voters == frozenset({"n1", "n4", "n5"})
    assert transition.pending_index == index
    committed = [
        record for record in sim.trace if record.kind == "raft-membership-committed"
    ]
    joint = [record for record in sim.trace if record.kind == "raft-membership-joint"]
    assert len(committed) == len(joint) == 1

    assert transition.activate_if_committed(replicator)
    assert transition.pending_index is None
    assert len([record for record in sim.trace if record.kind == "raft-membership-joint"]) == 1
    RaftSafetyHarness(cluster).checkpoint()


def test_stable_membership_becomes_live_at_objective_commit_boundary() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)
    transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)

    index = transition.propose_finalize()
    replicator = MembershipAwareLeaderReplicator(leader)
    assert replicator.replicate("n2", max_attempts=2)
    assert cluster.voting_configuration.is_joint

    assert replicator.replicate("n4", max_attempts=3)

    assert leader.commit_index == index
    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n4", "n5"})
    assert cluster.voting_configuration.new_voters is None
    assert transition.pending_index == index
    stable = [record for record in sim.trace if record.kind == "raft-membership-stable"]
    finalized = [
        record for record in sim.trace if record.kind == "raft-membership-finalized"
    ]
    assert len(stable) == len(finalized) == 1

    assert transition.finalize_if_committed(replicator)
    assert transition.pending_index is None
    assert len([record for record in sim.trace if record.kind == "raft-membership-stable"]) == 1
    RaftSafetyHarness(cluster).checkpoint()


def test_new_leader_recovers_uncommitted_joint_and_commits_through_new_term_barrier() -> None:
    sim, cluster = _cluster()
    first_leader = cluster.node("n1")
    first = ReplicatedMembershipTransition(first_leader)
    pending_index = first.propose_joint(("n1", "n4", "n5"))
    replicator = MembershipAwareLeaderReplicator(first_leader)

    assert replicator.replicate("n2", max_attempts=2)
    assert first_leader.commit_index < pending_index
    assert cluster.voting_configuration.new_voters is None
    assert max(
        int(sim.persistent_state[node_id].get("membership_commit_index", 0))
        for node_id in NODE_IDS
    ) == 0

    result = LeadershipTransfer(first_leader).transfer("n2")
    assert result.new_leader_id == "n2"
    second_leader = cluster.node("n2")
    assert second_leader.role is RaftRole.LEADER
    assert second_leader.current_term == 2

    recovered = ReplicatedMembershipTransition(second_leader)

    assert recovered.pending_index == pending_index
    recovered_events = [
        record
        for record in sim.trace
        if record.kind == "raft-membership-pending-recovered"
    ]
    assert recovered_events[-1].details["leader"] == "n2"
    assert recovered_events[-1].details["index"] == pending_index
    assert recovered_events[-1].details["command_term"] == 1

    with pytest.raises(MembershipChangeError, match="already pending"):
        recovered.propose_joint(("n1", "n3", "n5"))
    with pytest.raises(MembershipChangeError, match="generation is no longer current"):
        first.replicate_and_activate(max_attempts_per_peer=2)

    assert recovered.replicate_and_activate(max_attempts_per_peer=3)

    assert recovered.pending_index is None
    assert cluster.voting_configuration.old_voters == frozenset({"n1", "n2", "n3"})
    assert cluster.voting_configuration.new_voters == frozenset({"n1", "n4", "n5"})
    barriers = [
        record
        for record in sim.trace
        if record.kind == "raft-membership-handoff-barrier"
    ]
    assert len(barriers) == 1
    assert barriers[0].details["pending_index"] == pending_index
    barrier_index = barriers[0].details["barrier_index"]
    assert barrier_index == pending_index + 1
    assert second_leader.commit_index >= barrier_index
    assert max(
        int(sim.persistent_state[node_id].get("membership_commit_index", 0))
        for node_id in NODE_IDS
    ) == pending_index
    RaftSafetyHarness(cluster).checkpoint()


def test_new_leader_recovers_uncommitted_finalize_but_outgoing_only_leader_cannot_commit_it(
) -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)
    transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)
    finalize_index = transition.propose_finalize()

    replicator = MembershipAwareLeaderReplicator(leader)
    assert replicator.replicate("n2", max_attempts=2)
    assert leader.commit_index < finalize_index

    # n2 is still an active joint voter but is not part of the target stable set.
    cluster.node("n2").start_election()
    sim.run()
    assert cluster.node("n2").role is RaftRole.LEADER
    second = ReplicatedMembershipTransition(cluster.node("n2"))

    assert second.pending_index == finalize_index
    with pytest.raises(
        MembershipChangeError,
        match="current leader must belong to the new voter configuration",
    ):
        second.replicate_and_finalize(max_attempts_per_peer=3)
    assert cluster.voting_configuration.is_joint
    RaftSafetyHarness(cluster).checkpoint()
