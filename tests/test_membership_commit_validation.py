import pytest

from distlab.membership import MembershipChangeError, ReconfigurableRaftCluster
from distlab.membership_log import (
    JointConsensusCommand,
    ReplicatedMembershipTransition,
    StableConsensusCommand,
)
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import LogEntry, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator

NODE_IDS = ("n1", "n2", "n3", "n4", "n5")
BOOTSTRAP_VOTERS = ("n1", "n2", "n3")


def _cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, NODE_IDS, voters=BOOTSTRAP_VOTERS)
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_commit_rejects_conflicting_stable_membership_before_watermark_mutation() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)

    transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)
    joint_commit_index = leader.commit_index
    harness.checkpoint()

    invalid = StableConsensusCommand(("n1", "n2", "n3"))
    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=invalid),
    )
    invalid_index = leader.log_view.last_index
    replicator = MembershipAwareLeaderReplicator(leader)

    with pytest.raises(MembershipChangeError, match="does not match active joint"):
        replicator.advance_commit_index()

    assert invalid_index == joint_commit_index + 1
    assert leader.commit_index == joint_commit_index
    assert replicator.commit_index == joint_commit_index
    assert max(
        int(sim.persistent_state[node_id].get("membership_commit_index", 0))
        for node_id in NODE_IDS
    ) == joint_commit_index
    assert cluster.voting_configuration.is_joint
    harness.checkpoint()


def test_commit_rejects_multiple_uncommitted_membership_commands() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)

    first_index = transition.propose_joint(("n1", "n4", "n5"))
    second = JointConsensusCommand(("n2", "n4", "n5"))
    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=second),
    )
    second_index = leader.log_view.last_index
    replicator = MembershipAwareLeaderReplicator(leader)

    with pytest.raises(MembershipChangeError, match="multiple uncommitted membership commands"):
        replicator.advance_commit_index()

    assert second_index == first_index + 1
    assert leader.commit_index < first_index
    assert replicator.commit_index == leader.commit_index
    assert max(
        int(sim.persistent_state[node_id].get("membership_commit_index", 0))
        for node_id in NODE_IDS
    ) < first_index
    assert cluster.voting_configuration.new_voters is None
    harness.checkpoint()
