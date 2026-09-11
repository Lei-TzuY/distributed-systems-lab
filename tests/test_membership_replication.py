import pytest

from distlab.commit_recovery import append_current_term_barrier
from distlab.membership import ReconfigurableRaftCluster
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import ReplicationError
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


def test_stable_commit_quorum_ignores_preprovisioned_learners() -> None:
    _, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    leader = cluster.node("n1")
    index = append_current_term_barrier(leader)
    replicator = MembershipAwareLeaderReplicator(leader)

    assert replicator.replicate("n2", max_attempts=2)

    assert index == 1
    assert replicator.commit_index == index
    assert leader.commit_index == index
    harness.checkpoint()


def test_joint_commit_requires_old_and_new_majorities() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    cluster.begin_joint_consensus("n1", ("n3", "n4", "n5"))
    leader = cluster.node("n1")
    index = append_current_term_barrier(leader)
    replicator = MembershipAwareLeaderReplicator(leader)

    assert replicator.replicate("n2", max_attempts=2)
    assert replicator.commit_index == 0

    assert replicator.replicate("n3", max_attempts=2)
    assert replicator.commit_index == 0

    assert replicator.replicate("n4", max_attempts=2)
    assert replicator.commit_index == index
    assert leader.commit_index == index

    commit = [record for record in sim.trace if record.kind == "raft-commit-advance"][-1]
    assert commit.details["quorum_mode"] == "joint"
    assert set(commit.details["acknowledged_voters"]) == {"n1", "n2", "n3", "n4"}
    harness.checkpoint()


def test_removed_leader_cannot_continue_membership_aware_replication() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    leader = cluster.node("n1")
    append_current_term_barrier(leader)
    replicator = MembershipAwareLeaderReplicator(leader)

    cluster.begin_joint_consensus("n1", ("n2", "n3", "n4"))
    cluster.finalize_membership("n1")

    assert leader.role is RaftRole.LEADER
    assert not cluster.is_voter("n1")
    assert leader.commit_index == 0
    trace_before = tuple(sim.trace)

    with pytest.raises(ReplicationError, match="current voter authority"):
        replicator.replicate("n2", max_attempts=1)
    with pytest.raises(ReplicationError, match="current voter authority"):
        replicator.advance_commit_index()

    assert leader.commit_index == 0
    assert tuple(sim.trace) == trace_before
    harness.checkpoint()
