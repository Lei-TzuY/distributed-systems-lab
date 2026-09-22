import pytest

from distlab.leader_liveness import (
    LeaderAuthorityLost,
    LeaderQuorumMonitor,
    LeaderQuorumUnavailable,
)
from distlab.membership import ReconfigurableRaftCluster
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _elected_cluster() -> tuple[Simulator, RaftCluster]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_check_quorum_keeps_leader_with_same_term_majority() -> None:
    sim, cluster = _elected_cluster()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(LeaderReplicator(leader))

    evidence = monitor.check()

    assert evidence.leader == "n1"
    assert evidence.term == 1
    assert evidence.acknowledged_voters == ("n1", "n2")
    assert evidence.quorum_mode == "stable"
    assert evidence.old_majority == 2
    assert evidence.new_majority is None
    assert leader.role is RaftRole.LEADER
    assert any(record.kind == "raft-check-quorum" for record in sim.trace)
    harness.checkpoint()


def test_check_quorum_counts_same_term_append_rejection_as_activity() -> None:
    sim, cluster = _elected_cluster()
    leader = cluster.node("n1")
    leader._persist_log(
        (LogEntry(term=leader.current_term, command="leader-only"),)
    )
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(LeaderReplicator(leader))

    evidence = monitor.check()

    assert evidence.acknowledged_voters == ("n1", "n2")
    acknowledgements = [
        record
        for record in sim.trace
        if record.kind == "raft-quorum-probe-ack"
        and record.details["follower"] == "n2"
    ]
    assert acknowledgements[-1].details["append_success"] is False
    assert leader.role is RaftRole.LEADER
    harness.checkpoint()


def test_check_quorum_retires_isolated_leader_without_advancing_term() -> None:
    sim, cluster = _elected_cluster()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(LeaderReplicator(leader))
    original_term = leader.current_term
    original_vote = leader.voted_for
    sim.partition(("n1",), ("n2", "n3"))

    with pytest.raises(LeaderQuorumUnavailable, match="could not confirm"):
        monitor.check()

    assert leader.current_term == original_term == 1
    assert leader.voted_for == original_vote == "n1"
    assert leader.role is RaftRole.FOLLOWER
    failures = [record for record in sim.trace if record.kind == "raft-check-quorum-failed"]
    assert failures[-1].details["acknowledged_voters"] == ("n1",)
    step_down = [record for record in sim.trace if record.kind == "raft-leader-step-down"]
    assert step_down[-1].details["reason"] == "check-quorum-failed"
    harness.checkpoint()


def test_higher_term_quorum_response_uses_existing_term_fencing() -> None:
    sim, cluster = _elected_cluster()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(LeaderReplicator(leader))
    sim.persistent_state["n2"]["current_term"] = 2
    sim.persistent_state["n2"]["voted_for"] = None

    with pytest.raises(LeaderAuthorityLost, match="leader role"):
        monitor.check()

    assert leader.current_term == 2
    assert leader.role is RaftRole.FOLLOWER
    assert not any(record.kind == "raft-check-quorum-failed" for record in sim.trace)
    lost = [
        record
        for record in sim.trace
        if record.kind == "raft-check-quorum-authority-lost"
    ]
    assert lost[-1].details["current_term"] == 2
    harness.checkpoint()


def test_explicit_leader_step_down_rearms_election_timeout() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",), election_timeouts={"n1": 5})
    leader = cluster.node("n1")
    leader.start_election()
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 1

    assert leader.step_down(reason="test-check-quorum") is True

    assert leader.role is RaftRole.FOLLOWER
    assert leader.current_term == 1
    resets = [
        record
        for record in sim.trace
        if record.kind == "raft-election-timeout-reset"
        and record.details["node"] == "n1"
    ]
    assert resets[-1].details["reason"] == "test-check-quorum"
    assert resets[-1].details["deadline"] == 5


def _joint_cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    cluster.begin_joint_consensus("n1", ("n3", "n4", "n5"))
    return sim, cluster


def test_joint_check_quorum_requires_both_old_and_new_majorities() -> None:
    sim, cluster = _joint_cluster()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(MembershipAwareLeaderReplicator(leader))
    sim.partition(("n1",), ("n4", "n5"))

    with pytest.raises(LeaderQuorumUnavailable):
        monitor.check()

    assert leader.role is RaftRole.FOLLOWER
    failure = [record for record in sim.trace if record.kind == "raft-check-quorum-failed"][-1]
    assert failure.details["quorum_mode"] == "joint"
    assert failure.details["old_majority"] == 2
    assert failure.details["new_majority"] == 2
    assert set(failure.details["acknowledged_voters"]) == {"n1", "n2", "n3"}
    harness.checkpoint()


def test_joint_check_quorum_succeeds_with_dual_majority() -> None:
    sim, cluster = _joint_cluster()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    monitor = LeaderQuorumMonitor(MembershipAwareLeaderReplicator(leader))
    sim.partition(("n1",), ("n5",))

    evidence = monitor.check()

    assert leader.role is RaftRole.LEADER
    assert evidence.quorum_mode == "joint"
    assert evidence.old_majority == 2
    assert evidence.new_majority == 2
    assert {"n1", "n2"} <= set(evidence.acknowledged_voters)
    assert len({"n3", "n4"} & set(evidence.acknowledged_voters)) >= 2
    harness.checkpoint()


def test_stable_check_quorum_does_not_count_learners() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    monitor = LeaderQuorumMonitor(MembershipAwareLeaderReplicator(leader))
    sim.partition(("n1",), ("n2", "n3"))

    with pytest.raises(LeaderQuorumUnavailable):
        monitor.check()

    probed = {
        record.details["follower"]
        for record in sim.trace
        if record.kind == "raft-quorum-probe"
    }
    assert probed == {"n2", "n3"}
    assert leader.role is RaftRole.FOLLOWER
