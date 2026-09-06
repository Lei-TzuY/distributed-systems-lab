import pytest

from distlab.leadership_transfer import InvalidLeadershipTransferTarget, LeadershipTransfer
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import LogEntry, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _cluster_with_learner() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    full_log = (
        LogEntry(term=2, command="set x=1"),
        LogEntry(term=2, command="set y=2"),
    )
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = full_log
    sim.persistent_state["n2"]["log"] = full_log[:1]
    sim.persistent_state["n3"]["log"] = full_log

    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4"),
        voters=("n1", "n2", "n3"),
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 3
    return sim, cluster


def test_transfer_rejects_learner_before_catch_up_or_term_mutation() -> None:
    sim, cluster = _cluster_with_learner()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    source_term = cluster.node("n1").current_term
    learner_log = cluster.node("n4").log
    trace_start = len(sim.trace)

    with pytest.raises(InvalidLeadershipTransferTarget, match="active voter"):
        LeadershipTransfer(cluster.node("n1")).transfer("n4")

    assert cluster.node("n1").current_term == source_term
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n4").log == learner_log
    assert not any(
        record.kind.startswith("raft-leadership-transfer") for record in sim.trace[trace_start:]
    )
    harness.checkpoint()


def test_transfer_to_active_voter_preserves_membership_safety() -> None:
    _, cluster = _cluster_with_learner()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    result = LeadershipTransfer(cluster.node("n1")).transfer(
        "n2",
        max_replication_attempts=4,
    )

    assert result.previous_leader_id == "n1"
    assert result.new_leader_id == "n2"
    assert result.previous_term == 3
    assert result.new_term == 4
    assert cluster.node("n2").role is RaftRole.LEADER
    assert cluster.is_voter("n2")
    assert not cluster.is_voter("n4")
    assert cluster.leaders_by_term[3] == "n1"
    assert cluster.leaders_by_term[4] == "n2"
    harness.checkpoint()
