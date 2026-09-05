import pytest

from distlab.leadership_transfer import (
    InvalidLeadershipTransferTarget,
    LeadershipTransfer,
    LeadershipTransferTargetUnavailable,
)
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _lagging_transferee_cluster() -> tuple[Simulator, RaftCluster]:
    sim = Simulator()
    full_log = (
        LogEntry(term=3, command="set x=1"),
        LogEntry(term=3, command="set y=2"),
    )
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = full_log
    sim.persistent_state["n2"]["log"] = full_log[:1]
    sim.persistent_state["n3"]["log"] = full_log

    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 3
    return sim, cluster


def test_transfer_catches_up_target_and_elects_it_in_next_term() -> None:
    sim, cluster = _lagging_transferee_cluster()
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
    assert result.commit_index == 2
    assert cluster.node("n2").role is RaftRole.LEADER
    assert cluster.node("n2").log == cluster.node("n1").log
    assert cluster.node("n2").commit_index == 2
    assert cluster.node("n1").role is RaftRole.FOLLOWER
    assert cluster.node("n1").current_term == 4

    harness.checkpoint()
    assert cluster.leaders_by_term[3] == "n1"
    assert cluster.leaders_by_term[4] == "n2"

    transfer_kinds = [
        record.kind for record in sim.trace if record.kind.startswith("raft-leadership-transfer")
    ]
    assert transfer_kinds == [
        "raft-leadership-transfer-start",
        "raft-leadership-transfer-ready",
        "raft-leadership-transfer-complete",
    ]


def test_transfer_rejects_current_leader_as_target() -> None:
    _, cluster = _lagging_transferee_cluster()

    with pytest.raises(InvalidLeadershipTransferTarget, match="distinct peer"):
        LeadershipTransfer(cluster.node("n1")).transfer("n1")


def test_transfer_rejects_crashed_target_before_mutating_term() -> None:
    sim, cluster = _lagging_transferee_cluster()
    sim.crash("n2")
    source_term = cluster.node("n1").current_term

    with pytest.raises(LeadershipTransferTargetUnavailable, match="not live"):
        LeadershipTransfer(cluster.node("n1")).transfer("n2")

    assert cluster.node("n1").current_term == source_term
    assert cluster.node("n1").role is RaftRole.LEADER
    assert not any(record.kind == "raft-leadership-transfer-start" for record in sim.trace)
