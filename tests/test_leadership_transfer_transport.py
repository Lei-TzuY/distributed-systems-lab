import pytest

from distlab.leadership_transfer import LeadershipTransfer
from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def _elected_cluster(*, fault_plan: FaultPlan | None = None) -> tuple[Simulator, RaftCluster]:
    sim = Simulator(fault_plan=fault_plan)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    return sim, cluster


def _elected_reconfigurable_cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    return sim, cluster


def test_leadership_transfer_routes_timeout_now_through_simulator() -> None:
    sim, cluster = _elected_cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    result = LeadershipTransfer(cluster.node("n1")).transfer("n2")

    assert result.previous_leader_id == "n1"
    assert result.new_leader_id == "n2"
    assert result.previous_term == 1
    assert result.new_term == 2
    assert cluster.node("n2").role is RaftRole.LEADER

    requests = [record for record in sim.trace if record.kind == "raft-timeout-now-request"]
    delivered = [record for record in sim.trace if record.kind == "raft-timeout-now-delivered"]
    assert requests[-1].details == {
        "leader": "n1",
        "transferee": "n2",
        "term": 1,
        "attempt_id": 1,
    }
    assert delivered[-1].details == {
        "leader": "n1",
        "transferee": "n2",
        "term": 1,
        "attempt_id": 1,
    }
    assert any(
        record.kind == "send"
        and isinstance(record.details["payload"], TimeoutNow)
        and record.details["src"] == "n1"
        and record.details["dst"] == "n2"
        for record in sim.trace
    )
    harness.checkpoint()


def test_cancelled_timeout_now_attempt_cannot_trigger_later_election() -> None:
    sim, cluster = _elected_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    assert transport.cancel_timeout_now(attempt_id)
    sim.run()

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n2").current_term == 1
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    cancelled = [record for record in sim.trace if record.kind == "raft-timeout-now-cancelled"]
    assert cancelled[-1].details == {
        "leader": "n1",
        "transferee": "n2",
        "term": 1,
        "attempt_id": attempt_id,
    }
    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["attempt_id"] == attempt_id
    assert rejected[-1].details["reason"] == "stale-transfer-attempt"
    harness.checkpoint()


def test_timeout_now_serializes_overlapping_attempts_for_same_leader_term() -> None:
    sim, cluster = _elected_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    first_attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    with pytest.raises(RuntimeError, match="already active"):
        transport.send_timeout_now("n1", "n3", term=1)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n3").role is RaftRole.FOLLOWER
    assert not any(
        record.kind == "raft-election-start" and record.details["node"] == "n3"
        for record in sim.trace
    )

    rejected = [
        record for record in sim.trace if record.kind == "raft-timeout-now-request-rejected"
    ]
    assert rejected[-1].details == {
        "leader": "n1",
        "transferee": "n3",
        "term": 1,
        "active_attempt_id": first_attempt_id,
        "active_transferee": "n2",
        "reason": "transfer-attempt-already-active",
    }

    assert transport.cancel_timeout_now(first_attempt_id)
    second_attempt_id = transport.send_timeout_now("n1", "n3", term=1)
    assert second_attempt_id > first_attempt_id
    assert transport.cancel_timeout_now(second_attempt_id)
    sim.run()
    harness.checkpoint()


def test_timeout_now_rejects_matching_terminal_metadata_with_divergent_retained_log() -> None:
    sim = Simulator()
    common_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=1, command="b"),
    )
    for node_id in ("n1", "n2", "n3"):
        sim.persistent_state[node_id]["log"] = common_log
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    target = cluster.node("n2")
    assert target.last_log_index == cluster.node("n1").last_log_index == 2
    assert target.last_log_term == cluster.node("n1").last_log_term == 1
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="corrupt-a"),
        common_log[1],
    )

    transport = LeadershipTransferTransport.for_cluster(cluster)
    transport.send_timeout_now("n1", "n2", term=1)
    sim.run(max_events=1)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert target.role is RaftRole.FOLLOWER
    assert target.current_term == 1
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)
    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["reason"] == "transferee-log-mismatch"

    sim.persistent_state["n2"]["log"] = common_log
    harness.checkpoint()


def test_delayed_timeout_now_is_rejected_after_source_loses_leadership() -> None:
    fault_plan = FaultPlan(
        (
            FaultRule(
                action=FaultAction.DELAY,
                src="n1",
                dst="n2",
                ordinal=2,
                extra_delay=5,
            ),
            FaultRule(
                action=FaultAction.DROP,
                src="n3",
                dst="n2",
                ordinal=1,
            ),
        )
    )
    sim, cluster = _elected_cluster(fault_plan=fault_plan)
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    transport.send_timeout_now("n1", "n2", term=1)
    cluster.node("n3").start_election()
    sim.run()

    assert cluster.node("n3").role is RaftRole.LEADER
    assert cluster.node("n3").current_term == 2
    assert cluster.node("n1").role is RaftRole.FOLLOWER
    assert cluster.node("n1").current_term == 2
    assert cluster.node("n2").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["reason"] == "source-no-longer-current-leader"
    harness.checkpoint()


def test_timeout_now_rejects_transferee_that_becomes_outgoing_only_before_delivery() -> None:
    sim, cluster = _elected_reconfigurable_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    transport.send_timeout_now("n1", "n2", term=1)
    cluster.begin_joint_consensus("n1", ("n1", "n3"))
    sim.run(max_events=1)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n2").current_term == 1
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["reason"] == "transferee-outgoing-only"
    harness.checkpoint()


def test_timeout_now_rejects_transferee_removed_before_delivery() -> None:
    sim, cluster = _elected_reconfigurable_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    transport.send_timeout_now("n1", "n2", term=1)
    cluster.begin_joint_consensus("n1", ("n1", "n3"))
    cluster.finalize_membership("n1")
    sim.run(max_events=1)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n2").current_term == 1
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["reason"] == "transferee-not-voter"
    harness.checkpoint()
