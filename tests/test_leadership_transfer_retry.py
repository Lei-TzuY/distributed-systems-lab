import pytest

from distlab.leadership_transfer_retry import retry_timeout_now
from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def _leader_cluster() -> tuple[Simulator, RaftCluster]:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DROP,
                    src="n1",
                    dst="n2",
                    ordinal=2,
                    payload_type="TimeoutNow",
                ),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_retry_reuses_dropped_timeout_now_attempt_and_elects_transferee() -> None:
    sim, cluster = _leader_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    retry_timeout_now(transport, attempt_id)
    sim.run()

    assert cluster.node("n2").role is RaftRole.LEADER
    retries = [record for record in sim.trace if record.kind == "raft-timeout-now-retry"]
    assert retries[-1].details["attempt_id"] == attempt_id
    timeout_sends = [
        record
        for record in sim.trace
        if record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
    ]
    assert [record.details["ordinal"] for record in timeout_sends] == [2, 3]
    assert all(record.details["payload"].attempt_id == attempt_id for record in timeout_sends)
    harness.checkpoint()


def test_retry_rejects_cancelled_attempt() -> None:
    _, cluster = _leader_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    assert transport.cancel_timeout_now(attempt_id)

    with pytest.raises(ValueError, match="unknown active leadership transfer attempt"):
        retry_timeout_now(transport, attempt_id)


def test_retry_waits_for_crashed_transferee_without_consuming_attempt() -> None:
    sim, cluster = _leader_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER

    sim.crash("n2")
    retry_count = sum(record.kind == "raft-timeout-now-retry" for record in sim.trace)
    send_count = sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    )

    with pytest.raises(RuntimeError, match="live transferee"):
        retry_timeout_now(transport, attempt_id)

    assert sum(record.kind == "raft-timeout-now-retry" for record in sim.trace) == retry_count
    assert sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    ) == send_count

    sim.restart("n2")
    retry_timeout_now(transport, attempt_id)
    sim.run()

    assert cluster.node("n2").role is RaftRole.LEADER
    retries = [record for record in sim.trace if record.kind == "raft-timeout-now-retry"]
    assert retries[-1].details["attempt_id"] == attempt_id
    harness.checkpoint()


def test_retry_rejects_transferee_that_fell_behind_without_consuming_attempt() -> None:
    sim, cluster = _leader_cluster()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    leader = cluster.node("n1")
    target = cluster.node("n2")
    attempt_id = transport.send_timeout_now("n1", "n2", term=leader.current_term)
    sim.run()
    assert leader.role is RaftRole.LEADER

    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command="set after-drop=1"),
    )
    assert target.last_log_index < leader.last_log_index
    retry_count = sum(record.kind == "raft-timeout-now-retry" for record in sim.trace)
    send_count = sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    )

    with pytest.raises(RuntimeError, match="caught-up transferee"):
        retry_timeout_now(transport, attempt_id)

    assert sum(record.kind == "raft-timeout-now-retry" for record in sim.trace) == retry_count
    assert sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    ) == send_count
    assert attempt_id in transport._active_attempts
    assert target.role is RaftRole.FOLLOWER
    harness.checkpoint()
