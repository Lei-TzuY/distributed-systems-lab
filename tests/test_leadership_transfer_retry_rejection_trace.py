import pytest

from distlab.leadership_transfer_retry import retry_timeout_now
from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def _dropped_transfer() -> tuple[Simulator, RaftCluster, LeadershipTransferTransport, int]:
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
    transport = LeadershipTransferTransport.for_cluster(cluster)
    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    sim.run()
    return sim, cluster, transport, attempt_id


def _retry_rejections(sim: Simulator) -> list:
    return [
        record
        for record in sim.trace
        if record.kind == "raft-timeout-now-retry-rejected"
    ]


def test_retry_traces_crashed_transferee_preflight_rejection() -> None:
    sim, cluster, transport, attempt_id = _dropped_transfer()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    sim.crash("n2")

    send_count = sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    )
    with pytest.raises(RuntimeError, match="live transferee"):
        retry_timeout_now(transport, attempt_id)

    rejection = _retry_rejections(sim)[-1]
    assert rejection.details == {
        "leader": "n1",
        "transferee": "n2",
        "term": 1,
        "attempt_id": attempt_id,
        "reason": "transferee-crashed",
    }
    assert attempt_id in transport._active_attempts
    assert sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    ) == send_count

    sim.restart("n2")
    harness.checkpoint()


def test_retry_traces_divergent_log_preflight_rejection() -> None:
    sim, cluster, transport, attempt_id = _dropped_transfer()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    leader = cluster.node("n1")
    target = cluster.node("n2")
    shared = LogEntry(term=leader.current_term, command="set shared=1")
    sim.persistent_state[leader.node_id]["log"] = (*leader.log, shared)
    sim.persistent_state[target.node_id]["log"] = (*target.log, shared)
    original_target_log = target.log
    sim.persistent_state[target.node_id]["log"] = (
        *original_target_log[:-1],
        LogEntry(term=shared.term, command="set divergent=1"),
    )

    with pytest.raises(RuntimeError, match="matching retained logs"):
        retry_timeout_now(transport, attempt_id)

    rejection = _retry_rejections(sim)[-1]
    assert rejection.details["attempt_id"] == attempt_id
    assert rejection.details["reason"] == "transferee-log-mismatch"
    assert attempt_id in transport._active_attempts

    sim.persistent_state[target.node_id]["log"] = original_target_log
    harness.checkpoint()
