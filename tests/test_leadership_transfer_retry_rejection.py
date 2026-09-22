import pytest

import distlab.leadership_transfer as leadership_transfer_module
from distlab.leadership_transfer import (
    LeadershipTransfer,
    LeadershipTransferIncomplete,
    LeadershipTransferTargetUnavailable,
)
from distlab.leadership_transfer_retry import retry_timeout_now as real_retry_timeout_now
from distlab.leadership_transfer_transport import TimeoutNow
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def _retry_simulator() -> Simulator:
    return Simulator(
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


def test_transfer_reports_timeout_now_retry_rejection(monkeypatch) -> None:
    sim = _retry_simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    def reject_retry(transport, attempt_id: int) -> None:
        raise RuntimeError("retry authority rejected")

    monkeypatch.setattr(leadership_transfer_module, "retry_timeout_now", reject_retry)

    with pytest.raises(
        LeadershipTransferIncomplete,
        match="failed to retry TimeoutNow for leadership transferee 'n2'",
    ):
        LeadershipTransfer(source).transfer("n2", max_timeout_now_attempts=2)

    timeout_sends = [
        record
        for record in sim.trace
        if record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
    ]
    assert len(timeout_sends) == 1
    assert not any(record.kind == "raft-timeout-now-retry" for record in sim.trace)
    failures = [record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"]
    assert failures[-1].details["stage"] == "retry"
    assert failures[-1].details["reason"] == "TimeoutNow retry rejected: retry authority rejected"
    harness.checkpoint()


def test_transfer_classifies_target_crash_at_timeout_now_retry_dispatch(monkeypatch) -> None:
    sim = _retry_simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    def crash_then_retry(transport, attempt_id: int) -> None:
        sim.crash("n2")
        real_retry_timeout_now(transport, attempt_id)

    monkeypatch.setattr(leadership_transfer_module, "retry_timeout_now", crash_then_retry)

    with pytest.raises(
        LeadershipTransferTargetUnavailable,
        match="crashed before TimeoutNow retry dispatch",
    ):
        LeadershipTransfer(source).transfer("n2", max_timeout_now_attempts=2)

    retry_rejections = [
        record for record in sim.trace if record.kind == "raft-timeout-now-retry-rejected"
    ]
    assert retry_rejections[-1].details["reason"] == "transferee-crashed"
    failures = [record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"]
    assert failures[-1].details["stage"] == "retry"
    assert failures[-1].details["reason"] == "transferee crashed before TimeoutNow retry dispatch"
    assert not any(record.kind == "raft-timeout-now-retry" for record in sim.trace)

    sim.restart("n2")
    harness.checkpoint()
