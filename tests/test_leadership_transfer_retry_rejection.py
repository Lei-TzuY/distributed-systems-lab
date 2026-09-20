import pytest

import distlab.leadership_transfer as leadership_transfer_module
from distlab.leadership_transfer import LeadershipTransfer, LeadershipTransferIncomplete
from distlab.leadership_transfer_transport import TimeoutNow
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_transfer_reports_timeout_now_retry_rejection(monkeypatch) -> None:
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
