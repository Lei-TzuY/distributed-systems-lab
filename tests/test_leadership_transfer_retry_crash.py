import pytest

from distlab.leadership_transfer import LeadershipTransfer, LeadershipTransferTargetUnavailable
from distlab.leadership_transfer_transport import TimeoutNow
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_transfer_reports_transferee_crash_after_retry_catchup(monkeypatch) -> None:
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
    original_recover_peer = LeaderReplicator.recover_peer
    recoveries = 0

    def recover_then_crash_on_retry(
        replicator: LeaderReplicator,
        peer: str,
        *,
        max_attempts: int,
    ) -> bool:
        nonlocal recoveries
        recovered = original_recover_peer(replicator, peer, max_attempts=max_attempts)
        recoveries += 1
        if recoveries == 2:
            assert recovered
            sim.crash(peer)
        return recovered

    monkeypatch.setattr(LeaderReplicator, "recover_peer", recover_then_crash_on_retry)

    with pytest.raises(LeadershipTransferTargetUnavailable, match="crashed during retry catch-up"):
        LeadershipTransfer(source).transfer("n2", max_timeout_now_attempts=2)

    assert recoveries == 2
    assert source.role is RaftRole.LEADER
    assert not sim.is_alive("n2")
    timeout_sends = [
        record
        for record in sim.trace
        if record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
    ]
    assert len(timeout_sends) == 1
    assert not any(record.kind == "raft-timeout-now-retry" for record in sim.trace)
    failures = [record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"]
    assert failures[-1].details["stage"] == "retry"
    assert failures[-1].details["reason"] == "transferee crashed during retry catch-up"
    harness.checkpoint()
