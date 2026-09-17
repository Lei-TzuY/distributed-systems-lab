import pytest

from distlab.leadership_transfer_retry import retry_timeout_now
from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.raft import RaftCluster, RaftRole
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
