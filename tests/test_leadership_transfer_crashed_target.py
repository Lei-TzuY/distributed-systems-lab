import pytest

from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_initial_transfer_rejects_crashed_target_without_minting_authority() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER

    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    sim.crash("n2")

    request_count = sum(record.kind == "raft-timeout-now-request" for record in sim.trace)
    send_count = sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    )

    with pytest.raises(RuntimeError, match="live transferee"):
        transport.send_timeout_now("n1", "n2", term=1)

    assert transport._next_attempt_id == 0
    assert transport._active_attempts == {}
    assert transport._attempt_configurations == {}
    assert sum(record.kind == "raft-timeout-now-request" for record in sim.trace) == request_count
    assert sum(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    ) == send_count

    sim.restart("n2")
    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    assert attempt_id == 1
    sim.run()

    assert cluster.node("n2").role is RaftRole.LEADER
    harness.checkpoint()
