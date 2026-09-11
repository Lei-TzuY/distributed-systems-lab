import pytest

from distlab.leadership_transfer_transport import LeadershipTransferTransport
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_self_targeted_timeout_now_does_not_leak_active_attempt() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1

    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    with pytest.raises(ValueError, match="endpoints must be distinct"):
        transport.send_timeout_now("n1", "n1", term=1)

    assert not any(record.kind == "raft-timeout-now-request" for record in sim.trace)
    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    assert attempt_id == 1
    assert transport.cancel_timeout_now(attempt_id)
    sim.run()

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    harness.checkpoint()
