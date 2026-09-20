import pytest

from distlab.leadership_transfer import LeadershipTransfer, LeadershipTransferIncomplete
from distlab.leadership_transfer_transport import LeadershipTransferTransport
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_transfer_reports_source_authority_loss_while_waiting_for_election(
    monkeypatch,
) -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    transport = LeadershipTransferTransport.for_cluster(cluster)
    original_send_timeout_now = transport.send_timeout_now

    def send_then_retire_source(source_id: str, transferee_id: str, *, term: int) -> int:
        attempt_id = original_send_timeout_now(source_id, transferee_id, term=term)
        cluster.node("n3").start_election()
        sim.run()
        assert source.role is RaftRole.FOLLOWER
        assert cluster.node("n3").role is RaftRole.LEADER
        return attempt_id

    monkeypatch.setattr(transport, "send_timeout_now", send_then_retire_source)

    with pytest.raises(
        LeadershipTransferIncomplete,
        match="source leader lost leadership while waiting for transferee election",
    ):
        LeadershipTransfer(source).transfer("n2")

    assert source.role is RaftRole.FOLLOWER
    assert cluster.node("n3").role is RaftRole.LEADER
    assert not any(record.kind == "raft-leadership-transfer-complete" for record in sim.trace)
    failures = [record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"]
    assert failures[-1].details["stage"] == "election"
    assert (
        failures[-1].details["reason"]
        == "source leader lost leadership during transferee election"
    )
    harness.checkpoint()
