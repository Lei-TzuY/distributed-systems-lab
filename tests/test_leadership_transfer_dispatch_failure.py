import pytest

from distlab.leadership_transfer import (
    LeadershipTransfer,
    LeadershipTransferTargetUnavailable,
)
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_transfer_records_target_crash_at_timeout_now_dispatch(monkeypatch) -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    transfer = LeadershipTransfer(leader)
    original_send = transfer.transfer_transport.send_timeout_now

    def crash_then_send(leader_id: str, transferee_id: str, *, term: int) -> int:
        sim.crash(transferee_id)
        return original_send(leader_id, transferee_id, term=term)

    monkeypatch.setattr(transfer.transfer_transport, "send_timeout_now", crash_then_send)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    with pytest.raises(
        LeadershipTransferTargetUnavailable,
        match="crashed before TimeoutNow dispatch",
    ):
        transfer.transfer("n2")

    failures = [
        record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"
    ]
    assert len(failures) == 1
    assert failures[0].details["stage"] == "dispatch"
    assert failures[0].details["reason"] == "transferee crashed before TimeoutNow dispatch"
    assert not [record for record in sim.trace if record.kind == "raft-timeout-now-request"]

    sim.restart("n2")
    harness.checkpoint()
