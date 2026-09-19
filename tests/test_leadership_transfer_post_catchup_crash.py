import pytest

from distlab.leadership_transfer import LeadershipTransfer, LeadershipTransferTargetUnavailable
from distlab.leadership_transfer_transport import TimeoutNow
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def test_transfer_rejects_transferee_crash_after_catchup_without_timeout_authority(
    monkeypatch,
) -> None:
    sim = Simulator()
    full_log = (
        LogEntry(term=3, command="set x=1"),
        LogEntry(term=3, command="set y=2"),
    )
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = full_log
    sim.persistent_state["n2"]["log"] = full_log[:1]
    sim.persistent_state["n3"]["log"] = full_log
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    original_recover_peer = LeaderReplicator.recover_peer

    def recover_then_crash(
        replicator: LeaderReplicator,
        peer: str,
        *,
        max_attempts: int,
    ) -> bool:
        recovered = original_recover_peer(replicator, peer, max_attempts=max_attempts)
        assert recovered
        assert cluster.node(peer).log == source.log
        sim.crash(peer)
        return True

    monkeypatch.setattr(LeaderReplicator, "recover_peer", recover_then_crash)

    with pytest.raises(LeadershipTransferTargetUnavailable, match="crashed during catch-up"):
        LeadershipTransfer(source).transfer("n2", max_replication_attempts=4)

    assert source.role is RaftRole.LEADER
    assert source.current_term == 3
    assert not sim.is_alive("n2")
    assert not any(record.kind == "raft-leadership-transfer-ready" for record in sim.trace)
    assert not any(record.kind == "raft-timeout-now-request" for record in sim.trace)
    assert not any(
        record.kind == "send" and isinstance(record.details["payload"], TimeoutNow)
        for record in sim.trace
    )
    failures = [record for record in sim.trace if record.kind == "raft-leadership-transfer-failed"]
    assert failures[-1].details["stage"] == "catch-up"
    assert failures[-1].details["reason"] == "transferee crashed during catch-up"
    harness.checkpoint()
