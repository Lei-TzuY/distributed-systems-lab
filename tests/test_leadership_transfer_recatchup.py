from distlab.leadership_transfer import LeadershipTransfer
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_transfer_recatches_up_before_retrying_same_timeout_now_attempt(monkeypatch) -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    transfer = LeadershipTransfer(leader)
    original_send = transfer.transfer_transport.send_timeout_now
    appended = LogEntry(term=leader.current_term, command="set after-ready=1")

    def send_then_advance_leader(leader_id: str, transferee_id: str, *, term: int) -> int:
        attempt_id = original_send(leader_id, transferee_id, term=term)
        sim.persistent_state[leader_id]["log"] = (*leader.log, appended)
        return attempt_id

    monkeypatch.setattr(transfer.transfer_transport, "send_timeout_now", send_then_advance_leader)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    result = transfer.transfer(
        "n2",
        max_replication_attempts=4,
        max_timeout_now_attempts=2,
    )

    assert result.new_leader_id == "n2"
    assert cluster.node("n2").role is RaftRole.LEADER
    assert cluster.node("n2").log == leader.log
    rejections = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert len(rejections) == 1
    assert rejections[0].details["reason"] == "transferee-not-caught-up"
    assert rejections[0].details["retryable"] is True
    requests = [record for record in sim.trace if record.kind == "raft-timeout-now-request"]
    retries = [record for record in sim.trace if record.kind == "raft-timeout-now-retry"]
    delivered = [record for record in sim.trace if record.kind == "raft-timeout-now-delivered"]
    assert len(requests) == len(retries) == len(delivered) == 1
    assert requests[0].details["attempt_id"] == retries[0].details["attempt_id"]
    assert retries[0].details["attempt_id"] == delivered[0].details["attempt_id"]
    harness.checkpoint()
