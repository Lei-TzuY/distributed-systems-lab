import pytest

from distlab.leadership_transfer import LeadershipTransfer
from distlab.leadership_transfer_retry import retry_timeout_now
from distlab.leadership_transfer_transport import LeadershipTransferTransport
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_successful_transfer_retires_timeout_now_retry_authority() -> None:
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
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    transport = LeadershipTransferTransport.for_cluster(cluster)
    result = LeadershipTransfer(cluster.node("n1")).transfer(
        "n2",
        max_replication_attempts=4,
    )

    assert result.new_leader_id == "n2"
    completions = [
        record for record in sim.trace if record.kind == "raft-timeout-now-completed"
    ]
    assert len(completions) == 1
    attempt_id = completions[0].details["attempt_id"]
    assert completions[0].details["leader"] == "n1"
    assert completions[0].details["transferee"] == "n2"
    assert completions[0].details["term"] == result.previous_term

    with pytest.raises(ValueError, match="unknown active leadership transfer attempt"):
        retry_timeout_now(transport, attempt_id)

    assert not transport.cancel_timeout_now(attempt_id)
    harness.checkpoint()
