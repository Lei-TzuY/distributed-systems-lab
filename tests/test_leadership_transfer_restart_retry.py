from distlab.leadership_transfer_retry import retry_timeout_now
from distlab.leadership_transfer_transport import LeadershipTransferTransport
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_timeout_now_attempt_survives_transferee_crash_and_restart() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    attempt_id = transport.send_timeout_now("n1", "n2", term=leader.current_term)
    sim.crash("n2")
    sim.run(max_events=1)

    rejections = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert len(rejections) == 1
    assert rejections[0].details["reason"] == "transferee-crashed"
    assert rejections[0].details["retryable"] is True

    sim.restart("n2")
    retry_timeout_now(transport, attempt_id)
    sim.run()

    retries = [record for record in sim.trace if record.kind == "raft-timeout-now-retry"]
    delivered = [record for record in sim.trace if record.kind == "raft-timeout-now-delivered"]
    assert len(retries) == len(delivered) == 1
    assert retries[0].details["attempt_id"] == attempt_id
    assert delivered[0].details["attempt_id"] == attempt_id
    assert cluster.node("n2").role is RaftRole.LEADER
    assert cluster.node("n2").current_term > leader.current_term - 1
    harness.checkpoint()
