from distlab.leadership_transfer_transport import LeadershipTransferTransport
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_timeout_now_is_rejected_after_voting_configuration_changes() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1

    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    attempt_id = transport.send_timeout_now("n1", "n2", term=1)
    cluster.begin_joint_consensus("n1", ("n1", "n2"))
    sim.run(max_events=1)

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert cluster.node("n2").current_term == 1
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["attempt_id"] == attempt_id
    assert rejected[-1].details["reason"] == "membership-configuration-changed"
    harness.checkpoint()
