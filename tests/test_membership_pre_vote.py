from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_joint_consensus_pre_vote_requires_old_and_new_majorities() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    harness = RaftSafetyHarness(cluster)

    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER
    assert source.current_term == 1
    harness.checkpoint()

    cluster.begin_joint_consensus("n1", ("n3", "n4", "n5"))
    sim.crash("n4")
    sim.crash("n5")

    target = cluster.node("n3")
    target.start_pre_vote()
    sim.run()

    assert target.current_term == 1
    assert target.role is RaftRole.FOLLOWER
    assert target.pre_votes_received >= frozenset({"n1", "n2", "n3"})
    assert not cluster.has_election_quorum(target.pre_votes_received)
    assert not any(
        record.kind == "raft-election-start"
        and record.details["node"] == "n3"
        and record.details["term"] == 2
        for record in sim.trace
    )
    harness.checkpoint()

    sim.restart("n4")
    target.start_pre_vote()
    sim.run()

    assert target.current_term == 2
    assert target.role is RaftRole.LEADER
    assert cluster.leaders_by_term[2] == "n3"
    harness.checkpoint()
