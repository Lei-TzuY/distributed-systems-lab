from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import RaftRole, RequestVoteResponse
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_removed_voter_response_cannot_force_higher_term_adoption() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3"),
        voters=("n1", "n2", "n3"),
        election_timeouts={"n1": 100, "n2": 110, "n3": 120},
    )
    safety = RaftSafetyHarness(cluster)

    cluster.node("n1").start_election()
    sim.run(max_events=4)
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    safety.checkpoint()

    sim.send(
        "n2",
        "n1",
        RequestVoteResponse(term=5, voter_id="n2", vote_granted=False),
    )
    cluster.begin_joint_consensus("n1", ("n1", "n3"))
    cluster.finalize_membership("n1")
    assert not cluster.is_voter("n2")

    sim.run(max_events=1)

    assert cluster.node("n1").current_term == 1
    assert cluster.node("n1").role is RaftRole.LEADER
    rejected = [
        record
        for record in sim.trace
        if record.kind == "raft-vote-response-rejected"
        and record.details["node"] == "n1"
        and record.details["voter"] == "n2"
    ]
    assert rejected[-1].details["reason"] == "voter-not-voter"
    assert rejected[-1].details["term"] == 5
    safety.checkpoint()
