from distlab.raft import RaftCluster, RaftRole, _ElectionTimeout
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_current_generation_timeout_cannot_fire_before_scheduled_deadline() -> None:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 10, "n2": 20, "n3": 30},
    )
    node = cluster.node("n1")

    # Source identity and generation are both valid, but this control event is
    # delivered before the deadline owned by the active local timer.
    sim.send("n1", "n1", _ElectionTimeout(generation=1, deadline=10), delay=1)
    sim.run(max_events=1)

    assert sim.time == 1
    assert node.current_term == 0
    assert node.voted_for is None
    assert node.role is RaftRole.FOLLOWER
    assert not any(record.kind == "raft-election-start" for record in sim.trace)

    rejected = [
        record for record in sim.trace if record.kind == "raft-election-timeout-rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].details == {
        "node": "n1",
        "generation": 1,
        "deadline": 10,
        "expected_deadline": 10,
        "reason": "early-delivery",
    }
    RaftSafetyHarness(cluster).checkpoint()

    # The actual simulator-owned deadline remains authoritative and can still
    # start the election at the configured logical time.
    sim.run(max_events=1)

    assert sim.time == 10
    assert node.current_term == 1
    assert node.voted_for == "n1"
    assert node.role is RaftRole.CANDIDATE
    RaftSafetyHarness(cluster).checkpoint()
