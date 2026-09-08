from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_delayed_append_entries_from_removed_leader_is_rejected() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            rules=(
                FaultRule(
                    FaultAction.DELAY,
                    src="n2",
                    dst="n1",
                    ordinal=2,
                    extra_delay=10,
                ),
                FaultRule(
                    FaultAction.DELAY,
                    src="n2",
                    dst="n1",
                    ordinal=3,
                    extra_delay=20,
                ),
            )
        )
    )
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4"),
        voters=("n1", "n2", "n3", "n4"),
    )
    harness = RaftSafetyHarness(cluster)

    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1

    cluster.node("n2").start_election()
    while cluster.node("n2").role is not RaftRole.LEADER:
        sim.run(max_events=1)
    assert cluster.node("n2").current_term == 2
    assert cluster.node("n1").current_term == 1

    cluster.node("n2").send_append_entries("n1")
    cluster.begin_joint_consensus("n1", ("n1", "n3", "n4"))
    cluster.finalize_membership("n1")
    assert not cluster.is_voter("n2")

    sim.run()

    assert cluster.node("n1").current_term == 1
    assert cluster.node("n1").role is RaftRole.LEADER
    rejected = [
        record
        for record in sim.trace
        if record.kind == "raft-append-entries-rejected"
        and record.details["follower"] == "n1"
        and record.details["leader"] == "n2"
    ]
    assert rejected[-1].details["term"] == 2
    assert rejected[-1].details["current_term"] == 1
    assert rejected[-1].details["reason"] == "leader-not-voter"
    harness.checkpoint()
