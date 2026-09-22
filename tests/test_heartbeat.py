import pytest

from distlab.heartbeat import LeaderHeartbeatController
from distlab.leader_liveness import LeaderQuorumMonitor, LeaderQuorumUnavailable
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _cluster_with_timeouts() -> tuple[Simulator, RaftCluster]:
    sim = Simulator()
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 20, "n2": 10, "n3": 12},
    )
    cluster.node("n1").start_election()
    sim.run_until_time(2)
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def test_bounded_heartbeat_rounds_keep_followers_from_campaigning() -> None:
    sim, cluster = _cluster_with_timeouts()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    controller = LeaderHeartbeatController(
        LeaderQuorumMonitor(LeaderReplicator(leader)),
        heartbeat_interval=3,
        response_timeout=2,
    )

    rounds = controller.run(rounds=3)

    assert [round_.scheduled_time for round_ in rounds] == [5, 8, 11]
    assert [round_.completed_time for round_ in rounds] == [7, 10, 13]
    assert controller.next_heartbeat_time == 14
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 1
    assert not any(
        record.kind == "raft-pre-vote-start"
        and record.details["node"] in {"n2", "n3"}
        for record in sim.trace
    )
    completed = [
        record
        for record in sim.trace
        if record.kind == "raft-heartbeat-round-complete"
    ]
    assert [record.details["round"] for record in completed] == [1, 2, 3]
    harness.checkpoint()


def test_next_heartbeat_round_retires_partitioned_leader_in_same_term() -> None:
    sim, cluster = _cluster_with_timeouts()
    leader = cluster.node("n1")
    harness = RaftSafetyHarness(cluster)
    controller = LeaderHeartbeatController(
        LeaderQuorumMonitor(LeaderReplicator(leader)),
        heartbeat_interval=3,
        response_timeout=2,
    )

    first = controller.run(rounds=1)
    assert first[0].scheduled_time == 5
    assert leader.role is RaftRole.LEADER

    sim.partition(("n1",), ("n2", "n3"))
    with pytest.raises(LeaderQuorumUnavailable, match="could not confirm"):
        controller.run(rounds=1)

    assert sim.time == 10
    assert leader.current_term == 1
    assert leader.voted_for == "n1"
    assert leader.role is RaftRole.FOLLOWER
    failures = [
        record
        for record in sim.trace
        if record.kind == "raft-heartbeat-round-failed"
    ]
    assert failures[-1].details["round"] == 2
    assert failures[-1].details["scheduled_time"] == 8
    step_down = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-step-down"
    ]
    assert step_down[-1].details["reason"] == "check-quorum-failed"
    harness.checkpoint()


def test_heartbeat_window_does_not_consume_response_scheduled_after_deadline() -> None:
    sim, cluster = _cluster_with_timeouts()
    leader = cluster.node("n1")
    controller = LeaderHeartbeatController(
        LeaderQuorumMonitor(LeaderReplicator(leader)),
        heartbeat_interval=3,
        response_timeout=2,
    )
    sim.set_link_delay("n1", "n2", 5)
    sim.partition(("n1",), ("n3",))

    with pytest.raises(LeaderQuorumUnavailable):
        controller.run(rounds=1)

    assert sim.time == 7
    assert not any(
        record.kind == "raft-append-entries"
        and record.details["follower"] == "n2"
        and record.time <= 7
        for record in sim.trace
    )

    sim.run_until_time(11)

    assert any(
        record.kind == "raft-append-entries"
        and record.details["follower"] == "n2"
        and record.time == 11
        for record in sim.trace
    )


def test_heartbeat_controller_rejects_overlapping_response_window() -> None:
    sim, cluster = _cluster_with_timeouts()
    monitor = LeaderQuorumMonitor(LeaderReplicator(cluster.node("n1")))

    with pytest.raises(ValueError, match="cannot exceed"):
        LeaderHeartbeatController(
            monitor,
            heartbeat_interval=2,
            response_timeout=3,
        )
