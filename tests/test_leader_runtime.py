import pytest

from distlab.leader_liveness import LeaderQuorumUnavailable
from distlab.leader_runtime import (
    LeaderRuntimeSupervisor,
    LeaderRuntimeUnavailable,
    StaleLeaderRuntimeGeneration,
)
from distlab.leadership_transfer import LeadershipTransfer
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import LogEntry, RaftCluster, RaftRole, RequestVote
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _supervised_cluster() -> tuple[Simulator, RaftCluster, LeaderRuntimeSupervisor]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster, supervisor


def test_supervisor_replays_existing_leader_generation_on_attach() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()

    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )

    identity = supervisor.runtime_identity("n1")
    assert identity.leader_id == "n1"
    assert identity.term == 1
    assert identity.generation == 1
    starts = [record for record in sim.trace if record.kind == "raft-leader-runtime-start"]
    assert starts[-1].details["reason"] == "observer-replay"


def test_leadership_transfer_retires_source_and_starts_target_runtime() -> None:
    sim, cluster, supervisor = _supervised_cluster()
    harness = RaftSafetyHarness(cluster)
    source = supervisor.runtime_identity("n1")

    result = LeadershipTransfer(cluster.node("n1")).transfer("n2")

    assert result.previous_term == 1
    assert result.new_term == 2
    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    target = supervisor.runtime_identity("n2")
    assert target.term == 2
    assert target.generation > source.generation
    stops = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-stop" and record.details["node"] == "n1"
    ]
    assert stops[-1].details["reason"] == "higher-term-observed"
    starts = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-start" and record.details["node"] == "n2"
    ]
    assert starts[-1].details["generation"] == target.generation
    harness.checkpoint()


def test_partition_can_hold_two_local_runtime_generations_until_check_quorum() -> None:
    sim, cluster, supervisor = _supervised_cluster()
    harness = RaftSafetyHarness(cluster)
    first = supervisor.runtime_identity("n1")
    sim.partition(("n1",), ("n2", "n3"))

    cluster.node("n2").start_election()
    sim.run()

    second = supervisor.runtime_identity("n2")
    assert first.term == 1
    assert second.term == 2
    assert {identity.leader_id for identity in supervisor.active_runtimes} == {"n1", "n2"}
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n2").role is RaftRole.LEADER

    with pytest.raises(LeaderQuorumUnavailable):
        supervisor.run_heartbeats(
            "n1",
            rounds=1,
            expected_generation=first.generation,
        )

    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    assert supervisor.runtime_identity("n2") == second
    assert cluster.node("n1").role is RaftRole.FOLLOWER
    assert cluster.node("n2").role is RaftRole.LEADER
    harness.checkpoint()


def test_leader_crash_and_restart_retire_runtime_before_volatile_reset() -> None:
    sim, cluster, supervisor = _supervised_cluster()
    generation = supervisor.runtime_identity("n1").generation

    sim.crash("n1")

    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    retired = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-retired"
        and record.details["generation"] == generation
    ]
    assert retired[-1].details["reason"] == "node-crash"
    assert not sim.volatile_state["n1"]

    sim.restart("n1")
    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    assert cluster.node("n1").role is RaftRole.FOLLOWER


def test_direct_restart_of_live_leader_fails_closed_and_retires_runtime() -> None:
    sim, cluster, supervisor = _supervised_cluster()
    generation = supervisor.runtime_identity("n1").generation

    sim.restart("n1")

    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    retired = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-retired"
        and record.details["generation"] == generation
    ]
    assert retired[-1].details["reason"] == "node-restart"
    assert cluster.node("n1").role is RaftRole.FOLLOWER


def test_re_elected_same_node_gets_new_generation_and_fences_stale_callers() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    node = cluster.node("n1")
    node.start_election()
    first = supervisor.runtime_identity("n1")

    assert node.step_down(reason="test-generation-rollover")
    node.start_election()
    second = supervisor.runtime_identity("n1")

    assert second.term == 2
    assert second.generation > first.generation
    with pytest.raises(StaleLeaderRuntimeGeneration, match="stale"):
        supervisor.run_heartbeats(
            "n1",
            rounds=1,
            expected_generation=first.generation,
        )


def test_reconfigurable_supervisor_uses_joint_quorum_for_heartbeat_runtime() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run()
    identity = supervisor.runtime_identity("n1")
    start = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-start" and record.details["node"] == "n1"
    ][-1]
    assert start.details["membership_aware"] is True

    cluster.begin_joint_consensus("n1", ("n1", "n4", "n5"))
    sim.partition(("n1",), ("n4", "n5"))

    with pytest.raises(LeaderQuorumUnavailable):
        supervisor.run_heartbeats(
            "n1",
            rounds=1,
            expected_generation=identity.generation,
        )

    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    failure = [record for record in sim.trace if record.kind == "raft-check-quorum-failed"][-1]
    assert failure.details["quorum_mode"] == "joint"
    assert failure.details["old_majority"] == 2
    assert failure.details["new_majority"] == 2


def test_rejected_higher_term_vote_retires_runtime_and_rearms_former_leader() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command="leader-only"),)
    cluster = RaftCluster(
        sim,
        ("n1", "n2", "n3"),
        election_timeouts={"n1": 10, "n2": 100, "n3": 100},
    )
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run(max_events=4)
    assert cluster.node("n1").role is RaftRole.LEADER
    generation = supervisor.runtime_identity("n1").generation

    sim.send(
        "n2",
        "n1",
        RequestVote(
            term=2,
            candidate_id="n2",
            last_log_index=0,
            last_log_term=0,
        ),
    )
    sim.run(max_events=1)

    node = cluster.node("n1")
    assert node.current_term == 2
    assert node.role is RaftRole.FOLLOWER
    assert node.voted_for is None
    with pytest.raises(LeaderRuntimeUnavailable):
        supervisor.runtime_identity("n1")
    retired = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-runtime-retired"
        and record.details["generation"] == generation
    ]
    assert retired[-1].details["reason"] == "higher-term-observed"
    resets = [
        record
        for record in sim.trace
        if record.kind == "raft-election-timeout-reset"
        and record.details["node"] == "n1"
    ]
    assert resets[-1].details["reason"] == "higher-term-vote-rejected"
    assert resets[-1].details["deadline"] == 13
