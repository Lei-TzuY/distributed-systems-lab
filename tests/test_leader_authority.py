import pytest

from distlab.leader_authority import LeaderGenerationGuard
from distlab.leader_runtime import (
    LeaderRuntimeUnavailable,
    StaleLeaderRuntimeGeneration,
)
from distlab.raft import RaftCluster
from distlab.simulator import Simulator


def test_guard_tracks_active_node_local_generation_without_runtime_supervisor() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    guard = LeaderGenerationGuard(cluster)
    node = cluster.node("n1")

    node.start_election()
    first = guard.current("n1")

    assert first.term == 1
    assert first.generation == 1
    assert guard.require("n1", first.generation) == first
    assert not [record for record in sim.trace if record.kind == "raft-leader-runtime-start"]


def test_guard_fences_old_generation_after_same_node_re_election() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    guard = LeaderGenerationGuard(cluster)
    node = cluster.node("n1")
    node.start_election()
    first = guard.current("n1")

    assert node.step_down(reason="test-rollover")
    node.start_election()
    second = guard.current("n1")

    assert second.term == 2
    assert second.generation > first.generation
    with pytest.raises(StaleLeaderRuntimeGeneration, match="stale"):
        guard.require("n1", first.generation)


def test_guard_rejects_crashed_former_leader() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    guard = LeaderGenerationGuard(cluster)
    cluster.node("n1").start_election()
    sim.run()
    identity = guard.current("n1")

    sim.crash("n1")

    with pytest.raises(LeaderRuntimeUnavailable):
        guard.require("n1", identity.generation)
