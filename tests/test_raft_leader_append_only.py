import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import (
    LeaderAppendOnlyChecker,
    LeaderAppendOnlyViolation,
    RaftSafetyHarness,
)
from distlab.simulator import Simulator


def _elect_single_node_leader() -> tuple[Simulator, RaftCluster, RaftSafetyHarness]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    harness = RaftSafetyHarness(cluster)
    cluster.node("n1").start_election()
    sim.run()
    harness.checkpoint()
    return sim, cluster, harness


def test_leader_append_only_checker_accepts_monotonic_log_growth() -> None:
    sim, cluster, harness = _elect_single_node_leader()
    leader = cluster.node("n1")

    first = LogEntry(term=leader.current_term, command="first")
    sim.persistent_state["n1"]["log"] = (first,)
    harness.checkpoint()

    second = LogEntry(term=leader.current_term, command="second")
    sim.persistent_state["n1"]["log"] = (first, second)
    harness.checkpoint()

    assert harness.leader_append_only.observations[-1].log == (first, second)


def test_safety_harness_rejects_leader_log_truncation() -> None:
    sim, cluster, harness = _elect_single_node_leader()
    leader = cluster.node("n1")
    entries = (
        LogEntry(term=leader.current_term, command="first"),
        LogEntry(term=leader.current_term, command="second"),
    )
    sim.persistent_state["n1"]["log"] = entries
    harness.checkpoint()

    sim.persistent_state["n1"]["log"] = entries[:1]

    with pytest.raises(LeaderAppendOnlyViolation, match="shrunk its log"):
        harness.checkpoint()


def test_safety_harness_rejects_leader_log_overwrite() -> None:
    sim, cluster, harness = _elect_single_node_leader()
    leader = cluster.node("n1")
    original = LogEntry(term=leader.current_term, command="original")
    sim.persistent_state["n1"]["log"] = (original,)
    harness.checkpoint()

    replacement = LogEntry(term=leader.current_term, command="replacement")
    sim.persistent_state["n1"]["log"] = (replacement,)

    with pytest.raises(LeaderAppendOnlyViolation, match="overwrote an entry"):
        harness.checkpoint()


def test_checker_scopes_append_only_to_active_leadership_epoch() -> None:
    sim, cluster, harness = _elect_single_node_leader()
    leader = cluster.node("n1")
    old_entry = LogEntry(term=leader.current_term, command="uncommitted")
    sim.persistent_state["n1"]["log"] = (old_entry,)
    harness.checkpoint()

    sim.persistent_state["n1"]["current_term"] = leader.current_term + 1
    sim.volatile_state["n1"]["role"] = RaftRole.FOLLOWER.value
    sim.persistent_state["n1"]["log"] = ()

    harness.checkpoint()


def test_checker_rejects_non_leader_direct_observation() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    checker = LeaderAppendOnlyChecker()

    with pytest.raises(ValueError, match="currently be a leader"):
        checker.observe_leader(cluster.node("n1"))
