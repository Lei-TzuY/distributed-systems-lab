import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator
from distlab.state_machine import StateMachineApplier, StateMachineSafetyViolation


def _elect_leader_with_log(
    log: tuple[LogEntry, ...], *, node_ids: tuple[str, ...] = ("n1", "n2", "n3")
) -> tuple[Simulator, RaftCluster, LeaderReplicator]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, node_ids)
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.current_term == 3
    assert leader.role is RaftRole.LEADER
    return sim, cluster, LeaderReplicator(leader)


def test_apply_committed_advances_last_applied_in_log_order() -> None:
    log = (
        LogEntry(term=2, command="old"),
        LogEntry(term=3, command="new"),
    )
    sim, cluster, replicator = _elect_leader_with_log(log)
    applier = StateMachineApplier(cluster)

    assert replicator.replicate("n2") is True
    applied = applier.apply_committed("n1")

    assert [record.index for record in applied] == [1, 2]
    assert applier.last_applied("n1") == 2
    assert applier.applied_entries("n1") == log
    traces = [record for record in sim.trace if record.kind == "raft-state-machine-apply"]
    assert [(record.details["index"], record.details["command"]) for record in traces] == [
        (1, "old"),
        (2, "new"),
    ]


def test_apply_committed_is_idempotent_until_commit_index_advances() -> None:
    log = (LogEntry(term=3, command="once"),)
    _, cluster, replicator = _elect_leader_with_log(log)
    applier = StateMachineApplier(cluster)

    assert replicator.replicate("n2") is True
    assert len(applier.apply_committed("n1")) == 1
    assert applier.apply_committed("n1") == ()
    assert applier.last_applied("n1") == 1


def test_follower_applies_same_committed_prefix_after_leader_commit_propagates() -> None:
    log = (
        LogEntry(term=2, command="a"),
        LogEntry(term=3, command="b"),
    )
    _, cluster, replicator = _elect_leader_with_log(log)
    applier = StateMachineApplier(cluster)

    assert replicator.replicate("n2") is True
    assert applier.apply_committed("n1")
    assert cluster.node("n2").commit_index == 0

    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == 2
    assert applier.apply_committed("n2")
    assert applier.applied_entries("n2") == applier.applied_entries("n1")
    applier.assert_state_machine_safety()


def test_uncommitted_suffix_is_not_applied() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command="committed"),
        LogEntry(term=2, command="uncommitted"),
    )
    cluster = RaftCluster(sim, ("n1",))
    node = cluster.node("n1")
    node.advance_commit_index(1, source="test")
    applier = StateMachineApplier(cluster)

    assert applier.apply_committed("n1")
    assert applier.last_applied("n1") == 1
    assert applier.applied_entries("n1") == (LogEntry(term=1, command="committed"),)


def test_durable_applied_prefix_survives_crash_and_new_applier() -> None:
    log = (
        LogEntry(term=1, command="first"),
        LogEntry(term=1, command="second"),
    )
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(2, source="test")
    applier = StateMachineApplier(cluster)

    assert len(applier.apply_committed("n1")) == 2
    assert sim.persistent_state["n1"]["state_machine_applied"] == log

    sim.crash("n1")
    sim.restart("n1")
    recovered = StateMachineApplier(cluster)

    assert cluster.node("n1").commit_index == 0
    assert recovered.last_applied("n1") == 2
    assert recovered.applied_entries("n1") == log
    assert recovered.apply_committed("n1") == ()


def test_reestablished_commit_index_does_not_reapply_durable_commands() -> None:
    log = (LogEntry(term=1, command="once"),)
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    node = cluster.node("n1")
    node.advance_commit_index(1, source="test")
    applier = StateMachineApplier(cluster)
    assert len(applier.apply_committed("n1")) == 1

    sim.crash("n1")
    sim.restart("n1")
    recovered = StateMachineApplier(cluster)
    node.advance_commit_index(1, source="recovered-test")

    assert recovered.apply_committed("n1") == ()
    apply_traces = [record for record in sim.trace if record.kind == "raft-state-machine-apply"]
    assert len(apply_traces) == 1


def test_recovery_rejects_durable_history_that_diverges_from_log() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command="log"),)
    sim.persistent_state["n1"]["state_machine_applied"] = (
        LogEntry(term=1, command="different"),
    )
    cluster = RaftCluster(sim, ("n1",))

    with pytest.raises(StateMachineSafetyViolation, match="diverges from the persistent Raft log"):
        StateMachineApplier(cluster)


def test_state_machine_safety_rejects_different_entry_at_same_applied_index() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command="left"),)
    sim.persistent_state["n2"]["log"] = (LogEntry(term=2, command="right"),)
    cluster = RaftCluster(sim, ("n1", "n2"))
    cluster.node("n1").advance_commit_index(1, source="test")
    cluster.node("n2").advance_commit_index(1, source="test")
    applier = StateMachineApplier(cluster)

    assert applier.apply_committed("n1")
    with pytest.raises(StateMachineSafetyViolation, match="State Machine Safety violated"):
        applier.apply_committed("n2")


def test_safety_harness_checks_state_machine_after_application() -> None:
    log = (
        LogEntry(term=2, command="old"),
        LogEntry(term=3, command="new"),
    )
    _, cluster, replicator = _elect_leader_with_log(log)
    harness = RaftSafetyHarness(cluster)

    assert replicator.replicate("n2") is True
    assert harness.state_machine.apply_committed("n1")
    harness.checkpoint()

    assert replicator.replicate("n2") is True
    assert harness.state_machine.apply_committed("n2")
    harness.checkpoint()
    assert harness.state_machine.applied_entries("n1") == log
    assert harness.state_machine.applied_entries("n2") == log


def test_safety_harness_rejects_durable_applied_history_mutation_after_restart() -> None:
    log = (LogEntry(term=3, command="stable"),)
    sim, cluster, replicator = _elect_leader_with_log(log, node_ids=("n1", "n2", "n3"))
    harness = RaftSafetyHarness(cluster)

    assert replicator.replicate("n2") is True
    assert harness.state_machine.apply_committed("n1")
    harness.checkpoint()

    sim.crash("n1")
    sim.restart("n1")
    harness.checkpoint()
    sim.persistent_state["n1"]["state_machine_applied"] = (
        LogEntry(term=3, command="corrupt"),
    )

    with pytest.raises(StateMachineSafetyViolation, match="diverges from the persistent Raft log"):
        harness.checkpoint()


def test_unknown_node_is_rejected() -> None:
    cluster = RaftCluster(Simulator(), ("n1",))
    applier = StateMachineApplier(cluster)

    with pytest.raises(ValueError, match="unknown Raft node"):
        applier.apply_committed("missing")
