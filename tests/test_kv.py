import pytest

from distlab.kv import Delete, InvalidKVCommand, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


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


def test_put_and_delete_follow_committed_log_order() -> None:
    log = (
        LogEntry(term=3, command=Put("mode", "safe")),
        LogEntry(term=3, command=Put("mode", "fast")),
        LogEntry(term=3, command=Put("ephemeral", "value")),
        LogEntry(term=3, command=Delete("ephemeral")),
    )
    sim, cluster, replicator = _elect_leader_with_log(log)
    kv = ReplicatedKV(cluster)

    assert replicator.replicate("n2") is True
    applied = kv.apply_committed("n1")

    assert len(applied) == 4
    assert kv.snapshot("n1") == {"mode": "fast"}
    assert kv.get("n1", "ephemeral") is None
    traces = [record for record in sim.trace if record.kind == "kv-apply"]
    assert [record.details["operation"] for record in traces] == [
        "put",
        "put",
        "put",
        "delete",
    ]


def test_uncommitted_kv_suffix_is_not_visible() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command=Put("committed", "yes")),
        LogEntry(term=1, command=Put("uncommitted", "no")),
    )
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(1, source="test")
    kv = ReplicatedKV(cluster)

    kv.apply_committed("n1")

    assert kv.snapshot("n1") == {"committed": "yes"}
    assert kv.get("n1", "uncommitted") is None


def test_followers_converge_after_same_committed_prefix_is_applied() -> None:
    log = (
        LogEntry(term=3, command=Put("a", "1")),
        LogEntry(term=3, command=Put("b", "2")),
        LogEntry(term=3, command=Delete("a")),
    )
    _, cluster, replicator = _elect_leader_with_log(log)
    kv = ReplicatedKV(cluster)

    assert replicator.replicate("n2") is True
    kv.apply_committed("n1")
    assert replicator.replicate("n2") is True
    kv.apply_committed("n2")

    assert kv.snapshot("n1") == {"b": "2"}
    assert kv.snapshot("n2") == kv.snapshot("n1")
    kv.assert_replica_consistency()


def test_recovery_rebuilds_kv_from_durable_applied_history() -> None:
    log = (
        LogEntry(term=1, command=Put("survives", "restart")),
        LogEntry(term=1, command=Put("remove", "me")),
        LogEntry(term=1, command=Delete("remove")),
    )
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(3, source="test")
    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")

    sim.crash("n1")
    sim.restart("n1")
    recovered = ReplicatedKV(cluster)

    assert cluster.node("n1").commit_index == 0
    assert recovered.snapshot("n1") == {"survives": "restart"}
    assert recovered.apply_committed("n1") == ()


def test_invalid_committed_command_is_rejected_before_applied_progress_advances() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command="not-a-kv-command"),)
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(1, source="test")
    kv = ReplicatedKV(cluster)

    with pytest.raises(InvalidKVCommand, match="unsupported KV command"):
        kv.apply_committed("n1")

    assert kv.applier.last_applied("n1") == 0
    assert sim.persistent_state["n1"]["state_machine_applied"] == ()
    assert kv.snapshot("n1") == {}


def test_recovery_rejects_invalid_durable_kv_history() -> None:
    bad = LogEntry(term=1, command="not-a-kv-command")
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (bad,)
    sim.persistent_state["n1"]["state_machine_applied"] = (bad,)
    cluster = RaftCluster(sim, ("n1",))

    with pytest.raises(InvalidKVCommand, match="unsupported KV command"):
        ReplicatedKV(cluster)


def test_delete_of_missing_key_is_deterministic_noop() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command=Delete("missing")),)
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(1, source="test")
    kv = ReplicatedKV(cluster)

    assert len(kv.apply_committed("n1")) == 1
    assert kv.snapshot("n1") == {}


def test_empty_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Put("", "value")
    with pytest.raises(ValueError, match="non-empty"):
        Delete("")
