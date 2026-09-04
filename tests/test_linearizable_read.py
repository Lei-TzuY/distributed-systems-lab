import pytest

from distlab.commit_recovery import append_current_term_barrier
from distlab.kv import Put, ReplicatedKV
from distlab.linearizable_read import (
    CurrentTermCommitRequired,
    LinearizableKVReader,
    LinearizableReadError,
    ReadQuorumUnavailable,
)
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _elect_leader_with_old_value() -> tuple[
    Simulator, RaftCluster, LeaderReplicator, ReplicatedKV
]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = (LogEntry(term=2, command=Put("k", "v1")),)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.current_term == 3
    assert leader.role is RaftRole.LEADER
    return sim, cluster, LeaderReplicator(leader), ReplicatedKV(cluster)


def _commit_current_term_barrier(replicator: LeaderReplicator) -> None:
    leader = replicator.leader
    append_current_term_barrier(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 2


def test_linearizable_read_requires_current_term_commit() -> None:
    _, _, replicator, kv = _elect_leader_with_old_value()
    assert replicator.replicate("n2") is True
    assert replicator.leader.commit_index == 0

    reader = LinearizableKVReader(kv, replicator)
    with pytest.raises(CurrentTermCommitRequired):
        reader.get("k")


def test_linearizable_read_confirms_quorum_then_applies_committed_prefix() -> None:
    sim, cluster, replicator, kv = _elect_leader_with_old_value()
    _commit_current_term_barrier(replicator)

    reader = LinearizableKVReader(kv, replicator)
    assert kv.get("n1", "k") is None
    assert reader.get("k") == "v1"
    assert kv.get("n1", "k") == "v1"

    records = [record for record in sim.trace if record.kind == "raft-linearizable-read"]
    assert len(records) == 1
    assert records[0].details["leader"] == "n1"
    assert records[0].details["term"] == 3
    assert records[0].details["commit_index"] == 2
    assert records[0].details["majority"] == 2
    assert records[0].details["acknowledged_peers"] == ("n2",)
    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()


def test_isolated_leader_cannot_serve_local_value_as_linearizable() -> None:
    sim, _, replicator, kv = _elect_leader_with_old_value()
    _commit_current_term_barrier(replicator)
    kv.apply_committed("n1")
    assert kv.get("n1", "k") == "v1"

    sim.crash("n2")
    sim.crash("n3")
    reader = LinearizableKVReader(kv, replicator)

    with pytest.raises(ReadQuorumUnavailable):
        reader.get("k")

    failures = [
        record for record in sim.trace if record.kind == "raft-linearizable-read-quorum-failed"
    ]
    assert len(failures) == 1
    assert failures[0].details["acknowledgements"] == 1
    assert failures[0].details["majority"] == 2


def test_stale_replicator_cannot_authorize_read_after_new_term() -> None:
    _, cluster, replicator, kv = _elect_leader_with_old_value()
    _commit_current_term_barrier(replicator)

    leader = cluster.node("n1")
    leader.start_election()
    assert leader.current_term == 4

    reader = LinearizableKVReader(kv, replicator)
    with pytest.raises(LinearizableReadError, match="stale"):
        reader.get("k")
