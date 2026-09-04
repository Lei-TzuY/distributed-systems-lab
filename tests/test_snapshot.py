import pytest

from distlab.kv import ClientRequest, Delete, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshotStore, SnapshotClientRequest


def _committed_kv() -> tuple[Simulator, RaftCluster, ReplicatedKV]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    requests = (
        ClientRequest("client-a", 1, Put("k", "v1")),
        ClientRequest("client-a", 1, Put("k", "v1")),
        ClientRequest("client-b", 4, Put("other", "value")),
        Delete("other"),
    )
    for command in requests:
        leader._persist_log((*leader.log, LogEntry(term=leader.current_term, command=command)))

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 4
    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    return sim, cluster, kv


def test_snapshot_persists_state_and_dedup_identity_across_restart() -> None:
    sim, cluster, kv = _committed_kv()
    store = KVSnapshotStore(cluster, kv)

    snapshot = store.create("n1")
    assert snapshot.last_included_index == 4
    assert snapshot.last_included_term == 1
    assert snapshot.state == (("k", "v1"),)
    assert snapshot.client_requests == (
        SnapshotClientRequest("client-a", 1, Put("k", "v1")),
        SnapshotClientRequest("client-b", 4, Put("other", "value")),
    )

    sim.crash("n1")
    sim.restart("n1")
    recovered_kv = ReplicatedKV(cluster)
    recovered = KVSnapshotStore(cluster, recovered_kv).latest("n1")
    assert recovered == snapshot
    assert recovered_kv.snapshot("n1") == {"k": "v1"}
    assert recovered_kv.has_applied_request("n1", "client-a", 1)

    records = [record for record in sim.trace if record.kind == "raft-kv-snapshot-persist"]
    assert len(records) == 1
    assert records[0].details["last_included_index"] == 4
    assert records[0].details["client_request_count"] == 2
    cluster.assert_log_matching()
    recovered_kv.applier.assert_state_machine_safety()


def test_snapshot_creation_uses_absolute_index_after_durable_compaction() -> None:
    sim, cluster, kv = _committed_kv()
    leader = cluster.node("n1")
    compacted = leader.log_view.compact_through(2)
    sim.persistent_state["n1"]["log"] = compacted.entries
    sim.persistent_state["n1"]["log_base_index"] = compacted.base_index
    sim.persistent_state["n1"]["log_base_term"] = compacted.base_term

    snapshot = KVSnapshotStore(cluster, kv).create("n1")

    assert leader.log_base_index == 2
    assert leader.last_log_index == 4
    assert snapshot.last_included_index == 4
    assert snapshot.last_included_term == leader.log_view.term_at(4) == 1
    assert snapshot.state == (("k", "v1"),)
    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()


def test_snapshot_rejects_empty_state_machine() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    kv = ReplicatedKV(cluster)

    with pytest.raises(ValueError, match="empty"):
        KVSnapshotStore(cluster, kv).create("n1")


def test_snapshot_validation_detects_durable_metadata_corruption() -> None:
    sim, cluster, kv = _committed_kv()
    store = KVSnapshotStore(cluster, kv)
    snapshot = store.create("n1")
    sim.persistent_state["n1"][store._PERSISTENT_KEY] = type(snapshot)(
        last_included_index=snapshot.last_included_index,
        last_included_term=snapshot.last_included_term + 1,
        state=snapshot.state,
        client_requests=snapshot.client_requests,
    )

    with pytest.raises(AssertionError, match="term diverges"):
        KVSnapshotStore(cluster, kv)
