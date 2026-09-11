from distlab.kv import ReplicatedKV
from distlab.raft import RaftCluster
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshot, KVSnapshotStore
from distlab.snapshot_transport import InstallSnapshotRequest, SnapshotTransport


def test_delayed_stale_snapshot_is_acknowledged_without_rollback() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    kv = ReplicatedKV(cluster)
    store = KVSnapshotStore(cluster, kv)
    transport = SnapshotTransport(store)

    newer = KVSnapshot(
        last_included_index=4,
        last_included_term=2,
        state=(("k", "new"),),
        client_requests=(),
    )
    stale = KVSnapshot(
        last_included_index=2,
        last_included_term=1,
        state=(("k", "old"),),
        client_requests=(),
    )
    store.install("n3", newer)

    sim.send(
        "n1",
        "n3",
        InstallSnapshotRequest(
            leader_id="n1",
            follower_id="n3",
            term=2,
            snapshot=stale,
        ),
        delivery_dst=transport.endpoint("n3"),
    )
    sim.run()

    follower = cluster.node("n3")
    assert store.latest("n3") == newer
    assert follower.log_base_index == 4
    assert follower.log_base_term == 2
    assert follower.commit_index == 4
    assert kv.snapshot("n3") == {"k": "new"}

    stale_records = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-stale"
    ]
    responses = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-response"
    ]
    assert len(stale_records) == 1
    assert stale_records[0].details["incoming_index"] == 2
    assert stale_records[0].details["installed_index"] == 4
    assert responses[-1].details["success"] is True
    assert responses[-1].details["requested_last_included_index"] == 2
    assert responses[-1].details["last_included_index"] == 2

    RaftSafetyHarness(cluster).checkpoint()
    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()
