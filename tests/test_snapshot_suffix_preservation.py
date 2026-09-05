from distlab.kv import Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshotStore
from distlab.snapshot_transport import SnapshotTransport


def test_install_snapshot_transport_preserves_matching_follower_suffix() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    for index in range(1, 5):
        leader._persist_log(
            (*leader.log, LogEntry(term=leader.current_term, command=Put("k", str(index))))
        )

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 4

    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    store = KVSnapshotStore(cluster, kv)
    snapshot = store.compact("n1")

    follower = cluster.node("n2")
    retained = LogEntry(term=leader.current_term, command=Put("future", "kept"))
    follower._persist_log((*follower.log, retained))
    assert follower.log_view.prefix_matches(
        snapshot.last_included_index, snapshot.last_included_term
    )
    assert follower.last_log_index == 5

    transport = SnapshotTransport(store)
    transport.send_install_snapshot(
        leader_id="n1",
        follower_id="n2",
        term=leader.current_term,
        snapshot=snapshot,
    )
    sim.run()

    assert store.latest("n2") == snapshot
    assert follower.log_base_index == 4
    assert follower.log_base_term == snapshot.last_included_term
    assert follower.log == (retained,)
    assert follower.log_view.entry_at(5) == retained
    assert follower.last_log_index == 5
    assert follower.commit_index == 4
    assert kv.applier.last_applied("n2") == 4
    assert kv.snapshot("n2") == {"k": "4"}

    installs = [record for record in sim.trace if record.kind == "raft-kv-snapshot-install"]
    assert installs[-1].details["node"] == "n2"
    assert installs[-1].details["previous_last_log_index"] == 5
    assert installs[-1].details["retained_count"] == 1

    sim.crash("n2")
    sim.restart("n2")
    recovered_kv = ReplicatedKV(cluster)
    recovered_store = KVSnapshotStore(cluster, recovered_kv)

    assert recovered_store.latest("n2") == snapshot
    assert follower.log_base_index == 4
    assert follower.log == (retained,)
    assert follower.log_view.entry_at(5) == retained
    assert recovered_kv.snapshot("n2") == {"k": "4"}
    cluster.assert_log_matching()
    recovered_kv.applier.assert_state_machine_safety()
