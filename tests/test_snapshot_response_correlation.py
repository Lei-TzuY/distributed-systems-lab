from distlab.kv import Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshotStore
from distlab.snapshot_transport import InstallSnapshotResponse, SnapshotTransport


def test_delayed_old_snapshot_response_cannot_complete_newer_snapshot_attempt() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    for index in range(1, 5):
        leader._persist_log(
            (*leader.log, LogEntry(term=leader.current_term, command=Put("k", f"v{index}")))
        )

    initial = LeaderReplicator(leader)
    assert initial.replicate("n2") is True
    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    store = KVSnapshotStore(cluster, kv)
    snapshot = store.compact("n1")
    transport = SnapshotTransport(store)
    replicator = LeaderReplicator(leader, snapshot_transport=transport)

    assert replicator.replicate("n3", max_attempts=1) is False
    assert replicator.progress("n3").next_index == snapshot.last_included_index == 4

    sim.send(
        "n3",
        "n1",
        InstallSnapshotResponse(
            term=leader.current_term,
            leader_id="n1",
            follower_id="n3",
            success=True,
            last_included_index=2,
            requested_last_included_index=2,
        ),
        delay=100,
        delivery_dst=transport.endpoint("n1"),
    )

    assert replicator.replicate("n3", max_attempts=1) is True
    assert replicator.progress("n3").match_index == snapshot.last_included_index
    assert replicator.progress("n3").next_index == snapshot.last_included_index + 1
    assert cluster.node("n3").log_base_index == snapshot.last_included_index
    assert kv.snapshot("n3") == {"k": "v4"}

    responses = [
        record
        for record in sim.trace
        if record.kind == "raft-install-snapshot-response"
        and record.details["leader"] == "n1"
        and record.details["follower"] == "n3"
    ]
    assert responses[-1].details["requested_last_included_index"] == 2
    assert any(
        record.details["requested_last_included_index"] == snapshot.last_included_index
        for record in responses
    )

    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()
