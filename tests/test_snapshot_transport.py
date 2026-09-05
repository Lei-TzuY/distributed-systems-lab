import pytest

from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator, ReplicationResponseMissing
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator
from distlab.snapshot import KVSnapshotStore
from distlab.snapshot_transport import SnapshotTransport


def test_lagging_follower_installs_snapshot_then_resumes_append_entries() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    commands = (
        ClientRequest("client-a", 1, Put("k", "v1")),
        ClientRequest("client-a", 1, Put("k", "v1")),
        ClientRequest("client-b", 2, Put("other", "value")),
        Put("k", "v2"),
    )
    for command in commands:
        leader._persist_log((*leader.log, LogEntry(term=leader.current_term, command=command)))

    initial = LeaderReplicator(leader)
    assert initial.replicate("n2") is True
    assert leader.commit_index == 4

    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    store = KVSnapshotStore(cluster, kv)
    snapshot = store.compact("n1")
    follower = cluster.node("n3")
    assert follower.last_log_index == 0
    assert leader.log_base_index == snapshot.last_included_index == 4

    transport = SnapshotTransport(store)
    replicator = LeaderReplicator(leader, snapshot_transport=transport)
    assert replicator.recover_peer("n3", max_attempts=3) is True

    assert follower.log_base_index == 4
    assert follower.log_base_term == snapshot.last_included_term
    assert follower.commit_index == 4
    assert kv.snapshot("n3") == {"k": "v2", "other": "value"}
    assert kv.has_applied_request("n3", "client-a", 1)
    assert replicator.progress("n3").match_index == 4
    assert replicator.progress("n3").next_index == 5

    snapshots = [record for record in sim.trace if record.kind == "raft-replication-snapshot"]
    requests = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-request"
    ]
    responses = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-response"
    ]
    assert len(snapshots) == len(requests) == len(responses) == 1
    assert snapshots[0].details["next_index"] == 4
    assert responses[0].details["success"] is True
    assert responses[0].details["last_included_index"] == 4

    leader._persist_log(
        (*leader.log, LogEntry(term=leader.current_term, command=Put("post", "v3")))
    )
    assert replicator.replicate("n3") is True
    assert leader.commit_index == 5
    assert replicator.replicate("n3") is True
    assert follower.commit_index == 5

    kv.apply_committed("n1")
    kv.apply_committed("n3")
    assert kv.snapshot("n1") == kv.snapshot("n3") == {
        "k": "v2",
        "other": "value",
        "post": "v3",
    }

    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()


def test_snapshot_recovery_obeys_logical_partition_and_retries_after_heal() -> None:
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
    follower = cluster.node("n3")

    sim.partition(("n1",), ("n3",))
    with pytest.raises(ReplicationResponseMissing, match="no InstallSnapshot response"):
        replicator.recover_peer("n3", max_attempts=1)

    assert follower.log_base_index == 0
    partition_drops = [record for record in sim.trace if record.kind == "partition-drop"]
    assert partition_drops
    assert partition_drops[-1].details["src"] == "n1"
    assert partition_drops[-1].details["dst"] == "n3"

    sim.heal_partition(("n1",), ("n3",))
    assert replicator.recover_peer("n3", max_attempts=2) is True
    assert follower.log_base_index == snapshot.last_included_index == 4
    assert follower.log_base_term == snapshot.last_included_term
    assert kv.snapshot("n3") == {"k": "v4"}

    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()


def test_snapshot_recovery_obeys_explicit_logical_drop_rule() -> None:
    sim = Simulator(
        fault_plan=FaultPlan((FaultRule(FaultAction.DROP, src="n1", dst="n3"),))
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    leader._persist_log(
        (
            LogEntry(term=leader.current_term, command=Put("k", "v1")),
            LogEntry(term=leader.current_term, command=Put("k", "v2")),
        )
    )
    initial = LeaderReplicator(leader)
    assert initial.replicate("n2") is True

    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    store = KVSnapshotStore(cluster, kv)
    snapshot = store.compact("n1")
    transport = SnapshotTransport(store)
    replicator = LeaderReplicator(leader, snapshot_transport=transport)
    follower = cluster.node("n3")
    trace_start = len(sim.trace)

    with pytest.raises(ReplicationResponseMissing, match="no InstallSnapshot response"):
        replicator.recover_peer("n3", max_attempts=1)

    snapshot_drops = [
        record
        for record in sim.trace[trace_start:]
        if record.kind == "drop" and record.details["src"] == "n1" and record.details["dst"] == "n3"
    ]
    assert len(snapshot_drops) == 1
    assert follower.log_base_index == 0

    sim.fault_plan = FaultPlan()
    assert replicator.recover_peer("n3", max_attempts=2) is True
    assert follower.log_base_index == snapshot.last_included_index == 2
    assert kv.snapshot("n3") == {"k": "v2"}

    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()
