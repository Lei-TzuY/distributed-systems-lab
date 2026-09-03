from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _elect(cluster: RaftCluster, node_id: str):
    node = cluster.node(node_id)
    node.start_election()
    cluster.sim.run()
    assert node.role is RaftRole.LEADER
    return node


def test_restarted_follower_applies_only_newly_recovered_suffix_and_preserves_dedup() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = _elect(cluster, "n1")
    first = ClientRequest("client-a", 1, Put("key", "value"))

    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=leader.current_term, command=first),
    )
    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == 1

    before_crash = ReplicatedKV(cluster)
    applied = before_crash.apply_committed("n2")
    assert [record.index for record in applied] == [1]
    assert before_crash.snapshot("n2") == {"key": "value"}
    assert before_crash.has_applied_request("n2", "client-a", 1)
    assert before_crash.applier.last_applied("n2") == 1

    sim.crash("n2")

    second = ClientRequest("client-a", 2, Put("other", "fresh"))
    sim.persistent_state["n1"]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=first),
        LogEntry(term=leader.current_term, command=second),
    )
    assert replicator.replicate("n3") is True
    assert leader.commit_index == 3

    sim.restart("n2")
    follower = cluster.node("n2")
    assert follower.commit_index == 0
    assert follower.last_log_index == 1

    recovered = ReplicatedKV(cluster)
    assert recovered.applier.last_applied("n2") == 1
    assert recovered.snapshot("n2") == {"key": "value"}
    assert recovered.has_applied_request("n2", "client-a", 1)
    assert not recovered.has_applied_request("n2", "client-a", 2)

    assert replicator.recover_peer("n2") is True
    assert follower.log == leader.log
    assert follower.commit_index == leader.commit_index == 3

    applied = recovered.apply_committed("n2")
    assert [record.index for record in applied] == [2, 3]
    assert recovered.applier.last_applied("n2") == 3
    assert recovered.snapshot("n2") == {"key": "value", "other": "fresh"}
    assert recovered.has_applied_request("n2", "client-a", 1)
    assert recovered.has_applied_request("n2", "client-a", 2)

    duplicate_applies = [
        record
        for record in sim.trace
        if record.kind == "kv-apply"
        and record.details["node"] == "n2"
        and record.details["index"] == 2
    ]
    assert len(duplicate_applies) == 1
    assert duplicate_applies[0].details["client_id"] == "client-a"
    assert duplicate_applies[0].details["request_id"] == 1
    assert duplicate_applies[0].details["duplicate"] is True

    recovered.applier.assert_state_machine_safety()
    recovered.assert_replica_consistency()

    rebuilt = ReplicatedKV(cluster)
    assert rebuilt.applier.last_applied("n2") == 3
    assert rebuilt.snapshot("n2") == {"key": "value", "other": "fresh"}
    assert rebuilt.has_applied_request("n2", "client-a", 1)
    assert rebuilt.has_applied_request("n2", "client-a", 2)
