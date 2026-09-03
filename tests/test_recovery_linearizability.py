import pytest

from distlab import (
    KVClientHistory,
    LogEntry,
    Put,
    RaftCluster,
    RaftRole,
    ReplicatedKV,
    Simulator,
    SingleKeyKVLinearizabilityChecker,
)
from distlab.replication import LeaderReplicator


def _elect(cluster: RaftCluster, node_id: str):
    node = cluster.node(node_id)
    node.start_election()
    cluster.sim.run()
    assert node.role is RaftRole.LEADER
    return node


def test_client_retry_across_leader_crash_is_one_linearizable_write() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = _elect(cluster, "n1")
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)

    request = clients.invoke_write("write", "client-a", 1, Put("key", "value"))
    sim.persistent_state[leader.node_id]["log"] = (
        LogEntry(term=leader.current_term, command=request),
    )

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == 1
    kv.apply_committed("n2")
    assert kv.has_applied_request("n2", "client-a", 1)

    # The write is durable on a future leader but its response is lost with n1.
    sim.crash("n1")
    assert clients.pending_write("write") == request

    replacement = _elect(cluster, "n2")
    retried = clients.retry_write("write")
    assert retried is request
    assert len(clients.history.invocations()) == 1
    assert len(clients.history.pending()) == 1

    sim.persistent_state[replacement.node_id]["log"] = (
        *replacement.log,
        LogEntry(term=replacement.current_term, command=retried),
    )
    replacement_replicator = LeaderReplicator(replacement)
    assert replacement_replicator.replicate("n3") is True
    assert replacement.commit_index == 2

    applied = kv.apply_committed("n2")
    assert [record.index for record in applied] == [2]
    duplicate = [
        record
        for record in sim.trace
        if record.kind == "kv-apply"
        and record.details["node"] == "n2"
        and record.details["index"] == 2
    ]
    assert len(duplicate) == 1
    assert duplicate[0].details["duplicate"] is True

    clients.complete_write("write", "n2")
    assert clients.read("read", "client-b", "n2", "key") == "value"

    result = SingleKeyKVLinearizabilityChecker().check(clients.history)
    assert result.linearizable
    assert result.order == ("write", "read")
    assert len(clients.history.invocations()) == 2
    assert [record.kind for record in sim.trace if record.kind.startswith("client-")] == [
        "client-invoke",
        "client-retry",
        "client-response",
        "client-invoke",
        "client-response",
    ]


def test_retry_requires_the_original_write_to_still_be_pending() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)

    with pytest.raises(ValueError, match="unknown pending write"):
        clients.retry_write("missing")

    request = clients.invoke_write("write", "client-a", 1, Put("key", "value"))
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command=request),)
    cluster.node("n1").advance_commit_index(1, source="test")
    kv.apply_committed("n1")
    clients.complete_write("write", "n1")

    with pytest.raises(ValueError, match="unknown pending write"):
        clients.retry_write("write")
