import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_older_unseen_client_request_is_noop_after_newer_request_applied() -> None:
    newer = ClientRequest("writer", 5, Put("x", "five"))
    stale = ClientRequest("writer", 4, Put("x", "stale-four"))
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command=newer),
        LogEntry(term=1, command=stale),
    )
    cluster = RaftCluster(sim, ("n1",))
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    session = KVClientSession(clients, "writer", last_completed_request_id=3)
    invoked = session.invoke_write("write-4", 4, stale.operation)
    assert invoked == stale

    cluster.node("n1").advance_commit_index(2, source="test")
    applied = kv.apply_committed("n1")

    assert [record.index for record in applied] == [1, 2]
    assert kv.get("n1", "x") == "five"
    assert kv.client_requests("n1") == {("writer", 5): newer.operation}
    stale_apply = [
        record
        for record in sim.trace
        if record.kind == "kv-apply" and record.details["request_id"] == 4
    ]
    assert len(stale_apply) == 1
    assert stale_apply[0].details["stale"] is True
    assert stale_apply[0].details["duplicate"] is False

    with pytest.raises(RuntimeError, match="before the target replica applied"):
        session.complete_write("write-4", "n1")

    assert session.pending_write() is not None
    assert session.last_completed_request_id == 3
    assert [
        record
        for record in sim.trace
        if record.kind == "client-response" and record.details["operation_id"] == "write-4"
    ] == []
    RaftSafetyHarness(cluster).checkpoint()
