import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import ClientSessionError, KVClientSession
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def test_recovery_rejects_pending_write_older_than_durable_sequence_floor() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    session = KVClientSession(clients, "writer")

    session.invoke_write("write-1", 1, Put("x", "one"))
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command=ClientRequest("writer", 1, Put("x", "one"))),
        LogEntry(term=1, command=ClientRequest("writer", 2, Put("x", "two"))),
    )
    cluster.node("n1").advance_commit_index(2, source="test")
    kv.apply_committed("n1")
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    assert kv.client_requests("n1")["writer", 2] == Put("x", "two")
    assert [item.operation_id for item in clients.history.pending()] == ["write-1"]

    with pytest.raises(
        ClientSessionError,
        match="pending write request id precedes recovered durable sequence floor",
    ):
        KVClientSession.recover(clients, "writer", "n1")

    assert [item.operation_id for item in clients.history.pending()] == ["write-1"]
    assert [record for record in sim.trace if record.kind == "client-session-recover-pending"] == []
    safety.checkpoint()
