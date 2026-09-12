import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import ClientSessionError, KVClientSession
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _applied_single_node_request(operation: Put) -> tuple[
    Simulator,
    RaftCluster,
    ReplicatedKV,
]:
    sim = Simulator()
    request = ClientRequest("writer", 1, operation)
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1, command=request),)
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(1, source="test")
    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    return sim, cluster, kv


def test_session_recovery_allows_equal_floor_pending_write_with_matching_identity() -> None:
    sim, cluster, kv = _applied_single_node_request(Put("x", "one"))
    clients = KVClientHistory(kv)
    request = clients.invoke_write("write-1", "writer", 1, Put("x", "one"))
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    session = KVClientSession.recover(clients, "writer", "n1")

    assert session.last_completed_request_id == 1
    pending = session.pending_write()
    assert pending is not None
    assert pending.request == request
    assert session.retry_write("write-1") == request
    safety.checkpoint()
    recovery = [record for record in sim.trace if record.kind == "client-session-recover"]
    assert len(recovery) == 1
    assert recovery[0].details["pending_operation"] == "write-1"


def test_session_recovery_rejects_equal_floor_pending_write_with_conflicting_identity() -> None:
    _, cluster, kv = _applied_single_node_request(Put("x", "durable"))
    clients = KVClientHistory(kv)
    clients.invoke_write("write-1", "writer", 1, Put("x", "pending"))
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()
    before = clients.history.pending()

    with pytest.raises(ClientSessionError, match="conflicts with durable request identity"):
        KVClientSession.recover(clients, "writer", "n1")

    assert clients.history.pending() == before
    assert clients.pending_write("write-1") == ClientRequest(
        "writer", 1, Put("x", "pending")
    )
    safety.checkpoint()
