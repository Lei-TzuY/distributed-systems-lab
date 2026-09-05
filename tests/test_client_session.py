import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession, SessionWritePending, StaleClientRequest
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster
from distlab.simulator import Simulator


def _kv_with_log(log: tuple[LogEntry, ...]) -> tuple[Simulator, RaftCluster, ReplicatedKV]:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    return sim, cluster, ReplicatedKV(cluster)


def test_session_allows_exact_retry_but_rejects_concurrent_and_stale_requests() -> None:
    request = ClientRequest("writer", 7, Put("x", "one"))
    sim, cluster, kv = _kv_with_log((LogEntry(term=1, command=request),))
    clients = KVClientHistory(kv)
    session = KVClientSession(clients, "writer")

    submitted = session.invoke_write("write-7", 7, Put("x", "one"))
    assert session.retry_write("write-7") == submitted
    with pytest.raises(SessionWritePending):
        session.invoke_write("write-8", 8, Put("x", "two"))

    cluster.node("n1").advance_commit_index(1, source="test")
    kv.apply_committed("n1")
    session.complete_write("write-7", "n1")

    assert session.last_completed_request_id == 7
    assert session.pending_write() is None
    with pytest.raises(StaleClientRequest):
        session.invoke_write("stale", 7, Put("x", "stale"))

    assert [record.kind for record in sim.trace if record.kind == "client-session-advance"] == [
        "client-session-advance"
    ]


def test_session_recovers_sequence_floor_from_applied_replicated_dedup_state() -> None:
    log = (
        LogEntry(term=1, command=ClientRequest("writer", 2, Put("x", "two"))),
        LogEntry(term=1, command=ClientRequest("other", 99, Put("y", "other"))),
        LogEntry(term=1, command=ClientRequest("writer", 5, Put("x", "five"))),
    )
    sim, cluster, kv = _kv_with_log(log)
    cluster.node("n1").advance_commit_index(3, source="test")
    kv.apply_committed("n1")
    clients = KVClientHistory(kv)

    session = KVClientSession.recover(clients, "writer", "n1")

    assert session.last_completed_request_id == 5
    with pytest.raises(StaleClientRequest):
        session.invoke_write("stale", 4, Put("x", "four"))
    fresh = session.invoke_write("fresh", 6, Put("x", "six"))
    assert fresh == ClientRequest("writer", 6, Put("x", "six"))
    assert session.pending_write() is not None
    recovery = [record for record in sim.trace if record.kind == "client-session-recover"]
    assert len(recovery) == 1
    assert recovery[0].details["last_completed_request_id"] == 5
