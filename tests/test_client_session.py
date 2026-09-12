import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import (
    KVClientSession,
    SessionReadPending,
    SessionWritePending,
    StaleClientRequest,
)
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.linearizable_read import LinearizableKVReader, ReadQuorumUnavailable
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _kv_with_log(log: tuple[LogEntry, ...]) -> tuple[Simulator, RaftCluster, ReplicatedKV]:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    return sim, cluster, ReplicatedKV(cluster)


def _ready_session_cluster() -> tuple[
    Simulator,
    RaftCluster,
    ReplicatedKV,
    KVClientSession,
    LeaderReplicator,
    LinearizableKVReader,
]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    RaftSafetyHarness(cluster).checkpoint()

    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    session = KVClientSession(clients, "writer")
    replicator = LeaderReplicator(leader)
    reader = LinearizableKVReader(kv, replicator)
    return sim, cluster, kv, session, replicator, reader


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


def test_session_linearizable_read_preserves_completed_write_program_order() -> None:
    _, cluster, kv, session, replicator, reader = _ready_session_cluster()
    leader = cluster.node("n1")

    request = session.invoke_write("write-1", 1, Put("x", "one"))
    previous = leader.log
    leader.sim.persistent_state[leader.node_id]["log"] = (
        *previous,
        LogEntry(term=leader.current_term, command=request),
    )
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    kv.apply_committed("n1")
    session.complete_write("write-1", "n1")
    RaftSafetyHarness(cluster).checkpoint()

    assert session.linearizable_read("read-1", reader, "x") == "one"
    RaftSafetyHarness(cluster).checkpoint()

    completed = session.clients.history.completed()
    assert [operation.operation_id for operation in completed] == ["write-1", "read-1"]
    assert completed[1].completion.result == "one"
    assert SingleKeyKVLinearizabilityChecker().check(session.clients.history).linearizable is True


def test_session_rejects_linearizable_read_before_pending_write_history_mutation() -> None:
    sim, cluster, _, session, _, reader = _ready_session_cluster()
    session.invoke_write("write-1", 1, Put("x", "one"))
    before = session.clients.history.invocations()

    with pytest.raises(SessionWritePending, match="cannot read while write"):
        session.linearizable_read("read-1", reader, "x")

    assert session.clients.history.invocations() == before
    assert [record for record in sim.trace if record.kind == "raft-linearizable-read"] == []
    RaftSafetyHarness(cluster).checkpoint()


def test_failed_linearizable_read_remains_single_flight_until_exact_retry() -> None:
    sim, cluster, kv, session, replicator, reader = _ready_session_cluster()
    leader = cluster.node("n1")

    request = session.invoke_write("write-1", 1, Put("x", "one"))
    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=request),
    )
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    kv.apply_committed("n1")
    session.complete_write("write-1", "n1")
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    sim.crash("n2")
    sim.crash("n3")
    with pytest.raises(ReadQuorumUnavailable):
        session.linearizable_read("read-1", reader, "x")

    pending = session.pending_read()
    assert pending is not None
    assert pending.operation_id == "read-1"
    assert pending.key == "x"
    assert [item.operation_id for item in session.clients.history.pending()] == ["read-1"]
    with pytest.raises(SessionReadPending, match="cannot write while read"):
        session.invoke_write("write-2", 2, Put("x", "two"))
    with pytest.raises(SessionReadPending, match="already has pending read"):
        session.linearizable_read("read-2", reader, "x")

    sim.restart("n2")
    assert session.retry_linearizable_read("read-1", reader) == "one"
    assert session.pending_read() is None
    assert session.clients.history.pending() == ()
    read_invocations = [
        item for item in session.clients.history.invocations() if item.operation_id == "read-1"
    ]
    assert len(read_invocations) == 1
    retries = [
        record
        for record in sim.trace
        if record.kind == "client-retry" and record.details["operation_id"] == "read-1"
    ]
    assert len(retries) == 1
    safety.checkpoint()
    assert SingleKeyKVLinearizabilityChecker().check(session.clients.history).linearizable is True


def test_session_recovery_restores_pending_write_single_flight_fence() -> None:
    sim, cluster, _, session, _, reader = _ready_session_cluster()
    request = session.invoke_write("write-1", 1, Put("x", "one"))

    recovered = KVClientSession.recover(session.clients, "writer", "n1")

    pending = recovered.pending_write()
    assert pending is not None
    assert pending.operation_id == "write-1"
    assert pending.request == request
    assert recovered.retry_write("write-1") == request
    with pytest.raises(SessionWritePending):
        recovered.invoke_write("write-2", 2, Put("x", "two"))
    with pytest.raises(SessionWritePending):
        recovered.linearizable_read("read-1", reader, "x")
    evidence = [record for record in sim.trace if record.kind == "client-session-recover-pending"]
    assert len(evidence) == 1
    assert evidence[0].details["operation_id"] == "write-1"
    assert evidence[0].details["operation"] == "write"
    RaftSafetyHarness(cluster).checkpoint()


def test_session_recovery_restores_failed_read_for_exact_retry() -> None:
    sim, cluster, kv, session, replicator, reader = _ready_session_cluster()
    leader = cluster.node("n1")
    request = session.invoke_write("write-1", 1, Put("x", "one"))
    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=request),
    )
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    kv.apply_committed("n1")
    session.complete_write("write-1", "n1")
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    sim.crash("n2")
    sim.crash("n3")
    with pytest.raises(ReadQuorumUnavailable):
        session.linearizable_read("read-1", reader, "x")

    recovered = KVClientSession.recover(session.clients, "writer", "n1")
    pending = recovered.pending_read()
    assert pending is not None
    assert pending.operation_id == "read-1"
    assert pending.key == "x"
    with pytest.raises(SessionReadPending):
        recovered.invoke_write("write-2", 2, Put("x", "two"))

    sim.restart("n2")
    assert recovered.retry_linearizable_read("read-1", reader) == "one"
    assert recovered.pending_read() is None
    assert recovered.clients.history.pending() == ()
    invocations = [
        item for item in recovered.clients.history.invocations() if item.operation_id == "read-1"
    ]
    assert len(invocations) == 1
    evidence = [record for record in sim.trace if record.kind == "client-session-recover-pending"]
    assert len(evidence) == 1
    assert evidence[0].details["operation_id"] == "read-1"
    assert evidence[0].details["operation"] == "read"
    safety.checkpoint()
    assert SingleKeyKVLinearizabilityChecker().check(recovered.clients.history).linearizable is True
