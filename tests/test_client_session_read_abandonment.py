import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession
from distlab.kv import Put, ReplicatedKV
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.linearizable_read import LinearizableKVReader, ReadQuorumUnavailable
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def test_abandoned_failed_read_releases_session_without_recovery_resurrection() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    session = KVClientSession(clients, "writer")
    replicator = LeaderReplicator(leader)
    reader = LinearizableKVReader(kv, replicator)

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
    assert session.pending_read() is not None

    session.abandon_linearizable_read("read-1")

    assert session.pending_read() is None
    assert clients.is_abandoned_read("read-1") is True
    assert [item.operation_id for item in clients.history.pending()] == ["read-1"]
    with pytest.raises(ValueError, match="abandoned linearizable read"):
        clients.retry_linearizable_read("read-1", "writer", reader, "x")

    request2 = session.invoke_write("write-2", 2, Put("x", "two"))
    assert request2.request_id == 2
    assert session.pending_write() is not None

    recovered = KVClientSession.recover(clients, "writer", "n1")
    assert recovered.pending_read() is None
    assert recovered.pending_write() is not None
    assert recovered.pending_write().operation_id == "write-2"
    assert recovered.retry_write("write-2") == request2

    abandon_events = [
        record
        for record in sim.trace
        if record.kind == "client-abandon" and record.details["operation_id"] == "read-1"
    ]
    assert len(abandon_events) == 1
    safety.checkpoint()
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True
