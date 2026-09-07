import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession, StaleClientRequest
from distlab.kv import ClientRequest, ClientRequestConflict, Put, ReplicatedKV
from distlab.linearizable_read import LinearizableKVReader, ReadQuorumUnavailable
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _cluster_with_stale_follower() -> tuple[
    Simulator,
    RaftCluster,
    ReplicatedKV,
    KVClientHistory,
    LinearizableKVReader,
]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    request = ClientRequest("writer", 5, Put("x", "five"))
    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=request),
    )
    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1

    kv = ReplicatedKV(cluster)
    kv.apply_committed("n1")
    clients = KVClientHistory(kv)
    reader = LinearizableKVReader(kv, replicator)
    RaftSafetyHarness(cluster).checkpoint()
    return sim, cluster, kv, clients, reader


def test_linearizable_session_recovery_uses_quorum_confirmed_leader_dedup_state() -> None:
    sim, cluster, _, clients, reader = _cluster_with_stale_follower()

    stale = KVClientSession.recover(clients, "writer", "n3")
    assert stale.last_completed_request_id == -1

    recovered = KVClientSession.recover_linearizable(clients, "writer", reader)

    assert recovered.last_completed_request_id == 5
    with pytest.raises(StaleClientRequest):
        recovered.invoke_write("stale", 4, Put("x", "four"))
    assert [
        record for record in sim.trace if record.kind == "raft-linearizable-read-barrier"
    ]
    linearized = [
        record for record in sim.trace if record.kind == "client-session-recover-linearizable"
    ]
    assert len(linearized) == 1
    assert linearized[0].details["node"] == "n1"
    assert linearized[0].details["last_completed_request_id"] == 5
    RaftSafetyHarness(cluster).checkpoint()


def test_stale_session_cannot_complete_conflicting_committed_request_identity() -> None:
    sim, cluster, kv, clients, _ = _cluster_with_stale_follower()
    stale = KVClientSession.recover(clients, "writer", "n3")
    assert stale.last_completed_request_id == -1

    request = stale.invoke_write("conflict", 5, Put("x", "replacement"))

    with pytest.raises(ClientRequestConflict):
        stale.complete_write("conflict", "n1")

    assert stale.pending_write() is not None
    assert stale.pending_write().request == request
    assert clients.pending_write("conflict") == request
    assert kv.get("n1", "x") == "five"
    assert [
        record
        for record in sim.trace
        if record.kind == "client-response" and record.details["operation_id"] == "conflict"
    ] == []
    RaftSafetyHarness(cluster).checkpoint()


def test_linearizable_session_recovery_fails_without_quorum_before_recovery_state() -> None:
    sim, cluster, _, clients, reader = _cluster_with_stale_follower()
    sim.crash("n2")
    sim.crash("n3")
    before = [record for record in sim.trace if record.kind == "client-session-recover"]

    with pytest.raises(ReadQuorumUnavailable):
        KVClientSession.recover_linearizable(clients, "writer", reader)

    after = [record for record in sim.trace if record.kind == "client-session-recover"]
    assert after == before
    assert [
        record for record in sim.trace if record.kind == "client-session-recover-linearizable"
    ] == []
    RaftSafetyHarness(cluster).checkpoint()
