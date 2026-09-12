from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession
from distlab.kv import Put, ReplicatedKV
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def test_rebuilt_client_history_preserves_pending_write_for_exact_retry() -> None:
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

    request = session.invoke_write("write-7", 7, Put("x", "seven"))
    rebuilt_clients = KVClientHistory(kv, clients.history)
    assert rebuilt_clients.pending_write("write-7") == request

    recovered = KVClientSession.recover(rebuilt_clients, "writer", "n1")
    pending = recovered.pending_write()
    assert pending is not None
    assert pending.operation_id == "write-7"
    assert pending.request == request
    assert recovered.retry_write("write-7") == request

    sim.persistent_state[leader.node_id]["log"] = (
        *leader.log,
        LogEntry(term=leader.current_term, command=request),
    )
    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    kv.apply_committed("n1")
    recovered.complete_write("write-7", "n1")

    assert recovered.pending_write() is None
    assert recovered.last_completed_request_id == 7
    assert rebuilt_clients.history.pending() == ()

    completed_clients = KVClientHistory(kv, rebuilt_clients.history)
    assert completed_clients.pending_write("write-7") is None
    completed_session = KVClientSession.recover(completed_clients, "writer", "n1")
    assert completed_session.pending_write() is None
    assert completed_session.last_completed_request_id == 7
    assert completed_clients.history.client_request_id("write-7") is None

    RaftSafetyHarness(cluster).checkpoint()
    assert SingleKeyKVLinearizabilityChecker().check(completed_clients.history).linearizable is True