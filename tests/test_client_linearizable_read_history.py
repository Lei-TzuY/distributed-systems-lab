import pytest

from distlab.client_history import KVClientHistory
from distlab.commit_recovery import append_current_term_barrier
from distlab.kv import ReplicatedKV
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.linearizable_read import LinearizableKVReader, ReadQuorumUnavailable
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _ready_linearizable_reader() -> tuple[
    Simulator,
    RaftCluster,
    ReplicatedKV,
    KVClientHistory,
    LinearizableKVReader,
    RaftSafetyHarness,
]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    append_current_term_barrier(leader)
    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 1
    harness.checkpoint()

    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    reader = LinearizableKVReader(kv, replicator)
    return sim, cluster, kv, clients, reader, harness


def test_client_history_records_successful_linearizable_read() -> None:
    sim, cluster, _, clients, reader, _ = _ready_linearizable_reader()

    assert clients.linearizable_read("read-1", "client-a", reader, "k") is None
    RaftSafetyHarness(cluster).checkpoint()

    completed = clients.history.completed()
    assert len(completed) == 1
    assert completed[0].operation_id == "read-1"
    assert completed[0].completion.result is None
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True

    invocations = [record for record in sim.trace if record.kind == "client-invoke"]
    responses = [record for record in sim.trace if record.kind == "client-response"]
    protocol_reads = [record for record in sim.trace if record.kind == "raft-linearizable-read"]
    assert invocations[-1].details["consistency"] == "linearizable"
    assert responses[-1].details["consistency"] == "linearizable"
    assert len(protocol_reads) == 1


def test_failed_linearizable_read_stays_pending_in_client_history() -> None:
    sim, _, _, clients, reader, harness = _ready_linearizable_reader()
    sim.crash("n2")
    sim.crash("n3")
    harness.checkpoint()

    with pytest.raises(ReadQuorumUnavailable):
        clients.linearizable_read("read-1", "client-a", reader, "k")

    harness.checkpoint()
    assert clients.history.completed() == ()
    pending = clients.history.pending()
    assert len(pending) == 1
    assert pending[0].operation_id == "read-1"
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True

    responses = [record for record in sim.trace if record.kind == "client-response"]
    assert responses == []


def test_linearizable_read_rejects_foreign_kv_before_history_mutation() -> None:
    sim, _, _, clients, _, harness = _ready_linearizable_reader()

    other_cluster = RaftCluster(Simulator(), ("m1", "m2", "m3"))
    other_leader = other_cluster.node("m1")
    other_leader.start_election()
    other_cluster.sim.run()
    other_kv = ReplicatedKV(other_cluster)
    foreign_reader = LinearizableKVReader(other_kv, LeaderReplicator(other_leader))

    with pytest.raises(ValueError, match="same replicated KV"):
        clients.linearizable_read("read-1", "client-a", foreign_reader, "k")

    harness.checkpoint()
    assert clients.history.invocations() == ()
    assert [record for record in sim.trace if record.kind == "client-invoke"] == []
