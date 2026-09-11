import pytest

from distlab.kv import ReplicatedKV
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshot, KVSnapshotStore
from distlab.snapshot_transport import SnapshotTransport


def test_authoritative_leader_cannot_send_non_durable_snapshot() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    kv = ReplicatedKV(cluster)
    store = KVSnapshotStore(cluster, kv)
    transport = SnapshotTransport(store)

    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    assert store.latest("n1") is None

    fabricated = KVSnapshot(
        last_included_index=1,
        last_included_term=leader.current_term,
        state=(("forged", "state"),),
        client_requests=(),
    )
    follower = cluster.node("n3")
    trace_before = tuple(sim.trace)

    with pytest.raises(RuntimeError, match="leader's durable snapshot"):
        transport.send_install_snapshot(
            leader_id="n1",
            follower_id="n3",
            term=leader.current_term,
            snapshot=fabricated,
        )

    sim.run()
    assert follower.log_base_index == 0
    assert store.latest("n3") is None
    assert kv.snapshot("n3") == {}
    assert tuple(sim.trace) == trace_before
    RaftSafetyHarness(cluster).checkpoint()
