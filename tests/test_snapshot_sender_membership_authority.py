import pytest

from distlab.kv import ReplicatedKV
from distlab.membership import ReconfigurableRaftCluster, VotingConfiguration
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshot, KVSnapshotStore
from distlab.snapshot_transport import SnapshotTransport


def test_removed_leader_cannot_create_fresh_install_snapshot_traffic() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3"),
        voters=("n1", "n2", "n3"),
    )
    kv = ReplicatedKV(cluster)
    store = KVSnapshotStore(cluster, kv)
    transport = SnapshotTransport(store)
    safety = RaftSafetyHarness(cluster)

    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    term = leader.current_term
    safety.checkpoint()

    cluster._install_voting_configuration(
        VotingConfiguration(frozenset({"n2", "n3"})),
        reason="test-remove-current-leader",
    )
    assert leader.role is RaftRole.LEADER
    assert not cluster.is_voter("n1")

    snapshot = KVSnapshot(
        last_included_index=1,
        last_included_term=term,
        state=(("k", "v"),),
        client_requests=(),
    )
    follower = cluster.node("n2")
    trace_before = tuple(sim.trace)

    with pytest.raises(RuntimeError, match="current voter authority"):
        transport.send_install_snapshot(
            leader_id="n1",
            follower_id="n2",
            term=term,
            snapshot=snapshot,
            request_id=23,
        )

    sim.run()
    assert tuple(sim.trace) == trace_before
    assert follower.current_term == term
    assert follower.log_base_index == 0
    assert store.latest("n2") is None
    assert kv.snapshot("n2") == {}
    safety.checkpoint()
