from distlab.kv import ReplicatedKV
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshot, KVSnapshotStore
from distlab.snapshot_transport import (
    InstallSnapshotRequest,
    InstallSnapshotResponse,
    SnapshotTransport,
)


def _snapshot() -> KVSnapshot:
    return KVSnapshot(
        last_included_index=1,
        last_included_term=1,
        state=(("k", "v"),),
        client_requests=(),
    )


def _transport(sim: Simulator, cluster: RaftCluster) -> SnapshotTransport:
    kv = ReplicatedKV(cluster)
    return SnapshotTransport(KVSnapshotStore(cluster, kv))


def _rejections(sim: Simulator) -> list[object]:
    return [record for record in sim.trace if record.kind == "raft-snapshot-envelope-rejected"]


def test_unknown_snapshot_leader_cannot_advance_follower_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    transport = _transport(sim, cluster)
    follower = cluster.node("n3")

    sim.send(
        "intruder",
        "n3",
        InstallSnapshotRequest(
            term=9,
            leader_id="intruder",
            follower_id="n3",
            snapshot=_snapshot(),
        ),
        delivery_dst=transport.endpoint("n3"),
    )
    sim.run()

    assert follower.current_term == 0
    assert follower.log_base_index == 0
    rejected = _rejections(sim)
    assert len(rejected) == 1
    assert rejected[0].details["reason"] == "unknown-source-principal"
    RaftSafetyHarness(cluster).checkpoint()


def test_unknown_snapshot_follower_cannot_advance_leader_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    transport = _transport(sim, cluster)
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    term_before = leader.current_term

    sim.send(
        "intruder",
        "n1",
        InstallSnapshotResponse(
            term=term_before + 8,
            leader_id="n1",
            follower_id="intruder",
            success=False,
            last_included_index=0,
            requested_last_included_index=1,
        ),
        delivery_dst=transport.endpoint("n1"),
    )
    sim.run()

    assert leader.current_term == term_before
    assert leader.role is RaftRole.LEADER
    rejected = _rejections(sim)
    assert len(rejected) == 1
    assert rejected[0].details["reason"] == "unknown-source-principal"
    RaftSafetyHarness(cluster).checkpoint()
