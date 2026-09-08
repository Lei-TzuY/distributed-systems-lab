import pytest

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


def _transport(
    sim: Simulator,
    cluster: RaftCluster,
) -> tuple[ReplicatedKV, KVSnapshotStore, SnapshotTransport]:
    kv = ReplicatedKV(cluster)
    store = KVSnapshotStore(cluster, kv)
    return kv, store, SnapshotTransport(store)


def _rejections(sim: Simulator) -> list[object]:
    return [
        record
        for record in sim.trace
        if record.kind == "raft-snapshot-envelope-rejected"
    ]


def test_forged_install_snapshot_request_cannot_advance_follower_term_or_install() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    kv, store, transport = _transport(sim, cluster)
    follower = cluster.node("n3")

    sim.send(
        "n2",
        "n3",
        InstallSnapshotRequest(
            term=9,
            leader_id="n1",
            follower_id="n3",
            snapshot=_snapshot(),
        ),
        delivery_dst=transport.endpoint("n3"),
    )
    sim.run()

    assert follower.current_term == 0
    assert follower.log_base_index == 0
    assert store.latest("n3") is None
    assert kv.snapshot("n3") == {}
    rejected = _rejections(sim)
    assert len(rejected) == 1
    assert rejected[0].details["reason"] == "source-identity-mismatch"
    assert rejected[0].details["expected_src"] == "n1"
    RaftSafetyHarness(cluster).checkpoint()


def test_forged_install_snapshot_response_cannot_advance_leader_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    _, _, transport = _transport(sim, cluster)
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    term_before = leader.current_term

    sim.send(
        "n2",
        "n1",
        InstallSnapshotResponse(
            term=term_before + 8,
            leader_id="n1",
            follower_id="n3",
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
    assert rejected[0].details["reason"] == "source-identity-mismatch"
    assert rejected[0].details["expected_src"] == "n3"
    RaftSafetyHarness(cluster).checkpoint()


def test_misrouted_snapshot_delivery_endpoint_is_rejected_before_install() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    _, store, transport = _transport(sim, cluster)
    follower = cluster.node("n3")

    sim.send(
        "n1",
        "n3",
        InstallSnapshotRequest(
            term=7,
            leader_id="n1",
            follower_id="n3",
            snapshot=_snapshot(),
        ),
        delivery_dst=transport.endpoint("n2"),
    )
    sim.run()

    assert follower.current_term == 0
    assert follower.log_base_index == 0
    assert store.latest("n3") is None
    rejected = _rejections(sim)
    assert len(rejected) == 1
    assert rejected[0].details["reason"] == "delivery-endpoint-mismatch"
    assert rejected[0].details["expected_delivery_dst"] == transport.endpoint("n3")
    RaftSafetyHarness(cluster).checkpoint()


def test_snapshot_message_types_reject_self_directed_endpoints() -> None:
    with pytest.raises(ValueError, match="snapshot endpoints must be distinct"):
        InstallSnapshotRequest(
            term=1,
            leader_id="n1",
            follower_id="n1",
            snapshot=_snapshot(),
        )
    with pytest.raises(ValueError, match="snapshot endpoints must be distinct"):
        InstallSnapshotResponse(
            term=1,
            leader_id="n1",
            follower_id="n1",
            success=True,
            last_included_index=1,
            requested_last_included_index=1,
        )


def test_snapshot_transport_rejects_self_target_before_state_or_trace_mutation() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    kv, store, transport = _transport(sim, cluster)
    node = cluster.node("n1")
    term_before = node.current_term
    trace_before = tuple(sim.trace)

    with pytest.raises(ValueError, match="snapshot transport requires distinct Raft nodes"):
        transport.send_install_snapshot(
            leader_id="n1",
            follower_id="n1",
            term=term_before,
            snapshot=_snapshot(),
        )

    assert node.current_term == term_before
    assert node.log_base_index == 0
    assert store.latest("n1") is None
    assert kv.snapshot("n1") == {}
    assert tuple(sim.trace) == trace_before
    RaftSafetyHarness(cluster).checkpoint()
