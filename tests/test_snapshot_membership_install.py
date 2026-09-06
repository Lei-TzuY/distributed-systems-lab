import pytest

from distlab.kv import ReplicatedKV
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshot, KVSnapshotStore, SnapshotVotingConfiguration
from distlab.snapshot_transport import SnapshotTransport


def _snapshot(
    *,
    boundary: int,
    membership_index: int,
    voters: tuple[str, ...],
) -> KVSnapshot:
    return KVSnapshot(
        last_included_index=boundary,
        last_included_term=1,
        state=(),
        client_requests=(),
        voting_configuration=SnapshotVotingConfiguration(
            committed_index=membership_index,
            old_voters=voters,
        ),
    )


def test_install_rejects_membership_index_behind_durable_watermark() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1",), voters=("n1",))
    store = KVSnapshotStore(cluster, ReplicatedKV(cluster))
    sim.persistent_state["n1"]["membership_commit_index"] = 2

    with pytest.raises(ValueError, match="behind durable local evidence"):
        store.install(
            "n1",
            _snapshot(boundary=3, membership_index=1, voters=("n1",)),
        )

    assert "kv_snapshot" not in sim.persistent_state["n1"]
    assert sim.persistent_state["n1"]["membership_commit_index"] == 2
    RaftSafetyHarness(cluster).checkpoint()


def test_install_rejects_conflicting_configuration_at_same_membership_index() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1", "n2"), voters=("n1",))
    store = KVSnapshotStore(cluster, ReplicatedKV(cluster))
    previous = _snapshot(boundary=1, membership_index=1, voters=("n1",))
    sim.persistent_state["n1"]["kv_snapshot"] = previous
    sim.persistent_state["n1"]["membership_commit_index"] = 1

    with pytest.raises(ValueError, match="conflicts with durable configuration"):
        store.install(
            "n1",
            _snapshot(boundary=2, membership_index=1, voters=("n2",)),
        )

    assert sim.persistent_state["n1"]["kv_snapshot"] == previous
    assert sim.persistent_state["n1"]["membership_commit_index"] == 1
    RaftSafetyHarness(cluster).checkpoint()


def test_install_advances_membership_evidence_monotonically() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1",), voters=("n1",))
    kv = ReplicatedKV(cluster)
    store = KVSnapshotStore(cluster, kv)
    snapshot = _snapshot(boundary=2, membership_index=2, voters=("n1",))

    store.install("n1", snapshot)

    assert store.latest("n1") == snapshot
    assert sim.persistent_state["n1"]["membership_commit_index"] == 2
    assert cluster.node("n1").log_base_index == 2
    assert kv.applier.last_applied("n1") == 2
    RaftSafetyHarness(cluster).checkpoint()


def test_transport_returns_failure_for_rejected_membership_snapshot() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, ("n1", "n2"), voters=("n1", "n2"))
    store = KVSnapshotStore(cluster, ReplicatedKV(cluster))
    transport = SnapshotTransport(store)
    previous = _snapshot(boundary=2, membership_index=2, voters=("n1", "n2"))
    persistent = sim.persistent_state["n2"]
    persistent["kv_snapshot"] = previous
    persistent["membership_commit_index"] = 2
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    transport.send_install_snapshot(
        leader_id="n1",
        follower_id="n2",
        term=1,
        snapshot=_snapshot(boundary=3, membership_index=1, voters=("n1", "n2")),
        request_id=7,
    )
    sim.run()

    assert store.latest("n2") == previous
    assert persistent["membership_commit_index"] == 2
    assert cluster.node("n2").log_base_index == 0
    rejected = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-rejected"
    ]
    responses = [
        record for record in sim.trace if record.kind == "raft-install-snapshot-response"
    ]
    assert len(rejected) == 1
    assert rejected[0].details["request_id"] == 7
    assert "behind durable local evidence" in rejected[0].details["reason"]
    assert responses[-1].details["success"] is False
    assert responses[-1].details["last_included_index"] == 0
    assert responses[-1].details["requested_last_included_index"] == 3
    assert responses[-1].details["request_id"] == 7
    safety.checkpoint()
