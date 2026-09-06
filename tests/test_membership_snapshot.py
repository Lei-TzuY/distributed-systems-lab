import pytest

from distlab.kv import ReplicatedKV
from distlab.membership import NonVoterElectionError, ReconfigurableRaftCluster
from distlab.membership_log import ReplicatedMembershipTransition
from distlab.raft import RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator
from distlab.snapshot import KVSnapshotStore

NODE_IDS = ("n1", "n2", "n3", "n4", "n5")
BOOTSTRAP_VOTERS = ("n1", "n2", "n3")


def _cluster() -> tuple[Simulator, ReconfigurableRaftCluster]:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(sim, NODE_IDS, voters=BOOTSTRAP_VOTERS)
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    return sim, cluster


def _recreate(sim: Simulator) -> tuple[Simulator, ReconfigurableRaftCluster]:
    recovered_sim = Simulator()
    for node_id in NODE_IDS:
        recovered_sim.persistent_state[node_id].update(sim.persistent_state[node_id])
    recovered = ReconfigurableRaftCluster(
        recovered_sim,
        NODE_IDS,
        voters=BOOTSTRAP_VOTERS,
    )
    return recovered_sim, recovered


def _compact_leader(cluster: ReconfigurableRaftCluster) -> None:
    kv = ReplicatedKV(cluster)
    applied = kv.applier.apply_committed("n1")
    assert applied
    snapshot = KVSnapshotStore(cluster, kv).compact("n1")
    assert snapshot.last_included_index == cluster.node("n1").log_base_index
    assert snapshot.voting_configuration is not None
    cluster.assert_log_matching()
    kv.applier.assert_state_machine_safety()


def test_recreation_recovers_joint_membership_from_compacted_snapshot() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    transition = ReplicatedMembershipTransition(cluster.node("n1"))

    joint_index = transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)
    harness.checkpoint()

    _compact_leader(cluster)
    snapshot = sim.persistent_state["n1"]["kv_snapshot"]
    assert snapshot.voting_configuration is not None
    assert snapshot.voting_configuration.committed_index == joint_index
    assert snapshot.voting_configuration.old_voters == ("n1", "n2", "n3")
    assert snapshot.voting_configuration.new_voters == ("n1", "n4", "n5")
    assert cluster.node("n1").log_base_index == joint_index

    recovered_sim, recovered = _recreate(sim)

    assert recovered.voting_configuration.old_voters == frozenset({"n1", "n2", "n3"})
    assert recovered.voting_configuration.new_voters == frozenset({"n1", "n4", "n5"})
    assert any(
        record.kind == "raft-membership-recovery-source"
        and record.details["source"] == "n1"
        and record.details["log_base_index"] == joint_index
        and record.details["snapshot_membership_index"] == joint_index
        for record in recovered_sim.trace
    )
    RaftSafetyHarness(recovered).checkpoint()


def test_compacted_stable_snapshot_ignores_uncommitted_later_membership() -> None:
    sim, cluster = _cluster()
    harness = RaftSafetyHarness(cluster)
    leader = cluster.node("n1")
    transition = ReplicatedMembershipTransition(leader)

    transition.propose_joint(("n1", "n4", "n5"))
    assert transition.replicate_and_activate(max_attempts_per_peer=2)
    final_index = transition.propose_finalize()
    assert transition.replicate_and_finalize(max_attempts_per_peer=3)
    assert leader.commit_index == final_index
    harness.checkpoint()

    _compact_leader(cluster)
    snapshot = sim.persistent_state["n1"]["kv_snapshot"]
    assert snapshot.voting_configuration is not None
    assert snapshot.voting_configuration.committed_index == final_index
    assert snapshot.voting_configuration.old_voters == ("n1", "n4", "n5")
    assert snapshot.voting_configuration.new_voters is None

    later = ReplicatedMembershipTransition(leader)
    uncommitted_index = later.propose_joint(("n1", "n2", "n5"))
    assert uncommitted_index == final_index + 1
    assert leader.commit_index == final_index
    cluster.assert_log_matching()

    recovered_sim, recovered = _recreate(sim)

    assert recovered.voting_configuration.old_voters == frozenset({"n1", "n4", "n5"})
    assert recovered.voting_configuration.new_voters is None
    assert not recovered.is_voter("n2")
    assert any(
        record.kind == "raft-membership-recovered"
        and record.details["committed_index"] == final_index
        and record.details["old_voters"] == ("n1", "n4", "n5")
        and record.details["new_voters"] is None
        for record in recovered_sim.trace
    )
    with pytest.raises(NonVoterElectionError, match="non-voter"):
        recovered.node("n2").start_election()
    RaftSafetyHarness(recovered).checkpoint()
