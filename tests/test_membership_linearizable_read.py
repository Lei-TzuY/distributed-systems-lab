import pytest

from distlab.commit_recovery import append_current_term_barrier
from distlab.kv import Put, ReplicatedKV
from distlab.linearizable_read import LinearizableKVReader, ReadQuorumUnavailable
from distlab.membership import ReconfigurableRaftCluster
from distlab.membership_replication import MembershipAwareLeaderReplicator
from distlab.raft import LogEntry, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _reader_cluster() -> tuple[
    Simulator,
    ReconfigurableRaftCluster,
    MembershipAwareLeaderReplicator,
    ReplicatedKV,
]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = (LogEntry(term=2, command=Put("k", "v1")),)
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.current_term == 3
    assert leader.role is RaftRole.LEADER
    replicator = MembershipAwareLeaderReplicator(leader)
    append_current_term_barrier(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 2
    return sim, cluster, replicator, ReplicatedKV(cluster)


def test_learners_do_not_inflate_stable_linearizable_read_quorum() -> None:
    sim, cluster, replicator, kv = _reader_cluster()
    sim.crash("n3")
    sim.crash("n4")
    sim.crash("n5")
    RaftSafetyHarness(cluster).checkpoint()

    reader = LinearizableKVReader(kv, replicator)
    assert reader.get("k") == "v1"

    reads = [record for record in sim.trace if record.kind == "raft-linearizable-read"]
    assert reads[-1].details["acknowledged_voters"] == ("n1", "n2")
    assert reads[-1].details["quorum_mode"] == "stable"
    RaftSafetyHarness(cluster).checkpoint()


def test_joint_linearizable_read_requires_both_voter_majorities() -> None:
    sim, cluster, replicator, kv = _reader_cluster()
    cluster.begin_joint_consensus("n1", ("n1", "n4", "n5"))
    sim.crash("n4")
    sim.crash("n5")
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    reader = LinearizableKVReader(kv, replicator)
    with pytest.raises(ReadQuorumUnavailable):
        reader.get("k", max_attempts_per_peer=3)

    failures = [
        record for record in sim.trace if record.kind == "raft-linearizable-read-quorum-failed"
    ]
    assert failures[-1].details["acknowledged_voters"] == ("n1", "n2", "n3")
    assert failures[-1].details["quorum_mode"] == "joint"
    safety.checkpoint()


def test_read_barrier_revalidates_membership_after_peer_acknowledgement() -> None:
    sim, cluster, _, kv = _reader_cluster()
    sim.crash("n4")
    sim.crash("n5")
    safety = RaftSafetyHarness(cluster)
    safety.checkpoint()

    class ReconfiguringReplicator(MembershipAwareLeaderReplicator):
        def __init__(self) -> None:
            super().__init__(cluster.node("n1"))
            self.transitioned = False

        def replicate(self, peer: str, *, max_attempts: int | None = None) -> bool:
            replicated = super().replicate(peer, max_attempts=max_attempts)
            if peer == "n2" and replicated and not self.transitioned:
                cluster.begin_joint_consensus("n1", ("n1", "n4", "n5"))
                self.transitioned = True
            return replicated

    reconfiguring_replicator = ReconfiguringReplicator()
    reader = LinearizableKVReader(kv, reconfiguring_replicator)

    with pytest.raises(ReadQuorumUnavailable):
        reader.get("k", max_attempts_per_peer=1)

    assert reconfiguring_replicator.transitioned is True
    failures = [
        record for record in sim.trace if record.kind == "raft-linearizable-read-quorum-failed"
    ]
    assert failures[-1].details["acknowledged_voters"] == ("n1", "n2")
    assert failures[-1].details["quorum_mode"] == "joint"
    assert not [record for record in sim.trace if record.kind == "raft-linearizable-read"]
    safety.checkpoint()
