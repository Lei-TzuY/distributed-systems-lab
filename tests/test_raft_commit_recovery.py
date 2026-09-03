import pytest

from distlab.commit_recovery import (
    CommitRecoveryBarrier,
    CommitRecoveryError,
    append_current_term_barrier,
)
from distlab.kv import ClientRequest, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def _elect(cluster: RaftCluster, node_id: str):
    node = cluster.node(node_id)
    node.start_election()
    cluster.sim.run()
    assert node.role is RaftRole.LEADER
    return node


def _restart_former_leader_with_committed_old_term_entry():
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = _elect(cluster, "n1")

    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=leader.current_term, command="old-term-entry"),
    )
    first_term = leader.current_term
    initial = LeaderReplicator(leader)
    assert initial.replicate("n2") is True
    assert leader.commit_index == 1

    sim.crash("n1")
    sim.restart("n1")
    assert leader.commit_index == 0
    assert leader.log == (LogEntry(term=first_term, command="old-term-entry"),)

    leader = _elect(cluster, "n1")
    assert leader.current_term == first_term + 1
    assert leader.commit_index == 0
    return sim, cluster, leader


def test_restarted_former_leader_cannot_recommit_old_term_entry_from_replica_count() -> None:
    _, cluster, leader = _restart_former_leader_with_committed_old_term_entry()
    replicator = LeaderReplicator(leader)

    assert replicator.replicate("n2") is True

    assert replicator.progress("n2").match_index == 1
    assert replicator.commit_index == 0
    assert leader.commit_index == 0
    assert cluster.node("n2").commit_index <= 1


def test_current_term_barrier_reestablishes_commit_knowledge_after_restart() -> None:
    sim, cluster, leader = _restart_former_leader_with_committed_old_term_entry()
    replicator = LeaderReplicator(leader)

    assert replicator.replicate("n2") is True
    assert leader.commit_index == 0

    barrier_index = append_current_term_barrier(leader)
    assert barrier_index == 2
    assert leader.log[0].command == "old-term-entry"
    assert leader.log[1].term == leader.current_term
    assert isinstance(leader.log[1].command, CommitRecoveryBarrier)

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == barrier_index
    assert leader.commit_index == barrier_index

    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == barrier_index

    barrier_records = [
        record for record in sim.trace if record.kind == "raft-current-term-barrier"
    ]
    assert len(barrier_records) == 1
    assert barrier_records[0].details["term"] == leader.current_term
    assert barrier_records[0].details["index"] == barrier_index


def test_recovery_barrier_applies_as_kv_noop_after_crash_restart() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = _elect(cluster, "n1")
    request = ClientRequest("client-a", 1, Put("key", "value"))
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=leader.current_term, command=request),
    )

    initial = LeaderReplicator(leader)
    assert initial.replicate("n2") is True
    assert leader.commit_index == 1

    kv = ReplicatedKV(cluster)
    applied = kv.apply_committed("n1")
    assert len(applied) == 1
    assert kv.snapshot("n1") == {"key": "value"}
    assert kv.has_applied_request("n1", "client-a", 1)

    first_term = leader.current_term
    sim.crash("n1")
    sim.restart("n1")
    assert leader.commit_index == 0

    leader = _elect(cluster, "n1")
    assert leader.current_term == first_term + 1
    recovered = ReplicatedKV(cluster)
    assert recovered.snapshot("n1") == {"key": "value"}
    assert recovered.has_applied_request("n1", "client-a", 1)
    assert recovered.applier.last_applied("n1") == 1

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == 0

    barrier_index = append_current_term_barrier(leader)
    assert replicator.replicate("n2") is True
    assert leader.commit_index == barrier_index == 2

    applied = recovered.apply_committed("n1")
    assert len(applied) == 1
    assert applied[0].index == barrier_index
    assert isinstance(applied[0].entry.command, CommitRecoveryBarrier)
    assert recovered.snapshot("n1") == {"key": "value"}
    assert recovered.has_applied_request("n1", "client-a", 1)
    assert recovered.applier.last_applied("n1") == barrier_index

    rebuilt = ReplicatedKV(cluster)
    assert rebuilt.snapshot("n1") == {"key": "value"}
    assert rebuilt.has_applied_request("n1", "client-a", 1)
    assert rebuilt.applier.last_applied("n1") == barrier_index

    barrier_applies = [
        record
        for record in sim.trace
        if record.kind == "kv-apply"
        and record.details["node"] == "n1"
        and record.details["operation"] == "commit-recovery-barrier"
    ]
    assert len(barrier_applies) == 1
    assert barrier_applies[0].details["index"] == barrier_index


def test_current_term_barrier_requires_live_leader_and_preserves_follower_log() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    follower = cluster.node("n2")
    before = follower.log

    with pytest.raises(CommitRecoveryError, match="leader role"):
        append_current_term_barrier(follower)
    assert follower.log == before

    leader = _elect(cluster, "n1")
    sim.crash("n1")
    before = leader.log
    with pytest.raises(CommitRecoveryError, match="live leader"):
        append_current_term_barrier(leader)
    assert leader.log == before
