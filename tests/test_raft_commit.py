import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator, ReplicationError
from distlab.simulator import Simulator


def _elect_leader_with_log(
    log: tuple[LogEntry, ...], *, node_ids: tuple[str, ...] = ("n1", "n2", "n3")
) -> tuple[Simulator, RaftCluster, LeaderReplicator]:
    sim = Simulator()
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, node_ids)
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.current_term == 3
    assert leader.role is RaftRole.LEADER
    return sim, cluster, LeaderReplicator(leader)


def test_majority_replication_commits_current_term_entry() -> None:
    log = (
        LogEntry(term=2, command="old"),
        LogEntry(term=3, command="new"),
    )
    sim, _, replicator = _elect_leader_with_log(log)

    assert replicator.commit_index == 0
    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 2

    commits = [record for record in sim.trace if record.kind == "raft-commit-advance"]
    assert len(commits) == 1
    assert commits[0].details["previous_commit_index"] == 0
    assert commits[0].details["commit_index"] == 2
    assert commits[0].details["replicas"] == 2
    assert commits[0].details["majority"] == 2


def test_commit_waits_for_majority_in_five_node_cluster() -> None:
    log = (LogEntry(term=3, command="current"),)
    _, _, replicator = _elect_leader_with_log(
        log, node_ids=("n1", "n2", "n3", "n4", "n5")
    )

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 0

    assert replicator.replicate("n3") is True
    assert replicator.commit_index == 1


def test_old_term_entry_is_not_committed_by_replica_counting_alone() -> None:
    old_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
    )
    _, _, replicator = _elect_leader_with_log(old_log)

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 0
    assert replicator.advance_commit_index() == 0


def test_current_term_commit_implicitly_commits_prior_term_prefix() -> None:
    old_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
    )
    sim, cluster, replicator = _elect_leader_with_log(old_log)

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 0

    leader = cluster.node("n1")
    sim.persistent_state["n1"]["log"] = old_log + (
        LogEntry(term=leader.current_term, command="barrier"),
    )

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 3
    assert cluster.node("n2").log == leader.log


def test_commit_index_is_monotonic_and_uses_highest_eligible_index() -> None:
    log = (
        LogEntry(term=2, command="old"),
        LogEntry(term=3, command="new-1"),
        LogEntry(term=3, command="new-2"),
    )
    _, _, replicator = _elect_leader_with_log(log)

    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 3
    assert replicator.advance_commit_index() == 3
    assert replicator.replicate("n3") is True
    assert replicator.commit_index == 3


def test_stale_replicator_cannot_advance_commit_index() -> None:
    log = (LogEntry(term=3, command="current"),)
    _, cluster, replicator = _elect_leader_with_log(log)

    leader = cluster.node("n1")
    leader.start_election()
    assert leader.current_term == 4

    with pytest.raises(ReplicationError, match="leader role|stale"):
        replicator.advance_commit_index()
