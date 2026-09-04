import pytest

from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator, ReplicationError
from distlab.simulator import Simulator


def _elect_seeded_leader(
    *,
    leader_log: tuple[LogEntry, ...],
    follower_log: tuple[LogEntry, ...],
) -> tuple[Simulator, RaftCluster, LeaderReplicator]:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = leader_log
    sim.persistent_state["n2"]["log"] = follower_log
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    return sim, cluster, LeaderReplicator(leader)


def test_replication_backtracks_until_missing_suffix_matches() -> None:
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
        LogEntry(term=3, command="c"),
    )
    sim, cluster, replicator = _elect_seeded_leader(
        leader_log=leader_log,
        follower_log=(leader_log[0],),
    )

    assert replicator.progress("n2").next_index == 4
    assert replicator.replicate("n2") is True

    assert cluster.node("n2").log == leader_log
    assert replicator.progress("n2").match_index == 3
    assert replicator.progress("n2").next_index == 4

    probes = [record for record in sim.trace if record.kind == "raft-replication-probe"]
    assert [record.details["prev_log_index"] for record in probes] == [3, 2, 1]
    backtracks = [
        record for record in sim.trace if record.kind == "raft-replication-backtrack"
    ]
    assert [record.details["next_index"] for record in backtracks] == [3, 2]


def test_replication_repairs_missing_suffix_after_compacted_boundary() -> None:
    sim = Simulator()
    boundary = {"log_base_index": 2, "log_base_term": 2}
    sim.persistent_state["n1"].update(
        {
            **boundary,
            "log": (LogEntry(term=3, command="c"),),
        }
    )
    sim.persistent_state["n2"].update({**boundary, "log": ()})
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    replicator = LeaderReplicator(leader)
    assert replicator.progress("n2").next_index == 4
    assert replicator.replicate("n2") is True

    follower = cluster.node("n2")
    assert follower.log_base_index == 2
    assert follower.log_base_term == 2
    assert follower.log == (LogEntry(term=3, command="c"),)
    assert follower.last_log_index == 3
    assert replicator.progress("n2").match_index == 3
    assert replicator.progress("n2").next_index == 4

    probes = [record for record in sim.trace if record.kind == "raft-replication-probe"]
    assert [record.details["prev_log_index"] for record in probes] == [3, 2]
    assert [record.details["entry_count"] for record in probes] == [0, 1]
    cluster.assert_log_matching()


def test_replication_repairs_conflicting_follower_suffix() -> None:
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="leader-b"),
        LogEntry(term=3, command="leader-c"),
    )
    follower_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=8, command="stale-b"),
        LogEntry(term=8, command="stale-c"),
    )
    _, cluster, replicator = _elect_seeded_leader(
        leader_log=leader_log,
        follower_log=follower_log,
    )

    assert replicator.replicate("n2") is True

    assert cluster.node("n2").log == leader_log
    assert replicator.progress("n2").match_index == 3
    cluster.assert_log_matching()


def test_successful_up_to_date_probe_records_full_match() -> None:
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
    )
    sim, _, replicator = _elect_seeded_leader(
        leader_log=leader_log,
        follower_log=leader_log,
    )

    assert replicator.replicate("n2") is True
    progress = replicator.progress("n2")
    assert progress.match_index == 2
    assert progress.next_index == 3

    probes = [record for record in sim.trace if record.kind == "raft-replication-probe"]
    assert len(probes) == 1
    assert probes[0].details["entry_count"] == 0
    assert probes[0].details["prev_log_index"] == 2


def test_bounded_retry_preserves_backtracked_progress_for_next_call() -> None:
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="b"),
        LogEntry(term=3, command="c"),
    )
    _, cluster, replicator = _elect_seeded_leader(
        leader_log=leader_log,
        follower_log=(),
    )

    assert replicator.replicate("n2", max_attempts=2) is False
    assert replicator.progress("n2").next_index == 2
    assert replicator.progress("n2").match_index == 0

    assert replicator.replicate("n2", max_attempts=2) is True
    assert cluster.node("n2").log == leader_log
    assert replicator.progress("n2").match_index == 3


def test_next_index_never_backtracks_below_one() -> None:
    leader_log = (LogEntry(term=1, command="a"),)
    _, _, replicator = _elect_seeded_leader(
        leader_log=leader_log,
        follower_log=(),
    )

    assert replicator.replicate("n2", max_attempts=1) is False
    assert replicator.progress("n2").next_index == 1

    assert replicator.replicate("n2", max_attempts=1) is True
    assert replicator.progress("n2").match_index == 1
    assert replicator.progress("n2").next_index == 2


def test_replicator_rejects_non_leader_and_invalid_attempt_bound() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    with pytest.raises(ReplicationError, match="leader role"):
        LeaderReplicator(cluster.node("n1"))

    cluster.node("n1").start_election()
    sim.run()
    replicator = LeaderReplicator(cluster.node("n1"))

    with pytest.raises(ValueError, match="max_attempts"):
        replicator.replicate("n2", max_attempts=0)
    with pytest.raises(ValueError, match="unknown peer"):
        replicator.progress("missing")
