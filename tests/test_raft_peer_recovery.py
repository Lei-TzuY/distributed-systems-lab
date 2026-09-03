from distlab.raft import LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator
from distlab.simulator import Simulator


def test_restarted_follower_recovers_commit_from_leader_heartbeat() -> None:
    sim = Simulator()
    entry = LogEntry(term=2, command="x")
    sim.persistent_state["n1"]["current_term"] = 1
    sim.persistent_state["n1"]["log"] = (entry,)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")

    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 2

    replicator = LeaderReplicator(leader)
    assert replicator.replicate("n2") is True
    assert replicator.commit_index == 1
    assert replicator.replicate("n2") is True
    assert cluster.node("n2").commit_index == 1

    sim.crash("n2")
    sim.restart("n2")
    follower = cluster.node("n2")
    assert follower.log == (entry,)
    assert follower.commit_index == 0

    recovered = LeaderReplicator(leader)
    assert recovered.recover_peer("n2", max_attempts=1) is True
    assert follower.log == leader.log
    assert follower.commit_index == 1
    assert recovered.progress("n2").match_index == 1
    assert any(
        record.kind == "raft-peer-recovered"
        and record.details["follower"] == "n2"
        and record.details["commit_index"] == 1
        for record in sim.trace
    )


def test_restarted_follower_backtracks_conflicting_suffix_then_recovers_commit() -> None:
    sim = Simulator()
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=3, command="b"),
    )
    stale_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="stale"),
    )
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = leader_log
    sim.persistent_state["n2"]["log"] = stale_log
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")

    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    assert leader.current_term == 3

    sim.crash("n2")
    sim.restart("n2")
    follower = cluster.node("n2")
    assert follower.log == stale_log
    assert follower.commit_index == 0

    replicator = LeaderReplicator(leader)
    assert replicator.recover_peer("n2", max_attempts=3) is True

    assert follower.log == leader_log
    assert leader.commit_index == 2
    assert follower.commit_index == 2
    assert replicator.progress("n2").match_index == 2
    assert any(
        record.kind == "raft-replication-backtrack"
        and record.details["follower"] == "n2"
        for record in sim.trace
    )


def test_peer_recovery_bound_counts_commit_propagation_heartbeat() -> None:
    sim = Simulator()
    leader_log = (
        LogEntry(term=1, command="a"),
        LogEntry(term=3, command="b"),
    )
    sim.persistent_state["n1"]["current_term"] = 2
    sim.persistent_state["n1"]["log"] = leader_log
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1, command="a"),
        LogEntry(term=2, command="stale"),
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    sim.crash("n2")
    sim.restart("n2")
    follower = cluster.node("n2")
    replicator = LeaderReplicator(leader)

    assert replicator.recover_peer("n2", max_attempts=2) is False
    assert follower.log == leader_log
    assert leader.commit_index == 2
    assert follower.commit_index == 0

    assert replicator.recover_peer("n2", max_attempts=1) is True
    assert follower.commit_index == 2
