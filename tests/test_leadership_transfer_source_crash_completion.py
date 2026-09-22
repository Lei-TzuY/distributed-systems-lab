from distlab.leadership_transfer import LeadershipTransfer
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_transfer_completes_when_source_crashes_after_timeout_now_delivery(monkeypatch) -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    target = cluster.node("n2")
    original_start_election = target.start_election

    def start_election_then_crash_source() -> None:
        original_start_election()
        sim.crash(source.node_id)

    monkeypatch.setattr(target, "start_election", start_election_then_crash_source)

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    result = LeadershipTransfer(source).transfer("n2")

    assert not sim.is_alive("n1")
    assert target.role is RaftRole.LEADER
    assert result.previous_leader_id == "n1"
    assert result.new_leader_id == "n2"
    assert result.new_term == result.previous_term + 1
    assert any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)
    assert any(record.kind == "raft-leadership-transfer-complete" for record in sim.trace)
    assert not any(record.kind == "raft-leadership-transfer-failed" for record in sim.trace)
    harness.checkpoint()


def test_transfer_completes_when_source_restarts_before_target_wins(monkeypatch) -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DROP,
                    src="n2",
                    dst="n1",
                    ordinal=2,
                    payload_type="RequestVote",
                ),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    source = cluster.node("n1")
    source.start_election()
    sim.run()
    assert source.role is RaftRole.LEADER

    target = cluster.node("n2")
    original_start_election = target.start_election

    def start_election_then_restart_source() -> None:
        original_start_election()
        sim.crash(source.node_id)
        sim.restart(source.node_id)

    monkeypatch.setattr(target, "start_election", start_election_then_restart_source)

    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()
    result = LeadershipTransfer(source).transfer("n2")

    assert sim.is_alive("n1")
    assert source.role is RaftRole.FOLLOWER
    assert source.current_term == result.previous_term
    assert target.role is RaftRole.LEADER
    assert result.new_term == result.previous_term + 1
    assert any(
        record.kind == "drop"
        and record.details["src"] == "n2"
        and record.details["dst"] == "n1"
        and record.details["ordinal"] == 2
        and type(record.details["payload"]).__name__ == "RequestVote"
        for record in sim.trace
    )
    assert any(record.kind == "raft-leadership-transfer-complete" for record in sim.trace)
    assert not any(record.kind == "raft-leadership-transfer-failed" for record in sim.trace)
    harness.checkpoint()
