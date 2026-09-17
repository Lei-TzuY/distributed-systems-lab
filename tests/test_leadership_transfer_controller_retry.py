from distlab.leadership_transfer import LeadershipTransfer
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_transfer_controller_retries_same_timeout_now_attempt_after_drop() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DROP,
                    src="n1",
                    dst="n2",
                    ordinal=2,
                    payload_type="TimeoutNow",
                ),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    result = LeadershipTransfer(cluster.node("n1")).transfer(
        "n2",
        max_timeout_now_attempts=2,
    )

    assert result.new_leader_id == "n2"
    assert cluster.node("n2").role is RaftRole.LEADER
    retries = [record for record in sim.trace if record.kind == "raft-timeout-now-retry"]
    assert len(retries) == 1
    timeout_sends = [
        record
        for record in sim.trace
        if record.kind == "send" and record.details["payload"].__class__.__name__ == "TimeoutNow"
    ]
    assert [record.details["ordinal"] for record in timeout_sends] == [2, 3]
    attempt_ids = {record.details["payload"].attempt_id for record in timeout_sends}
    assert len(attempt_ids) == 1
    harness.checkpoint()
