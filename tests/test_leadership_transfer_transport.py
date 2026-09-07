from distlab.leadership_transfer import LeadershipTransfer
from distlab.leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def _elected_cluster(*, fault_plan: FaultPlan | None = None) -> tuple[Simulator, RaftCluster]:
    sim = Simulator(fault_plan=fault_plan)
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.node("n1").current_term == 1
    return sim, cluster


def test_leadership_transfer_routes_timeout_now_through_simulator() -> None:
    sim, cluster = _elected_cluster()
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    result = LeadershipTransfer(cluster.node("n1")).transfer("n2")

    assert result.previous_leader_id == "n1"
    assert result.new_leader_id == "n2"
    assert result.previous_term == 1
    assert result.new_term == 2
    assert cluster.node("n2").role is RaftRole.LEADER

    requests = [record for record in sim.trace if record.kind == "raft-timeout-now-request"]
    delivered = [record for record in sim.trace if record.kind == "raft-timeout-now-delivered"]
    assert requests[-1].details == {"leader": "n1", "transferee": "n2", "term": 1}
    assert delivered[-1].details == {"leader": "n1", "transferee": "n2", "term": 1}
    assert any(
        record.kind == "send"
        and isinstance(record.details["payload"], TimeoutNow)
        and record.details["src"] == "n1"
        and record.details["dst"] == "n2"
        for record in sim.trace
    )
    harness.checkpoint()


def test_delayed_timeout_now_is_rejected_after_source_loses_leadership() -> None:
    fault_plan = FaultPlan(
        (
            FaultRule(
                action=FaultAction.DELAY,
                src="n1",
                dst="n2",
                ordinal=2,
                extra_delay=5,
            ),
            FaultRule(
                action=FaultAction.DROP,
                src="n3",
                dst="n2",
                ordinal=1,
            ),
        )
    )
    sim, cluster = _elected_cluster(fault_plan=fault_plan)
    transport = LeadershipTransferTransport.for_cluster(cluster)
    harness = RaftSafetyHarness(cluster)
    harness.checkpoint()

    transport.send_timeout_now("n1", "n2", term=1)
    cluster.node("n3").start_election()
    sim.run()

    assert cluster.node("n3").role is RaftRole.LEADER
    assert cluster.node("n3").current_term == 2
    assert cluster.node("n1").role is RaftRole.FOLLOWER
    assert cluster.node("n1").current_term == 2
    assert cluster.node("n2").current_term == 1
    assert cluster.node("n2").role is RaftRole.FOLLOWER
    assert not any(record.kind == "raft-timeout-now-delivered" for record in sim.trace)

    rejected = [record for record in sim.trace if record.kind == "raft-timeout-now-rejected"]
    assert rejected[-1].details["leader"] == "n1"
    assert rejected[-1].details["transferee"] == "n2"
    assert rejected[-1].details["term"] == 1
    assert rejected[-1].details["reason"] == "source-no-longer-current-leader"
    harness.checkpoint()
