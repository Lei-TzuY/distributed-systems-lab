import pytest

from distlab.paxos import (
    AcceptedValue,
    AcceptRequest,
    PaxosCluster,
    PaxosSafetyHarness,
    Prepare,
    ProposalNumber,
)
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_single_decree_paxos_chooses_and_learns_one_value() -> None:
    sim = Simulator()
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))

    proposal = cluster.node("n1").start_proposal("alpha")
    sim.run()

    assert proposal == ProposalNumber(1, "n1")
    assert cluster.chosen is not None
    assert cluster.chosen.value == "alpha"
    assert len(cluster.chosen_acceptors) >= cluster.quorum_size
    learned = {
        cluster.node(node_id).learned_decision.value
        for node_id in cluster.node_ids
        if cluster.node(node_id).learned_decision is not None
    }
    assert learned == {"alpha"}
    PaxosSafetyHarness(cluster).checkpoint()


def test_concurrent_proposers_preserve_single_chosen_value() -> None:
    sim = Simulator()
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_proposal("alpha", round_number=1)
    cluster.node("n2").start_proposal("beta", round_number=1)
    sim.run()

    assert cluster.chosen is not None
    assert cluster.chosen.value == "beta"
    assert cluster.chosen.proposal == ProposalNumber(1, "n2")
    PaxosSafetyHarness(cluster).checkpoint()


def test_higher_proposer_adopts_partially_accepted_value_after_source_crash() -> None:
    fault_plan = FaultPlan(
        rules=tuple(
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst=peer,
                payload_type="AcceptRequest",
            )
            for peer in ("n3", "n4", "n5")
        )
    )
    sim = Simulator(fault_plan=fault_plan)
    cluster = PaxosCluster(sim, ("n1", "n2", "n3", "n4", "n5"))

    first = cluster.node("n1").start_proposal("alpha", round_number=1)
    sim.run()

    assert cluster.chosen is None
    assert cluster.node("n1").accepted_value == AcceptedValue(first, "alpha")
    assert cluster.node("n2").accepted_value == AcceptedValue(first, "alpha")
    assert cluster.node("n3").accepted_value is None

    sim.crash("n1")
    second = cluster.node("n3").start_proposal("beta", round_number=2)
    sim.run()

    assert second == ProposalNumber(2, "n3")
    assert cluster.chosen is not None
    assert cluster.chosen.value == "alpha"
    assert cluster.chosen.proposal == second
    assert cluster.node("n3").learned_decision is not None
    assert cluster.node("n3").learned_decision.value == "alpha"
    PaxosSafetyHarness(cluster).checkpoint()


def test_acceptor_promise_survives_crash_restart_and_rejects_lower_prepare() -> None:
    sim = Simulator()
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))
    high = ProposalNumber(5, "n1")

    sim.send("n1", "n2", Prepare(high, "n1"))
    sim.run()
    assert cluster.node("n2").promised_proposal == high

    sim.crash("n2")
    sim.restart("n2")
    assert cluster.node("n2").promised_proposal == high

    lower = ProposalNumber(4, "n3")
    sim.send("n3", "n2", Prepare(lower, "n3"))
    sim.run()

    assert cluster.node("n2").promised_proposal == high
    rejected = [
        record
        for record in sim.trace
        if record.kind == "paxos-prepare-rejected"
        and record.details["acceptor"] == "n2"
    ]
    assert rejected[-1].details["proposal"] == lower
    PaxosSafetyHarness(cluster).checkpoint()


def test_duplicate_prepare_and_accept_delivery_is_idempotent() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            rules=(
                FaultRule(
                    FaultAction.DUPLICATE,
                    src="n1",
                    dst="n2",
                    payload_type="Prepare",
                    extra_delay=1,
                ),
                FaultRule(
                    FaultAction.DUPLICATE,
                    src="n1",
                    dst="n2",
                    payload_type="AcceptRequest",
                    extra_delay=1,
                ),
            )
        )
    )
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))

    proposal = cluster.node("n1").start_proposal("alpha")
    sim.run()

    assert cluster.chosen is not None
    assert cluster.chosen.value == "alpha"
    n2_matches = [
        item
        for item in cluster.node("n2").accept_history
        if item.proposal == proposal
    ]
    assert n2_matches == [AcceptedValue(proposal, "alpha")]
    PaxosSafetyHarness(cluster).checkpoint()


def test_proposer_round_is_durable_and_cannot_be_reused_after_restart() -> None:
    sim = Simulator()
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))
    node = cluster.node("n1")

    node.start_proposal("alpha", round_number=7)
    sim.crash("n1")
    sim.restart("n1")

    with pytest.raises(ValueError, match="must exceed durable local round 7"):
        node.start_proposal("different", round_number=7)

    proposal = node.start_proposal("beta")
    assert proposal == ProposalNumber(8, "n1")
    PaxosSafetyHarness(cluster).checkpoint()


def test_lower_accept_request_cannot_cross_durable_promise() -> None:
    sim = Simulator()
    cluster = PaxosCluster(sim, ("n1", "n2", "n3"))
    promised = ProposalNumber(9, "n1")
    lower = ProposalNumber(8, "n3")

    sim.send("n1", "n2", Prepare(promised, "n1"))
    sim.run()
    sim.send("n3", "n2", AcceptRequest(lower, "n3", "unsafe"))
    sim.run()

    assert cluster.node("n2").promised_proposal == promised
    assert cluster.node("n2").accepted_value is None
    assert [
        record
        for record in sim.trace
        if record.kind == "paxos-accept-rejected"
        and record.details["acceptor"] == "n2"
        and record.details["proposal"] == lower
    ]
    PaxosSafetyHarness(cluster).checkpoint()
