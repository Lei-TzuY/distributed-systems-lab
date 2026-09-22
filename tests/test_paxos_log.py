import pytest

from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import (
    PaxosLogCluster,
    PaxosLogSafetyHarness,
    SlotAcceptedValue,
    SlotPrepare,
)
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_two_slots_choose_independently_and_form_ordered_prefix() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))

    first = cluster.node("n1").start_proposal(1, "alpha")
    second = cluster.node("n2").start_proposal(2, "beta")
    sim.run()

    assert first == ProposalNumber(1, "n1")
    assert second == ProposalNumber(1, "n2")
    assert cluster.chosen(1) is not None
    assert cluster.chosen(1).value == "alpha"
    assert cluster.chosen(2) is not None
    assert cluster.chosen(2).value == "beta"
    assert [decision.value for decision in cluster.chosen_prefix()] == ["alpha", "beta"]
    for node_id in cluster.node_ids:
        assert [decision.value for decision in cluster.node(node_id).learned_prefix()] == [
            "alpha",
            "beta",
        ]
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_chosen_slot_after_gap_does_not_advance_prefix() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_proposal(2, "beta")
    sim.run()

    assert cluster.chosen(2) is not None
    assert cluster.chosen_prefix() == ()
    for node_id in cluster.node_ids:
        assert cluster.node(node_id).learned_decision(2) is not None
        assert cluster.node(node_id).learned_prefix() == ()

    cluster.node("n2").start_proposal(1, "alpha")
    sim.run()

    assert [decision.value for decision in cluster.chosen_prefix()] == ["alpha", "beta"]
    for node_id in cluster.node_ids:
        assert [decision.value for decision in cluster.node(node_id).learned_prefix()] == [
            "alpha",
            "beta",
        ]
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_higher_proposal_adopts_only_same_slot_accepted_value() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            rules=(
                FaultRule(
                    FaultAction.DROP,
                    src="n1",
                    dst="n3",
                    ordinal=1,
                    payload_type="SlotAcceptRequest",
                ),
            )
        )
    )
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))

    slot_one = cluster.node("n1").start_proposal(1, "alpha", round_number=1)
    cluster.node("n3").start_proposal(2, "other-slot", round_number=4)
    sim.run()

    assert cluster.node("n1").accepted_value(1) == SlotAcceptedValue(1, slot_one, "alpha")
    assert cluster.node("n2").accepted_value(1) == SlotAcceptedValue(1, slot_one, "alpha")
    assert cluster.chosen(1) is not None
    assert cluster.chosen(1).value == "alpha"
    assert cluster.chosen(2) is not None
    assert cluster.chosen(2).value == "other-slot"

    proposal = cluster.node("n3").start_proposal(1, "replacement", round_number=5)
    sim.run()

    assert proposal == ProposalNumber(5, "n3")
    assert cluster.chosen(1).value == "alpha"
    assert cluster.node("n3").accepted_value(1).value == "alpha"
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_acceptor_slot_state_and_prefix_survive_crash_restart() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_proposal(1, "alpha")
    cluster.node("n2").start_proposal(2, "beta")
    sim.run()
    before = cluster.node("n3").learned_prefix()
    promised_one = cluster.node("n3").promised_proposal(1)
    promised_two = cluster.node("n3").promised_proposal(2)

    sim.crash("n3")
    sim.restart("n3")

    assert cluster.node("n3").learned_prefix() == before
    assert cluster.node("n3").promised_proposal(1) == promised_one
    assert cluster.node("n3").promised_proposal(2) == promised_two
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_proposer_global_round_is_durable_across_slots_and_restart() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    node = cluster.node("n1")

    first = node.start_proposal(1, "alpha", round_number=7)
    assert first == ProposalNumber(7, "n1")
    sim.crash("n1")
    sim.restart("n1")

    with pytest.raises(ValueError, match="must exceed durable local round 7"):
        node.start_proposal(2, "beta", round_number=7)

    second = node.start_proposal(2, "beta")
    assert second == ProposalNumber(8, "n1")
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_unknown_acceptor_cannot_forge_multi_decree_quorum() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    proposal = cluster.node("n1").start_proposal(1, "alpha", round_number=1)

    from distlab.paxos_log import SlotAccepted

    sim.send(
        "forged",
        "n1",
        SlotAccepted(1, proposal, "forged", "alpha"),
        delay=0,
    )
    sim.run(max_events=1)

    assert cluster.chosen(1) is None
    rejected = [
        record
        for record in sim.trace
        if record.kind == "paxos-log-envelope-rejected"
        and record.details["reason"] == "unknown-acceptor"
    ]
    assert rejected
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_lower_prepare_cannot_cross_slot_specific_durable_promise() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    high = ProposalNumber(5, "n1")

    sim.send("n1", "n2", SlotPrepare(1, high, "n1"))
    sim.run()
    assert cluster.node("n2").promised_proposal(1) == high
    assert cluster.node("n2").promised_proposal(2) is None

    low = ProposalNumber(4, "n3")
    sim.send("n3", "n2", SlotPrepare(1, low, "n3"))
    sim.run()

    assert cluster.node("n2").promised_proposal(1) == high
    assert [
        record
        for record in sim.trace
        if record.kind == "paxos-log-prepare-rejected"
        and record.details["slot"] == 1
    ]
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_unhashable_values_are_supported_by_durable_safety_reconstruction() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    value = {"command": ["set", "x", 1]}

    cluster.node("n1").start_proposal(1, value)
    sim.run()

    assert cluster.chosen(1) is not None
    assert cluster.chosen(1).value == value
    PaxosLogSafetyHarness(cluster).checkpoint()


def test_harness_rejects_conflicting_majority_values_in_one_slot() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    p1 = ProposalNumber(1, "n1")
    p2 = ProposalNumber(2, "n2")

    for node_id in ("n1", "n2"):
        sim.persistent_state[node_id]["paxos_log_promised"] = {1: p2}
        sim.persistent_state[node_id]["paxos_log_accepted"] = {
            1: SlotAcceptedValue(1, p2, "beta")
        }
        sim.persistent_state[node_id]["paxos_log_accept_history"] = (
            SlotAcceptedValue(1, p1, "alpha"),
            SlotAcceptedValue(1, p2, "beta"),
        )

    with pytest.raises(PaxosSafetyViolation, match="multiple values"):
        PaxosLogSafetyHarness(cluster).checkpoint()
