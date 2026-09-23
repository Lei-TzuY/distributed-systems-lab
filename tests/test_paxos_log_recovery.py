from copy import deepcopy

import pytest

from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import PaxosLogCluster, SlotAcceptedValue
from distlab.paxos_log_recovery import RecoveringPaxosLogCluster
from distlab.simulator import Simulator


def _reconstruct(source: Simulator) -> tuple[Simulator, RecoveringPaxosLogCluster]:
    recovered_sim = Simulator()
    recovered_sim.persistent_state = deepcopy(source.persistent_state)
    recovered = RecoveringPaxosLogCluster(recovered_sim, ("n1", "n2", "n3"))
    return recovered_sim, recovered


def test_runtime_reconstruction_recovers_chosen_slots_and_gap_sensitive_prefix() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_proposal(2, "beta")
    sim.run()
    assert cluster.chosen(2) is not None

    recovered_sim, recovered = _reconstruct(sim)

    assert recovered.chosen(2) is not None
    assert recovered.chosen(2).value == "beta"
    assert recovered.chosen_prefix() == ()
    assert recovered.chosen_acceptors(2) == ("n1", "n2", "n3")
    assert [
        record
        for record in recovered_sim.trace
        if record.kind == "paxos-log-chosen-recovered"
        and record.details["slot"] == 2
    ]

    recovered.node("n2").start_proposal(1, "alpha")
    recovered_sim.run()

    assert [decision.value for decision in recovered.chosen_prefix()] == ["alpha", "beta"]
    recovered.assert_safety()


def test_recovery_uses_durable_majority_even_when_learn_messages_were_not_delivered() -> None:
    sim = Simulator()
    cluster = PaxosLogCluster(sim, ("n1", "n2", "n3"))
    proposal = cluster.node("n1").start_proposal(1, {"set": ["x", 1]})

    while cluster.chosen(1) is None:
        sim.run(max_events=1)

    assert proposal == ProposalNumber(1, "n1")
    assert any(cluster.node(node_id).learned_decision(1) is None for node_id in cluster.node_ids)

    _, recovered = _reconstruct(sim)

    assert recovered.chosen(1) is not None
    assert recovered.chosen(1).value == {"set": ["x", 1]}
    recovered.assert_safety()


def test_recovery_fails_closed_on_conflicting_durable_quorum_evidence() -> None:
    sim = Simulator()
    PaxosLogCluster(sim, ("n1", "n2", "n3"))
    first = ProposalNumber(1, "n1")
    second = ProposalNumber(2, "n2")

    for node_id in ("n1", "n2"):
        sim.persistent_state[node_id]["paxos_log_promised"] = {1: second}
        sim.persistent_state[node_id]["paxos_log_accepted"] = {
            1: SlotAcceptedValue(1, second, "beta")
        }
        sim.persistent_state[node_id]["paxos_log_accept_history"] = (
            SlotAcceptedValue(1, first, "alpha"),
            SlotAcceptedValue(1, second, "beta"),
        )

    recovered_sim = Simulator()
    recovered_sim.persistent_state = deepcopy(sim.persistent_state)
    with pytest.raises(PaxosSafetyViolation, match="multiple values"):
        RecoveringPaxosLogCluster(recovered_sim, ("n1", "n2", "n3"))
