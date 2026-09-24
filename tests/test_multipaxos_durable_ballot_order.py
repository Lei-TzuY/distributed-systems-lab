import copy

import pytest

from distlab.multipaxos import MultiPaxosCluster
from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotAcceptedValue
from distlab.simulator import Simulator


def test_reconstruction_rejects_accepted_ballot_above_durable_promise() -> None:
    sim = Simulator()
    MultiPaxosCluster(sim, ("n1", "n2", "n3"))
    accepted = SlotAcceptedValue(1, ProposalNumber(3, "n1"), "alpha")
    state = sim.persistent_state["n1"]
    state["multipaxos_promised_ballot"] = ProposalNumber(2, "n1")
    state["multipaxos_accepted"] = {1: accepted}
    state["multipaxos_accept_history"] = (accepted,)

    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))

    with pytest.raises(
        PaxosSafetyViolation,
        match="durable accepted proposal exceeds promised ballot",
    ):
        MultiPaxosCluster(recovered_sim, ("n1", "n2", "n3"))


def test_reconstruction_rejects_accepted_history_without_durable_promise() -> None:
    sim = Simulator()
    MultiPaxosCluster(sim, ("n1", "n2", "n3"))
    accepted = SlotAcceptedValue(1, ProposalNumber(1, "n1"), "alpha")
    state = sim.persistent_state["n1"]
    state["multipaxos_accepted"] = {1: accepted}
    state["multipaxos_accept_history"] = (accepted,)

    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))

    with pytest.raises(
        PaxosSafetyViolation,
        match="durable accepted proposal exceeds promised ballot",
    ):
        MultiPaxosCluster(recovered_sim, ("n1", "n2", "n3"))
