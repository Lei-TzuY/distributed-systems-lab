import copy

import pytest

from distlab.multipaxos import MultiPaxosCluster
from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotAcceptedValue
from distlab.simulator import Simulator


def test_reconstruction_rejects_same_acceptor_conflicting_same_ballot_values() -> None:
    sim = Simulator()
    MultiPaxosCluster(sim, ("n1", "n2", "n3"))
    ballot = ProposalNumber(3, "n1")
    alpha = SlotAcceptedValue(1, ballot, "alpha")
    beta = SlotAcceptedValue(1, ballot, "beta")
    state = sim.persistent_state["n1"]
    state["multipaxos_promised_ballot"] = ballot
    state["multipaxos_accepted"] = {1: beta}
    state["multipaxos_accept_history"] = (alpha, beta)

    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))

    with pytest.raises(
        PaxosSafetyViolation,
        match="same Multi-Paxos ballot has conflicting durable accepted values",
    ):
        MultiPaxosCluster(recovered_sim, ("n1", "n2", "n3"))
