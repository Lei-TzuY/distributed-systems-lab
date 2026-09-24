import copy

import pytest

from distlab.multipaxos import MultiPaxosCluster
from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.simulator import Simulator


def test_recovered_node_fails_closed_on_non_member_durable_promise() -> None:
    sim = Simulator()
    MultiPaxosCluster(sim, ("n1", "n2", "n3"))
    sim.persistent_state["n1"]["multipaxos_promised_ballot"] = ProposalNumber(
        99, "outsider"
    )

    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))
    recovered = MultiPaxosCluster(recovered_sim, ("n1", "n2", "n3"))

    with pytest.raises(
        PaxosSafetyViolation,
        match="durable promise has non-member proposer 'outsider'",
    ):
        recovered.node("n1").prepare_leadership()
