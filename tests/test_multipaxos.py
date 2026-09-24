from __future__ import annotations

import pytest

from distlab.multipaxos import (
    LeaderAccept,
    LeaderAccepted,
    LeaderPrepare,
    LeaderPromise,
    MultiPaxosCluster,
)
from distlab.paxos import PaxosError, PaxosSafetyViolation, ProposalNumber
from distlab.simulator import Simulator
from distlab.storage import SlotAcceptedValue


def _cluster() -> tuple[Simulator, MultiPaxosCluster]:
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, ("n1", "n2", "n3"))
    return sim, cluster


def _reconstruct(sim: Simulator) -> tuple[Simulator, MultiPaxosCluster]:
    recovered = Simulator()
    for node_id, state in sim.persistent_state.items():
        recovered.persistent_state[node_id].update(state)
    return recovered, MultiPaxosCluster(recovered, ("n1", "n2", "n3"))


def test_leader_establishes_phase1_once_and_reuses_ballot_across_slots() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")

    ballot = leader.prepare_leadership()
    sim.run()

    assert leader.has_leader_authority
    leader.propose(1, "alpha")
    leader.propose(2, "beta")
    sim.run()

    assert cluster.chosen(1).value == "alpha"
    assert cluster.chosen(2).value == "beta"
    assert cluster.chosen(1).proposal == ballot
    assert cluster.chosen(2).proposal == ballot
    assert sum(record.kind == "multipaxos-prepare" for record in sim.trace) == 1
    assert sum(record.kind == "multipaxos-phase1-complete" for record in sim.trace) == 1
    cluster.assert_safety()


def test_new_leader_adopts_highest_previously_accepted_value() -> None:
    sim, cluster = _cluster()
    first = cluster.node("n1")
    first_ballot = first.prepare_leadership()
    sim.run()
    assert first.has_leader_authority

    sim.partition(("n1",), ("n2", "n3"))
    first.propose(5, "alpha")
    assert cluster.chosen(5) is None
    assert cluster.node("n1").accepted_value(5).proposal == first_ballot

    sim.heal_partition(("n1",), ("n2", "n3"))
    second = cluster.node("n2")
    second_ballot = second.prepare_leadership(round_number=2)
    sim.run()
    assert second.has_leader_authority

    adopted = second.propose(5, "beta")
    sim.run()

    assert adopted == "alpha"
    assert cluster.chosen(5).value == "alpha"
    assert cluster.chosen(5).proposal == second_ballot
    cluster.assert_safety()


def test_lower_ballot_cannot_regain_leadership_after_higher_global_promise() -> None:
    sim, cluster = _cluster()
    high = cluster.node("n2")
    high.prepare_leadership(round_number=3)
    sim.run()
    assert high.has_leader_authority

    low = cluster.node("n1")
    low.prepare_leadership(round_number=2)
    sim.run()

    assert not low.has_leader_authority
    with pytest.raises(PaxosError, match="phase 1 quorum"):
        low.propose(1, "stale")
    cluster.assert_safety()


def test_restart_preserves_global_promise_and_accepted_slots() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    leader.propose(3, "durable")
    sim.run()

    follower = cluster.node("n2")
    assert follower.promised_ballot == ballot
    assert follower.accepted_value(3).value == "durable"

    sim.crash("n2")
    sim.restart("n2")

    assert follower.promised_ballot == ballot
    assert follower.accepted_value(3).value == "durable"
    cluster.assert_safety()


def test_runtime_reconstruction_recovers_chosen_slots_from_durable_acceptance() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    leader.propose(1, "alpha")
    leader.propose(2, "beta")
    sim.run()

    recovered_sim, recovered = _reconstruct(sim)

    assert recovered.chosen(1).value == "alpha"
    assert recovered.chosen(1).proposal == ballot
    assert recovered.chosen(2).value == "beta"
    assert recovered.chosen(2).proposal == ballot
    assert sum(
        record.kind == "multipaxos-chosen-recovered" for record in recovered_sim.trace
    ) == 2
    recovered.assert_safety()


def test_runtime_reconstruction_does_not_infer_chosen_from_minority_acceptance() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    leader.prepare_leadership()
    sim.run()
    sim.partition(("n1",), ("n2", "n3"))
    leader.propose(7, "minority")

    _, recovered = _reconstruct(sim)

    assert recovered.chosen(7) is None
    recovered.assert_safety()


def test_non_member_accepted_response_cannot_form_quorum() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority

    sim.partition(("n1",), ("n2", "n3"))
    leader.propose(8, "member-only")
    assert cluster.chosen(8) is None

    sim.send(
        "outsider",
        "n1",
        LeaderAccepted(8, ballot, "outsider", "member-only"),
    )
    sim.run()

    assert cluster.chosen(8) is None
    assert leader._accepted_by[8] == {"n1"}
    cluster.assert_safety()


def test_non_member_proposer_cannot_mutate_acceptor_state() -> None:
    sim, cluster = _cluster()
    acceptor = cluster.node("n1")
    outsider_ballot = ProposalNumber(99, "outsider")

    sim.send("outsider", "n1", LeaderPrepare(outsider_ballot))
    sim.send("outsider", "n1", LeaderAccept(9, outsider_ballot, "forged"))
    sim.run()

    assert acceptor.promised_ballot is None
    assert all(item.slot != 9 for item in acceptor.accept_history)
    cluster.assert_safety()


def test_member_cannot_persist_non_positive_slot_from_transport() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    acceptor = cluster.node("n2")

    sim.send("n1", "n2", LeaderAccept(0, ballot, "invalid"))
    sim.send("n1", "n2", LeaderAccept(-1, ballot, "invalid"))
    sim.run()

    assert acceptor.accept_history == ()
    assert sum(
        record.kind == "multipaxos-accept-invalid-slot" for record in sim.trace
    ) == 2
    cluster.assert_safety()


def test_runtime_reconstruction_fails_closed_on_invalid_durable_slot() -> None:
    sim, _ = _cluster()
    ballot = ProposalNumber(1, "n1")
    corrupt = object.__new__(SlotAcceptedValue)
    object.__setattr__(corrupt, "slot", 0)
    object.__setattr__(corrupt, "proposal", ballot)
    object.__setattr__(corrupt, "value", "corrupt")
    sim.persistent_state["n1"]["multipaxos_accept_history"] = (corrupt,)

    with pytest.raises(PaxosSafetyViolation, match="invalid slot 0"):
        _reconstruct(sim)


@pytest.mark.parametrize(
    "accepted",
    (
        SlotAcceptedValue(4, ProposalNumber(3, "n2"), "future"),
        SlotAcceptedValue(4, ProposalNumber(1, "outsider"), "outsider"),
    ),
)
def test_forged_promise_evidence_cannot_grant_phase1_authority(
    accepted: SlotAcceptedValue,
) -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    sim.partition(("n1",), ("n2", "n3"))
    ballot = leader.prepare_leadership(round_number=2)
    assert not leader.has_leader_authority

    sim.heal_link("n2", "n1")
    sim.send("n2", "n1", LeaderPromise(ballot, "n2", (accepted,)))
    sim.run()

    assert not leader.has_leader_authority
    assert set(leader._promises) == {"n1"}
    assert sum(
        record.kind == "multipaxos-promise-invalid-evidence" for record in sim.trace
    ) == 1
    cluster.assert_safety()
