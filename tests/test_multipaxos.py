import copy

import pytest

from distlab.multipaxos import (
    LeaderAccept,
    LeaderAccepted,
    LeaderPrepare,
    LeaderPromise,
    MultiPaxosCluster,
)
from distlab.paxos import PaxosError, PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotAcceptedValue
from distlab.simulator import Simulator


def _cluster() -> tuple[Simulator, MultiPaxosCluster]:
    sim = Simulator()
    return sim, MultiPaxosCluster(sim, ("n1", "n2", "n3"))


def _reconstruct(sim: Simulator) -> tuple[Simulator, MultiPaxosCluster]:
    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))
    return recovered_sim, MultiPaxosCluster(recovered_sim, ("n1", "n2", "n3"))


def test_stable_leader_amortizes_phase1_across_slots() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")

    ballot = leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority

    leader.propose(1, "alpha")
    leader.propose(2, "beta")
    leader.propose(3, "gamma")
    sim.run()

    assert cluster.chosen(1).value == "alpha"
    assert cluster.chosen(2).value == "beta"
    assert cluster.chosen(3).value == "gamma"
    assert sum(record.kind == "multipaxos-phase1-start" for record in sim.trace) == 1
    assert sum(record.kind == "multipaxos-phase1-complete" for record in sim.trace) == 1
    assert sum(record.kind == "multipaxos-phase2-start" for record in sim.trace) == 3
    assert all(cluster.node(node).promised_ballot == ballot for node in cluster.node_ids)
    cluster.assert_safety()


def test_new_leader_phase1_adopts_highest_previously_accepted_value() -> None:
    sim, cluster = _cluster()
    first = cluster.node("n1")
    first_ballot = first.prepare_leadership(round_number=1)
    sim.run()

    # Leave an accepted value below quorum: partition before Phase 2 so the
    # leader accepts locally while its peer messages are deterministically dropped.
    sim.partition(("n1",), ("n2", "n3"))
    first.propose(1, "old")
    sim.heal_partition(("n1",), ("n2", "n3"))

    # A higher ballot must discover and preserve that accepted value through Phase 1.
    second = cluster.node("n2")
    second.prepare_leadership(round_number=2)
    sim.run()
    assert second.has_leader_authority

    second.propose(1, "new")
    sim.run()
    assert cluster.chosen(1).value == "old"
    assert cluster.chosen(1).proposal > first_ballot
    cluster.assert_safety()


def test_higher_global_promise_fences_lower_ballot_for_every_slot() -> None:
    sim, cluster = _cluster()
    first = cluster.node("n1")
    first.prepare_leadership(round_number=1)
    sim.run()

    second = cluster.node("n2")
    higher = second.prepare_leadership(round_number=2)
    sim.run()
    assert second.has_leader_authority

    stale = LeaderAccept(99, ProposalNumber(1, "n1"), "stale")
    for node_id in cluster.node_ids:
        cluster.node(node_id)._receive_accept(stale)
        assert cluster.node(node_id).promised_ballot == higher
        assert all(item.slot != 99 for item in cluster.node(node_id).accept_history)

    cluster.assert_safety()


def test_crash_discards_volatile_authority_but_keeps_durable_global_promise() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority

    sim.crash("n1")
    sim.restart("n1")
    assert not leader.has_leader_authority
    assert leader.promised_ballot == ballot
    with pytest.raises(PaxosError, match="Phase-1 quorum"):
        leader.propose(1, "unsafe")

    new_ballot = leader.prepare_leadership()
    sim.run()
    assert new_ballot > ballot
    leader.propose(1, "safe")
    sim.run()
    assert cluster.chosen(1).value == "safe"
    cluster.assert_safety()


def test_runtime_reconstruction_recovers_chosen_slots_from_durable_quorum() -> None:
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


def test_runtime_reconstruction_fails_closed_on_non_member_durable_proposer() -> None:
    sim, _ = _cluster()
    corrupt = SlotAcceptedValue(5, ProposalNumber(1, "outsider"), "corrupt")
    sim.persistent_state["n1"]["multipaxos_accept_history"] = (corrupt,)

    with pytest.raises(PaxosSafetyViolation, match="non-member proposer 'outsider'"):
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


def test_conflicting_same_ballot_promise_evidence_cannot_grant_phase1_authority() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    sim.partition(("n1",), ("n2", "n3"))
    ballot = leader.prepare_leadership(round_number=2)
    assert not leader.has_leader_authority

    accepted = (
        SlotAcceptedValue(4, ProposalNumber(1, "n2"), "alpha"),
        SlotAcceptedValue(4, ProposalNumber(1, "n2"), "beta"),
    )
    sim.heal_link("n2", "n1")
    sim.send("n2", "n1", LeaderPromise(ballot, "n2", accepted))
    sim.run()

    assert not leader.has_leader_authority
    assert set(leader._promises) == {"n1"}
    assert sum(
        record.kind == "multipaxos-promise-invalid-evidence" for record in sim.trace
    ) == 1
    cluster.assert_safety()
