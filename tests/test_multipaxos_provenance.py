"""Ballot-only corruption against real quorum choices; no fabricated chosen map."""

import copy
from dataclasses import replace

import pytest

from distlab.kv import Delete, Put
from distlab.linearizability import Get, SingleKeyKVLinearizabilityChecker
from distlab.multipaxos import DecisionRecoveryResponse, LeaderDecision, MultiPaxosCluster
from distlab.multipaxos_client import ClientStatus, MultiPaxosClient
from distlab.multipaxos_state_machine import Append, MultiPaxosRequest, MultiPaxosStateMachine
from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotAcceptedValue, SlotLearnedDecision
from distlab.simulator import Simulator

NODES = ("n1", "n2", "n3")


def ready():
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    ballot = cluster.node("n1").prepare_leadership()
    sim.run()
    assert cluster.node("n1").has_leader_authority
    return sim, cluster, machine, ballot


def reconstruct(sim):
    recovered = Simulator()
    recovered.persistent_state.update(copy.deepcopy(sim.persistent_state))
    cluster = MultiPaxosCluster(recovered, NODES, learn_decisions=True)
    return recovered, cluster, MultiPaxosStateMachine(cluster)


def chosen():
    sim, cluster, machine, ballot = ready()
    cluster.node("n1").propose(1, Put("k", "chosen"))
    sim.run()
    decision = cluster.node("n2").learned_decision(1)
    assert decision.proposal == ballot
    assert all(
        any(a.slot == 1 and a.proposal == ballot and a.value == decision.value
            for a in cluster.node(node).accept_history)
        for node in NODES
    )
    machine.assert_safety()
    return sim, cluster, machine, decision


@pytest.mark.parametrize("reload", [False, True], ids=["checkpoint", "reconstruction"])
def test_rejects_correct_value_with_forged_learned_ballot(reload):
    sim, cluster, _, real = chosen()
    forged = replace(real, proposal=ProposalNumber(99, "n3"))
    assert not any(
        a.proposal == forged.proposal for node in NODES for a in cluster.node(node).accept_history
    )
    # The only changed field is the durable learned ballot; no applied state exists yet.
    sim.persistent_state["n2"]["multipaxos_learned"][1] = forged
    with pytest.raises(PaxosSafetyViolation, match="durable quorum evidence"):
        if reload:
            reconstruct(sim)
        else:
            cluster.assert_safety()


@pytest.mark.parametrize("reload", [False, True], ids=["checkpoint", "reconstruction"])
def test_rejects_applied_ballot_different_from_learned(reload):
    sim, _, machine, real = chosen()
    machine.apply_ready("n2")
    assert machine.applied("n2") == (real,)
    sim.persistent_state["n2"]["multipaxos_applied"] = (
        replace(real, proposal=ProposalNumber(99, "n3")),
    )
    assert sim.persistent_state["n2"]["multipaxos_learned"][1] == real
    with pytest.raises(PaxosSafetyViolation, match="matching durable learned decision"):
        if reload:
            reconstruct(sim)
        else:
            machine.assert_safety()


@pytest.mark.parametrize("duplicate", [False, True], ids=["first-learn", "duplicate"])
def test_runtime_rejects_forged_ballot_before_learning(duplicate):
    sim, cluster, machine, real = chosen()
    if not duplicate:
        # A follower can miss a notification even though the quorum chose it.
        del sim.persistent_state["n2"]["multipaxos_learned"][1]
    before = cluster.node("n2").learned_decisions
    forged = ProposalNumber(99, "n3")
    sim.send("n3", "n2", LeaderDecision(1, forged, real.value))
    with pytest.raises(PaxosSafetyViolation, match="durable quorum evidence"):
        sim.run()
    assert cluster.node("n2").learned_decisions == before
    assert machine.applied("n2") == ()
    assert dict(machine.state("n2")) == {}
    machine.assert_safety()


@pytest.mark.parametrize("value", [Put("k", "v"), Append("k", "once"), None])
def test_valid_ballot_repeated_phase2_and_duplicate_learning_apply_once(value):
    sim, cluster, machine, ballot = ready()
    leader = cluster.node("n1")
    leader.propose(1, value)
    sim.run()
    for node in NODES:
        assert len(machine.apply_ready(node)) == 1
    sim.run()
    leader.propose(1, value)
    sim.run()
    for node in NODES:
        assert cluster.node(node).accept_history == (SlotAcceptedValue(1, ballot, value),)
        assert machine.apply_ready(node) == ()
        assert machine.applied(node) == (SlotLearnedDecision(1, ballot, value),)
        expected = {"k": "v"} if isinstance(value, Put) else (
            {"k": "once"} if isinstance(value, Append) else {}
        )
        assert dict(machine.state(node)) == expected
    machine.assert_safety()


def rechoose(value):
    sim, cluster, machine, first = ready()
    sim.block_link("n1", "n3")
    cluster.node("n1").propose(1, value)
    sim.run()
    assert cluster.node("n3").learned_decision(1) is None
    for node in ("n1", "n2"):
        machine.apply_ready(node)
    sim.run()
    sim.crash("n1")
    sim.heal_link("n1", "n3")
    second = cluster.node("n3").prepare_leadership(round_number=2)
    sim.run()
    assert cluster.node("n3").has_leader_authority and second > first
    cluster.node("n3").propose(1, "must-adopt-existing-value")
    sim.run()
    # n3 first learns the higher ballot; n2 retains its old learned record.
    assert cluster.chosen(1).proposal == first
    assert cluster.node("n2").learned_decision(1).proposal == first
    assert cluster.node("n3").learned_decision(1).proposal == second
    for ballot in (first, second):
        assert sum(
            any(a.slot == 1 and a.proposal == ballot and a.value == value
                for a in cluster.node(node).accept_history)
            for node in NODES
        ) >= cluster.quorum_size
    sim.restart("n1")
    return sim, cluster, machine, first, second


@pytest.mark.parametrize("value", [Append("k", "once"), None])
def test_higher_ballot_rechoice_preserves_both_provenances_across_restart_and_reconstruction(value):
    sim, cluster, machine, first, second = rechoose(value)
    for node in NODES:
        machine.apply_ready(node)
    prefixes = {node: machine.applied(node) for node in NODES}
    # Delayed older evidence and newer duplicate notifications are both valid;
    # neither may rewrite a node's learned/applied ballot or repeat Append.
    old = cluster.node("n2").learned_decision(1)
    sim.send("n2", "n3", DecisionRecoveryResponse(1, (old,)), delay=5)
    sim.send("n3", "n1", LeaderDecision(1, second, value), delay=3)
    sim.run()
    for node in NODES:
        assert machine.apply_ready(node) == ()
        assert machine.applied(node) == prefixes[node]
    machine.assert_safety()
    recovered_sim, recovered, rebuilt = reconstruct(sim)
    # Reconstruction picks the highest objective ballot, while older learned
    # and applied ballots remain valid because their own historical quorums exist.
    assert recovered.chosen(1).proposal == second
    assert rebuilt.applied("n2")[0].proposal == first
    assert rebuilt.applied("n3")[0].proposal == second
    for node in NODES:
        assert rebuilt.applied(node) == prefixes[node]
        assert dict(rebuilt.state(node)) == ({"k": "once"} if value is not None else {})
    recovered_sim.crash("n3")
    recovered_sim.restart("n3")
    assert rebuilt.apply_ready("n3") == ()
    rebuilt.assert_safety()


def test_applied_ballot_must_equal_local_learned_even_if_both_ballots_have_quorums():
    sim, _, machine, _, second = rechoose(Append("k", "once"))
    real = machine.applied("n2")[0]
    sim.persistent_state["n2"]["multipaxos_applied"] = (replace(real, proposal=second),)
    with pytest.raises(PaxosSafetyViolation, match="matching durable learned decision"):
        reconstruct(sim)


@pytest.mark.parametrize("bad_ballot", [
    None, "not-a-ballot", ProposalNumber(True, "n1"), ProposalNumber(1.5, "n1"),
    ProposalNumber(1, "outsider"), ProposalNumber(1, 7),
])
def test_malformed_or_nonmember_durable_ballot_fails_closed(bad_ballot):
    sim, _, _, real = chosen()
    sim.persistent_state["n2"]["multipaxos_learned"][1] = replace(real, proposal=bad_ballot)
    with pytest.raises(PaxosSafetyViolation, match=r"invalid ballot|non-member proposer"):
        reconstruct(sim)


def test_duplicate_acceptance_records_on_one_node_cannot_prove_a_majority():
    sim, cluster, machine, ballot = ready()
    sim.partition(("n1",), ("n2", "n3"))
    cluster.node("n1").propose(1, Put("k", "minority"))
    accepted = cluster.node("n1").accept_history[0]
    sim.persistent_state["n1"]["multipaxos_accept_history"] = (accepted,) * 3
    assert cluster.chosen(1) is None
    sim.heal_partition(("n1",), ("n2", "n3"))
    sim.send("n1", "n2", LeaderDecision(1, ballot, accepted.value))
    with pytest.raises(PaxosSafetyViolation, match="durable quorum evidence"):
        sim.run()
    assert cluster.node("n2").learned_decision(1) is None
    assert machine.applied("n2") == ()


def test_recovery_validates_later_ballot_before_learning_any_response_record():
    sim, cluster, machine, _ = ready()
    sim.block_link("n1", "n3")
    for slot in (1, 2):
        cluster.node("n1").propose(slot, Append("k", str(slot)))
    sim.run()
    first, second = cluster.node("n1").learned_decisions
    forged = replace(second, proposal=ProposalNumber(99, "n2"))
    sim.send("n2", "n3", DecisionRecoveryResponse(1, (first, forged)))
    with pytest.raises(PaxosSafetyViolation, match="durable quorum evidence"):
        sim.run()
    assert cluster.node("n3").learned_decisions == ()
    assert machine.applied("n3") == ()
    assert dict(machine.state("n3")) == {}


@pytest.mark.parametrize("corruption", ["learned", "applied", "both"])
@pytest.mark.parametrize("apply", [False, True])
def test_cached_client_result_cannot_acknowledge_corrupt_provenance(corruption, apply):
    sim, cluster, machine, _ = ready()
    client = MultiPaxosClient(machine)
    request = MultiPaxosRequest("A", 1, "A-put", Put("k", "v"))
    client.submit("n1", request)
    sim.run()
    assert client.poll("n1", request, acknowledge=False).status == ClientStatus.APPLIED
    real = cluster.node("n1").learned_decision(1)
    forged = replace(real, proposal=ProposalNumber(99, "n3"))
    if corruption in ("learned", "both"):
        sim.persistent_state["n1"]["multipaxos_learned"][1] = forged
    if corruption in ("applied", "both"):
        sim.persistent_state["n1"]["multipaxos_applied"] = (forged,)
    with pytest.raises(PaxosSafetyViolation):
        client.poll("n1", request, apply=apply)
    assert client.history.completed() == ()
    assert [op.operation_id for op in client.history.pending()] == ["A-put"]


def test_crash_restart_rejects_corruption_before_rebuilding_kv_or_dedup():
    sim, _, machine, real = chosen()
    machine.apply_ready("n2")
    forged = replace(real, proposal=ProposalNumber(99, "n3"))
    sim.persistent_state["n2"]["multipaxos_learned"][1] = forged
    sim.persistent_state["n2"]["multipaxos_applied"] = (forged,)
    sim.crash("n2")
    sim.restart("n2")
    with pytest.raises(PaxosSafetyViolation, match="durable quorum evidence"):
        machine.state("n2")
    assert machine._VOLATILE not in sim.volatile_state["n2"]


def test_client_lost_ack_takeover_catchup_reconstruction_and_log_read():
    sim, cluster, machine, first = ready()
    client = MultiPaxosClient(machine)
    write = MultiPaxosRequest("A", 1, "A-put", Put("k", "v"))
    read = MultiPaxosRequest("B", 1, "B-get", Get("k"))
    append = MultiPaxosRequest("A", 2, "A-append", Append("k", "!"))
    assert client.execute("n1", write).status == ClientStatus.COMPLETED
    assert client.execute("n1", read).value == "v"
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable
    sim.partition(("n3",), ("n1", "n2"))
    client.submit("n1", append)
    sim.run()
    assert client.poll("n1", append, acknowledge=False).status == ClientStatus.APPLIED
    assert cluster.node("n3").learned_decision(3) is None
    sim.crash("n1")
    sim.heal_partition(("n3",), ("n1", "n2"))
    second = cluster.node("n2").prepare_leadership()
    sim.run()
    assert second > first
    assert client.execute("n2", append).status == ClientStatus.COMPLETED
    # Chosen again under the new ballot in a different slot: one logical effect.
    cluster.node("n2").propose(4, append)
    sim.run()
    sim.restart("n1")
    for _ in range(4):
        for node in NODES:
            machine.apply_ready(node)
        sim.run()
    read_after = MultiPaxosRequest("B", 2, "B-get-after", Get("k"))
    assert client.execute("n2", read_after).value == "v!"
    delete = MultiPaxosRequest("A", 3, "A-delete", Delete("k"))
    assert client.execute("n2", delete).status == ClientStatus.COMPLETED
    empty = MultiPaxosRequest("B", 3, "B-get-empty", Get("k"))
    assert client.execute("n2", empty).value is None
    for node in NODES:
        assert machine.last_applied(node) == 7
        assert dict(machine.state(node)) == {}
        assert machine.client_requests(node)[append.identity].slot == 3
        assert machine.applied(node)[2].proposal == first
        assert machine.applied(node)[3].proposal == second
    machine.assert_safety()
    # Append is outside the Put/Delete/Get checker. Its effect is verified by
    # the log reads above; the supported write/read prefix was checked earlier.
    assert [op.completion.result for op in client.history.completed()][:2] == [None, "v"]
    recovered_sim, recovered, rebuilt = reconstruct(sim)
    restored_client = MultiPaxosClient(rebuilt, copy.deepcopy(client.journal))
    recovered.node("n2").prepare_leadership()
    recovered_sim.run()
    assert restored_client.execute("n2", append).status == ClientStatus.COMPLETED
    assert restored_client.execute("n2", read_after).value == "v!"
    for node in NODES:
        assert dict(rebuilt.state(node)) == {}
        assert rebuilt.client_requests(node) == machine.client_requests(node)
    rebuilt.assert_safety()
