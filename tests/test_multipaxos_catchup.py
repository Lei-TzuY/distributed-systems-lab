import copy

import pytest

from distlab.multipaxos import (
    RECOVERY_ATTEMPTS,
    RECOVERY_RETRY_DELAY,
    RECOVERY_WINDOW,
    DecisionRecoveryRequest,
    DecisionRecoveryResponse,
    LeaderDecision,
    MultiPaxosCluster,
)
from distlab.multipaxos_state_machine import Append, MultiPaxosStateMachine
from distlab.paxos import PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotLearnedDecision
from distlab.randomized_faults import (
    FaultOpportunity,
    SeededFaultGenerator,
    SeededFaultSchedule,
)
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator

NODES = ("n1", "n2", "n3")


def _partitioned(last=4):
    """n2 has applied 1, misses the entire suffix, and has just been healed."""
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    leader = cluster.node("n1")
    leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority
    leader.propose(1, Append("k", "1,"))
    sim.run()
    for node_id in NODES:
        machine.apply_ready(node_id)
    sim.run()
    sim.partition(("n2",), ("n1", "n3"))
    for slot in range(2, last + 1):
        leader.propose(slot, Append("k", f"{slot},"))
    sim.run()
    assert [d.slot for d in cluster.node("n2").learned_decisions] == [1]
    assert machine.last_applied("n2") == 1
    sim.heal_partition(("n2",), ("n1", "n3"))
    return sim, cluster, machine


def _drive(sim, machine, last, *, rounds=12):
    """Normal application polling plus scheduled work, with a finite test budget."""
    prefixes = {n: machine.applied(n) for n in NODES}
    for _ in range(rounds):
        for node_id in NODES:
            if sim.is_alive(node_id):
                machine.apply_ready(node_id)
            current = machine.applied(node_id)
            assert current[: len(prefixes[node_id])] == prefixes[node_id]
            assert [d.slot for d in current] == list(range(1, len(current) + 1))
            prefixes[node_id] = current
        sim.run()
        machine.assert_safety()
        if all(machine.last_applied(n) == last for n in NODES):
            expected = {"k": "".join(f"{slot}," for slot in range(1, last + 1))}
            assert all(dict(machine.state(n)) == expected for n in NODES)
            return
    pytest.fail("catch-up failed within the bounded application driver")


def _sends(sim, payload_type, *, src=None, dst=None):
    return [
        r for r in sim.trace
        if r.kind == "send"
        and isinstance(r.details["payload"], payload_type)
        and (src is None or r.details["src"] == src)
        and (dst is None or r.details["dst"] == dst)
    ]


def test_partition_heal_recovers_suffix_without_reproposing_and_appends_once():
    sim, cluster, machine = _partitioned()
    # No later local decision hints at this lost suffix. An ordinary application
    # pass probes slot 2; no recovery API or replacement consensus proposal is used.
    assert machine.blocked_on("n2") is None
    _drive(sim, machine, 4)
    assert cluster.node("n2").learned_decision(2).value == Append("k", "2,")
    assert [
        r.details["slot"] for r in sim.trace if r.kind == "multipaxos-phase2-start"
    ] == [1, 2, 3, 4]
    for node_id in NODES:
        assert [
            r.details["slot"] for r in sim.trace
            if r.kind == "multipaxos-apply" and r.details["node"] == node_id
        ] == [1, 2, 3, 4]
        assert machine.apply_ready(node_id) == ()


def test_known_gap_and_multiple_windows_are_repaired_in_order():
    last = 2 * RECOVERY_WINDOW + 3
    sim, cluster, machine = _partitioned(last)
    # Only the final notification gets through after healing.
    decision = cluster.node("n1").learned_decision(last)
    sim.send("n1", "n2", LeaderDecision(last, decision.proposal, decision.value))
    sim.run()
    assert machine.blocked_on("n2") == 2
    assert machine.apply_ready("n2") == ()
    _drive(sim, machine, last)
    assert any(r.kind == "multipaxos-apply-gap" for r in sim.trace)
    requests = _sends(sim, DecisionRecoveryRequest, src="n2")
    assert {2, 2 + RECOVERY_WINDOW, 2 + 2 * RECOVERY_WINDOW} <= {
        r.details["payload"].start_slot for r in requests
    }
    for record in _sends(sim, DecisionRecoveryResponse):
        response = record.details["payload"]
        assert len(response.decisions) <= RECOVERY_WINDOW
        assert all(
            response.start_slot <= d.slot < response.start_slot + RECOVERY_WINDOW
            for d in response.decisions
        )


@pytest.mark.parametrize("payload_type", ["DecisionRecoveryRequest", "DecisionRecoveryResponse"])
def test_lost_first_round_is_retried_by_scheduled_work(payload_type):
    sim, _, machine = _partitioned()
    # Black-hole BOTH peers for the first round so success requires a retry.
    sim.fault_plan = FaultPlan((FaultRule(FaultAction.DROP, payload_type=payload_type),))
    before = len(_sends(sim, DecisionRecoveryRequest, src="n2"))
    machine.apply_ready("n2")
    deadline = sim.time + RECOVERY_RETRY_DELAY - 1
    sim.run_until_time(deadline)
    assert machine.last_applied("n2") == 1
    assert machine.cluster.node("n2").learned_decision(2) is None
    assert any(r.kind == "drop" for r in sim.trace)
    sim.fault_plan = FaultPlan()
    sim.run()  # No second application poll is needed to resend the lost round.
    assert machine.cluster.node("n2").learned_decision(2) is not None
    assert len(_sends(sim, DecisionRecoveryRequest, src="n2")) - before == 4
    _drive(sim, machine, 4)


def test_out_of_order_responses_wait_for_gap_and_duplicates_do_not_reapply():
    sim, cluster, machine = _partitioned(3)
    peer = cluster.node("n3")
    response = DecisionRecoveryResponse(2, (peer.learned_decision(2),))
    sim.send("n3", "n2", DecisionRecoveryResponse(2, (peer.learned_decision(3),)))
    sim.send("n3", "n2", response, delay=3)
    sim.send("n3", "n2", response, delay=8)
    sim.run(max_events=1)
    # Suppress new recovery traffic to isolate the two deliberately reordered replies.
    sim.fault_plan = FaultPlan((
        FaultRule(FaultAction.DROP, payload_type="DecisionRecoveryRequest"),
    ))
    assert machine.apply_ready("n2") == ()
    assert machine.blocked_on("n2") == 2
    assert dict(machine.state("n2")) == {"k": "1,"}
    sim.run_until_time(sim.time + 2)
    assert [d.slot for d in machine.apply_ready("n2")] == [2, 3]
    sim.run()
    assert machine.apply_ready("n2") == ()
    assert dict(machine.state("n2")) == {"k": "1,2,3,"}
    machine.assert_safety()


@pytest.mark.parametrize("payload_type", ["DecisionRecoveryRequest", "DecisionRecoveryResponse"])
def test_duplicate_recovery_messages_preserve_non_idempotent_append(payload_type):
    sim, _, machine = _partitioned()
    sim.fault_plan = FaultPlan((
        FaultRule(FaultAction.DUPLICATE, payload_type=payload_type, extra_delay=9),
    ))
    _drive(sim, machine, 4)
    sim.run()
    assert machine.apply_ready("n2") == ()
    assert dict(machine.state("n2")) == {"k": "1,2,3,4,"}
    machine.assert_safety()


def test_delayed_old_ballot_response_remains_valid_after_takeover():
    sim, cluster, machine = _partitioned()
    old = cluster.node("n1").learned_decision(2)
    sim.fault_plan = FaultPlan((
        FaultRule(FaultAction.DELAY, payload_type="DecisionRecoveryResponse", extra_delay=20),
    ))
    machine.apply_ready("n2")
    sim.run_until_time(sim.time + 1)  # Both peers have sent their old-ballot records.
    new_ballot = cluster.node("n3").prepare_leadership()
    assert new_ballot > old.proposal
    sim.crash("n1")
    sim.run()
    assert cluster.node("n2").promised_ballot == new_ballot
    assert cluster.node("n2").learned_decision(2) == old
    assert [d.slot for d in machine.apply_ready("n2")] == [2, 3, 4]
    assert dict(machine.state("n2")) == {"k": "1,2,3,4,"}
    machine.assert_safety()


def test_crash_during_recovery_preserves_learned_window_and_retries_after_restart():
    last = RECOVERY_WINDOW + 3
    sim, cluster, machine = _partitioned(last)
    machine.apply_ready("n2")
    # Deliver two requests, then one response; crash before application.
    assert sim.run(max_events=3) == 3
    durable = cluster.node("n2").learned_decisions
    assert [d.slot for d in durable] == list(range(1, RECOVERY_WINDOW + 2))
    prefix = machine.applied("n2")
    sim.crash("n2")
    sim.run()  # Other response and old retry timer are discarded while crashed.
    sim.restart("n2")
    assert cluster.node("n2").learned_decisions == durable
    assert machine.applied("n2") == prefix
    assert dict(machine.state("n2")) == {"k": "1,"}
    _drive(sim, machine, last)
    assert machine.applied("n2")[: len(prefix)] == prefix


def test_restart_before_old_timer_or_response_is_delivered_needs_no_request_state():
    sim, cluster, machine = _partitioned()
    machine.apply_ready("n2")
    sim.run(max_events=2)  # Replies are in flight, timer has not fired.
    sim.crash("n2")
    sim.restart("n2")
    sim.run()  # Delayed responses are safe even though no request is active.
    assert cluster.node("n2").learned_decision(2) is not None
    _drive(sim, machine, 4)


def test_full_reconstruction_retries_a_durable_gap_without_volatile_state():
    sim, cluster, machine = _partitioned()
    last = cluster.node("n1").learned_decision(4)
    sim.send("n1", "n2", LeaderDecision(4, last.proposal, last.value))
    sim.run()
    machine.apply_ready("n2")  # Leave requests and the retry timer in the old simulator.
    recovered = Simulator()
    recovered.persistent_state.update(copy.deepcopy(sim.persistent_state))
    new_cluster = MultiPaxosCluster(recovered, NODES, learn_decisions=True)
    new_machine = MultiPaxosStateMachine(new_cluster)
    assert new_machine.blocked_on("n2") == 2
    assert dict(new_machine.state("n2")) == {"k": "1,"}
    _drive(recovered, new_machine, 4)


def test_permanent_partition_exhausts_bounded_work_then_normal_poll_retries_after_heal():
    sim, cluster, machine = _partitioned()
    last = cluster.node("n1").learned_decision(4)
    sim.send("n1", "n2", LeaderDecision(4, last.proposal, last.value))
    sim.run()
    sim.partition(("n2",), ("n1", "n3"))
    before = len(_sends(sim, DecisionRecoveryRequest, src="n2"))
    for _ in range(10):
        machine.apply_ready("n2")  # Coalesced while the same episode is active.
    assert sim.run() == RECOVERY_ATTEMPTS
    assert sim.run() == 0
    assert len(_sends(sim, DecisionRecoveryRequest, src="n2")) - before == 2 * RECOVERY_ATTEMPTS
    assert machine.last_applied("n2") == 1
    assert machine.blocked_on("n2") == 2
    exhausted = [r for r in sim.trace if r.kind == "multipaxos-recovery-exhausted"]
    assert exhausted[-1].details == {
        "node": "n2", "start_slot": 2, "attempts": RECOVERY_ATTEMPTS, "known_gap": True,
    }
    sim.heal_partition(("n2",), ("n1", "n3"))
    _drive(sim, machine, 4)


def test_peer_without_local_learned_evidence_cannot_answer_from_global_chosen_map():
    # No ordinary decision reaches either follower. n1 is the only evidence holder.
    sim = Simulator(fault_plan=FaultPlan((
        FaultRule(FaultAction.DROP, payload_type="LeaderDecision"),
    )))
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    cluster.node("n1").prepare_leadership()
    sim.run()
    cluster.node("n1").propose(1, Append("k", "1,"))
    sim.run()
    assert cluster.chosen(1) is not None
    assert cluster.node("n3").learned_decisions == ()
    sim.crash("n1")
    machine.apply_ready("n2")
    sim.run()
    assert cluster.node("n2").learned_decisions == ()
    assert all(r.details["payload"].decisions == () for r in _sends(
        sim, DecisionRecoveryResponse, src="n3"
    ))
    assert machine.last_applied("n2") == 0
    assert sim.run() == 0
    machine.assert_safety()


def test_unchosen_minority_hole_remains_blocked_without_inventing_a_decision():
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    leader = cluster.node("n1")
    leader.prepare_leadership()
    sim.run()
    leader.propose(1, Append("k", "1,"))
    sim.run()
    sim.partition(("n1",), ("n2", "n3"))
    leader.propose(2, Append("k", "minority"))
    sim.run()
    sim.heal_partition(("n1",), ("n2", "n3"))
    leader.propose(3, Append("k", "3,"))
    sim.run()
    for node_id in NODES:
        machine.apply_ready(node_id)
    sim.run()
    assert sim.run() == 0
    assert cluster.chosen(2) is None
    for node_id in NODES:
        assert cluster.node(node_id).learned_decision(2) is None
        assert machine.last_applied(node_id) == 1
        assert machine.blocked_on(node_id) == 2
        assert dict(machine.state(node_id)) == {"k": "1,"}
    machine.assert_safety()


def test_valid_first_record_does_not_hide_conflicting_later_response_value():
    sim, cluster, machine = _partitioned()
    peer = cluster.node("n1")
    known = peer.learned_decision(4)
    sim.send("n1", "n2", LeaderDecision(4, known.proposal, known.value))
    sim.run()
    forged = SlotLearnedDecision(4, known.proposal, Append("k", "forged"))
    response = DecisionRecoveryResponse(2, (peer.learned_decision(2), forged))
    sim.send("n3", "n2", response)
    with pytest.raises(PaxosSafetyViolation, match="conflicting decisions for slot 4"):
        sim.run()
    assert cluster.node("n2").learned_decision(2) is None
    assert machine.last_applied("n2") == 1
    machine.assert_safety()


@pytest.mark.parametrize("forged_ballot", [False, True])
def test_recovery_rejects_unproven_value_or_ballot_before_learning(forged_ballot):
    sim, cluster, machine = _partitioned()
    chosen = cluster.node("n1").learned_decision(2)
    forged = SlotLearnedDecision(
        2, ProposalNumber(99, "n3") if forged_ballot else chosen.proposal,
        chosen.value if forged_ballot else Append("k", "never-chosen"),
    )
    sim.send("n3", "n2", DecisionRecoveryResponse(2, (forged,)))
    with pytest.raises(PaxosSafetyViolation, match="lacks matching durable quorum evidence"):
        sim.run()
    assert cluster.node("n2").learned_decision(2) is None
    assert machine.applied("n2")[-1].slot == 1
    machine.assert_safety()


@pytest.mark.parametrize("bad_slot", [0, -1, True, 1.5, "2", None])
def test_invalid_request_and_response_start_slots_are_ignored(bad_slot):
    sim, cluster, machine = _partitioned()
    before = len(_sends(sim, DecisionRecoveryResponse))
    sim.send("n2", "n1", DecisionRecoveryRequest(bad_slot))
    sim.send("n1", "n2", DecisionRecoveryResponse(bad_slot, ()))
    sim.run()
    assert len(_sends(sim, DecisionRecoveryResponse)) == before + 1  # Only the injected one.
    assert cluster.node("n2").learned_decision(2) is None
    machine.assert_safety()


@pytest.mark.parametrize("malformation", [
    "list", "object", "oversize", "repeated", "reverse", "outside", "bool-slot",
    "foreign-proposer", "bad-ballot", "bool-round",
])
def test_malformed_response_is_ignored_atomically(malformation):
    sim, cluster, machine = _partitioned()
    first = cluster.node("n1").learned_decision(2)
    second = cluster.node("n1").learned_decision(3)
    malformed = {
        "list": [first],
        "object": (object(),),
        "oversize": (first,) * (RECOVERY_WINDOW + 1),
        "repeated": (first, first),
        "reverse": (second, first),
        "outside": (first, SlotLearnedDecision(99, first.proposal, first.value)),
        "bool-slot": (SlotLearnedDecision(True, first.proposal, first.value),),
        "foreign-proposer": (SlotLearnedDecision(2, ProposalNumber(1, "outsider"), first.value),),
        "bad-ballot": (SlotLearnedDecision(2, None, first.value),),
        "bool-round": (SlotLearnedDecision(2, ProposalNumber(True, "n1"), first.value),),
    }[malformation]
    sim.send("n1", "n2", DecisionRecoveryResponse(2, malformed))
    sim.run()
    assert cluster.node("n2").learned_decision(2) is None
    machine.assert_safety()


@pytest.mark.parametrize("sender", ["outsider", "n2"])
def test_non_peer_recovery_envelopes_are_ignored(sender):
    sim, cluster, machine = _partitioned()
    decision = cluster.node("n1").learned_decision(2)
    before = len(_sends(sim, DecisionRecoveryResponse))
    sim.send(sender, "n2", DecisionRecoveryRequest(1))
    sim.send(sender, "n2", DecisionRecoveryResponse(2, (decision,)))
    sim.run()
    assert len(_sends(sim, DecisionRecoveryResponse)) == before + 1  # Only the injected one.
    assert cluster.node("n2").learned_decision(2) is None
    machine.assert_safety()


def test_negative_trusting_a_peer_claim_is_detected_by_application_invariant(monkeypatch):
    sim, cluster, machine = _partitioned()
    chosen = cluster.node("n1").learned_decision(2)
    forged = SlotLearnedDecision(2, chosen.proposal, Append("k", "forged"))
    # Deliberately introduce the plausible bug: trust any well-formed member claim.
    monkeypatch.setattr(cluster, "_require_recovery_evidence", lambda decision: None)
    sim.send("n3", "n2", DecisionRecoveryResponse(2, (forged,)))
    sim.run()
    assert cluster.node("n2").learned_decision(2) == forged
    with pytest.raises(PaxosSafetyViolation, match="without matching chosen quorum"):
        machine.apply_ready("n2")
    assert machine.last_applied("n2") == 1
    assert dict(machine.state("n2")) == {"k": "1,"}
    with pytest.raises(PaxosSafetyViolation, match="without matching durable quorum"):
        cluster.assert_safety()


def test_learner_disabled_ignores_recovery_and_schedules_no_work():
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES)
    cluster.node("n1").recover_decisions(1)
    assert sim.trace == []
    sim.send("n2", "n1", DecisionRecoveryRequest(1))
    sim.send("n2", "n1", DecisionRecoveryResponse(1, ()))
    assert sim.run() == 2
    assert all(cluster.node(n).learned_decisions == () for n in NODES)
    assert all("multipaxos_learned" not in sim.persistent_state[n] for n in NODES)


def _faulted_catchup(schedule):
    sim, _, machine = _partitioned(2 * RECOVERY_WINDOW + 2)
    sim.fault_plan = schedule.to_fault_plan()
    machine.apply_ready("n2")
    sim.run(max_events=3)
    sim.crash("n2")
    sim.run()
    sim.restart("n2")
    _drive(sim, machine, 2 * RECOVERY_WINDOW + 2, rounds=24)
    return (
        sim.trace, {n: machine.applied(n) for n in NODES},
        {n: dict(machine.state(n)) for n in NODES},
    )


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 919, 1729, 2026, 65537])
def test_seeded_recovery_fault_campaign_replays_trace_and_state_exactly(seed):
    opportunities = tuple(
        FaultOpportunity(src, dst, ordinal, payload_type)
        for src in NODES for dst in NODES if src != dst
        for ordinal in range(1, 140)
        for payload_type in ("DecisionRecoveryRequest", "DecisionRecoveryResponse")
    )
    schedule = SeededFaultGenerator(
        drop_rate=0.15, delay_rate=0.2, duplicate_rate=0.2, max_extra_delay=9,
    ).compile(seed, opportunities)
    restored = SeededFaultSchedule.from_json(schedule.to_json())
    assert restored == schedule
    first = _faulted_catchup(schedule)
    assert any(
        r.kind == "send" and r.details["action"] in ("drop", "delay", "duplicate")
        for r in first[0]
    )
    assert first == _faulted_catchup(restored)
