import copy
import itertools

import pytest

from distlab.kv import Delete, Put
from distlab.multipaxos import LeaderDecision, MultiPaxosCluster
from distlab.multipaxos_state_machine import Append, MultiPaxosStateMachine
from distlab.paxos import PaxosError, PaxosSafetyViolation, ProposalNumber
from distlab.paxos_log import SlotLearnedDecision
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator

NODES = ("n1", "n2", "n3")


def _cluster(
    fault_plan: FaultPlan | None = None,
) -> tuple[Simulator, MultiPaxosCluster, MultiPaxosStateMachine]:
    sim = Simulator(fault_plan=fault_plan)
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    return sim, cluster, MultiPaxosStateMachine(cluster)


def _reconstruct(
    sim: Simulator,
) -> tuple[Simulator, MultiPaxosCluster, MultiPaxosStateMachine]:
    recovered_sim = Simulator()
    recovered_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))
    cluster = MultiPaxosCluster(recovered_sim, NODES, learn_decisions=True)
    return recovered_sim, cluster, MultiPaxosStateMachine(cluster)


def _lead(sim: Simulator, cluster: MultiPaxosCluster, node_id: str = "n1") -> ProposalNumber:
    ballot = cluster.node(node_id).prepare_leadership()
    sim.run()
    assert cluster.node(node_id).has_leader_authority
    return ballot


def _slots(decisions: tuple[SlotLearnedDecision, ...]) -> list[tuple[int, object]]:
    return [(decision.slot, decision.value) for decision in decisions]


def _applied_trace(sim: Simulator, node_id: str) -> list[int]:
    return [
        record.details["slot"]
        for record in sim.trace
        if record.kind == "multipaxos-apply" and record.details["node"] == node_id
    ]


def test_chosen_slots_apply_in_contiguous_order_on_every_node() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)

    leader.propose(3, Put("k", "c"))
    leader.propose(1, Put("k", "a"))
    leader.propose(2, Delete("k"))
    sim.run()

    for node_id in NODES:
        applied = machine.apply_ready(node_id)
        assert [decision.slot for decision in applied] == [1, 2, 3]
        assert machine.last_applied(node_id) == 3
        assert dict(machine.state(node_id)) == {"k": "c"}
        assert _applied_trace(sim, node_id) == [1, 2, 3]
    machine.assert_safety()


def test_out_of_order_decision_waits_for_earlier_slot() -> None:
    # n1 -> n2 ordinals: 1 prepare, 2 accept(1), 3 accept(2), 4 decision(1), 5 decision(2).
    delay_slot_one = FaultRule(
        FaultAction.DELAY,
        src="n1",
        dst="n2",
        ordinal=4,
        payload_type="LeaderDecision",
        extra_delay=5,
    )
    sim, cluster, machine = _cluster(FaultPlan((delay_slot_one,)))
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Put("k", "first"))
    leader.propose(2, Put("k", "second"))

    follower = cluster.node("n2")
    while follower.learned_decision(2) is None:
        assert sim.run(max_events=1) == 1
    assert follower.learned_decision(1) is None

    assert machine.apply_ready("n2") == ()
    assert machine.blocked_on("n2") == 1
    assert machine.last_applied("n2") == 0
    assert dict(machine.state("n2")) == {}
    gap = [record for record in sim.trace if record.kind == "multipaxos-apply-gap"]
    assert gap[-1].details == {"node": "n2", "missing_slot": 1, "learned_beyond": (2,)}

    sim.run()
    assert _slots(machine.apply_ready("n2")) == [
        (1, Put("k", "first")),
        (2, Put("k", "second")),
    ]
    assert dict(machine.state("n2")) == {"k": "second"}
    machine.assert_safety()


def test_missing_slot_gap_blocks_later_slots_until_filled() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)

    sim.block_link("n1", "n2")
    leader.propose(1, Put("a", "1"))
    sim.run()
    sim.heal_link("n1", "n2")
    leader.propose(2, Put("b", "2"))
    leader.propose(3, Put("c", "3"))
    sim.run()

    assert cluster.chosen(1) is not None
    assert cluster.node("n2").learned_decision(1) is None
    for _ in range(2):
        assert machine.apply_ready("n2") == ()
    assert machine.blocked_on("n2") == 1
    assert machine.applied("n2") == ()
    for node_id in ("n1", "n3"):
        assert [decision.slot for decision in machine.apply_ready(node_id)] == [1, 2, 3]

    # The stable leader re-proposes the identical chosen value to refill the gap.
    leader.propose(1, Put("a", "1"))
    sim.run()
    assert [decision.slot for decision in machine.apply_ready("n2")] == [1, 2, 3]
    assert machine.blocked_on("n2") is None
    assert dict(machine.state("n2")) == {"a": "1", "b": "2", "c": "3"}
    machine.assert_safety()


def test_minority_accepted_slot_is_never_learned_or_applied() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)

    sim.partition(("n1",), ("n2", "n3"))
    leader.propose(1, Put("k", "minority"))
    sim.run()

    assert cluster.chosen(1) is None
    for node_id in NODES:
        assert cluster.node(node_id).learned_decisions == ()
        assert machine.apply_ready(node_id) == ()
        assert dict(machine.state(node_id)) == {}
    machine.assert_safety()


def test_forged_decision_for_unchosen_slot_is_never_applied() -> None:
    sim, cluster, machine = _cluster()
    ballot = _lead(sim, cluster)

    # A member-authenticated envelope is insufficient: reject before persisting
    # a learned record when no durable quorum chose the value at this ballot.
    sim.send("n1", "n2", LeaderDecision(1, ballot, Put("k", "forged")))
    with pytest.raises(PaxosSafetyViolation, match="lacks matching durable quorum"):
        sim.run()
    assert cluster.node("n2").learned_decision(1) is None
    assert machine.apply_ready("n2") == ()
    assert machine.applied("n2") == ()
    cluster.assert_safety()


def test_non_member_and_mismatched_decision_envelopes_are_ignored() -> None:
    sim, cluster, machine = _cluster()
    ballot = _lead(sim, cluster)

    sim.send("outsider", "n2", LeaderDecision(1, ProposalNumber(9, "outsider"), "x"))
    sim.send("n3", "n2", LeaderDecision(1, ballot, "spoofed-src"))
    sim.send("n1", "n2", LeaderDecision(0, ballot, "invalid-slot"))
    sim.run()

    assert cluster.node("n2").learned_decisions == ()
    assert machine.apply_ready("n2") == ()
    machine.assert_safety()


def test_duplicate_decision_deliveries_apply_each_slot_once() -> None:
    duplicate = FaultRule(
        FaultAction.DUPLICATE, payload_type="LeaderDecision", extra_delay=20
    )
    sim, cluster, machine = _cluster(FaultPlan((duplicate,)))
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Append("k", "a"))
    leader.propose(2, Append("k", "b"))

    # Apply after the originals land, then let every duplicate arrive.
    while any(cluster.node(n).learned_decision(2) is None for n in NODES):
        assert sim.run(max_events=1) == 1
    for node_id in NODES:
        assert [decision.slot for decision in machine.apply_ready(node_id)] == [1, 2]
        assert dict(machine.state(node_id)) == {"k": "ab"}
    sim.run()

    duplicates = [r for r in sim.trace if r.kind == "multipaxos-decision-duplicate"]
    assert sorted((r.details["node"], r.details["slot"]) for r in duplicates) == [
        ("n2", 1),
        ("n2", 2),
        ("n3", 1),
        ("n3", 2),
    ]
    for node_id in NODES:
        assert machine.apply_ready(node_id) == ()
        assert machine.apply_ready(node_id) == ()
        assert machine.last_applied(node_id) == 2
        assert _applied_trace(sim, node_id) == [1, 2]
        # Append is non-idempotent: any second application would show as "aabb".
        assert dict(machine.state(node_id)) == {"k": "ab"}
    machine.assert_safety()


def test_crash_before_application_resumes_from_durable_learned_decisions() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Put("k", "a"))
    leader.propose(2, Put("j", "b"))
    sim.run()
    assert [d.slot for d in cluster.node("n2").learned_decisions] == [1, 2]

    sim.crash("n2")
    with pytest.raises(PaxosError, match="crashed node"):
        machine.apply_ready("n2")
    sim.restart("n2")

    assert machine.last_applied("n2") == 0
    assert [decision.slot for decision in machine.apply_ready("n2")] == [1, 2]
    assert dict(machine.state("n2")) == {"k": "a", "j": "b"}
    machine.assert_safety()


def test_decision_lost_while_crashed_leaves_a_gap_not_a_skip() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    sim.crash("n3")
    leader.propose(1, Put("k", "a"))
    sim.run()
    sim.restart("n3")
    leader.propose(2, Put("k", "b"))
    sim.run()

    assert cluster.node("n3").learned_decision(1) is None
    assert machine.apply_ready("n3") == ()
    assert machine.blocked_on("n3") == 1
    assert dict(machine.state("n3")) == {}
    machine.assert_safety()


def test_crash_after_application_rebuilds_state_without_reapplying() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Append("k", "a"))
    leader.propose(2, Append("k", "b"))
    sim.run()
    assert len(machine.apply_ready("n2")) == 2
    before = dict(machine.state("n2"))
    assert before == {"k": "ab"}

    sim.crash("n2")
    assert "multipaxos_state_machine" not in sim.volatile_state["n2"]
    sim.restart("n2")

    assert machine.last_applied("n2") == 2
    assert dict(machine.state("n2")) == before
    loads = [
        r.details
        for r in sim.trace
        if r.kind == "multipaxos-state-machine-load" and r.details["node"] == "n2"
    ]
    assert loads[-1] == {"node": "n2", "last_applied": 2}
    assert machine.apply_ready("n2") == ()
    assert dict(machine.state("n2")) == {"k": "ab"}

    leader.propose(3, Append("k", "c"))
    sim.run()
    assert [decision.slot for decision in machine.apply_ready("n2")] == [3]
    assert _applied_trace(sim, "n2") == [1, 2, 3]
    assert dict(machine.state("n2")) == {"k": "abc"}
    machine.assert_safety()


def test_full_reconstruction_recovers_applied_prefix_and_continues() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Append("k", "a"))
    leader.propose(2, Append("k", "b"))
    sim.run()
    machine.apply_ready("n1")
    machine.apply_ready("n2")
    # n3 learned both slots but crashed before applying them.

    recovered_sim, recovered, recovered_machine = _reconstruct(sim)

    for node_id in ("n1", "n2"):
        assert recovered_machine.applied(node_id) == machine.applied(node_id)
        assert dict(recovered_machine.state(node_id)) == {"k": "ab"}
        assert recovered_machine.apply_ready(node_id) == ()
        assert dict(recovered_machine.state(node_id)) == {"k": "ab"}
    assert recovered_machine.last_applied("n3") == 0
    assert [d.slot for d in recovered_machine.apply_ready("n3")] == [1, 2]
    assert dict(recovered_machine.state("n3")) == {"k": "ab"}

    new_leader = recovered.node("n2")
    new_leader.prepare_leadership()
    recovered_sim.run()
    new_leader.propose(3, Append("k", "c"))
    recovered_sim.run()
    for node_id in NODES:
        assert recovered_machine.apply_ready(node_id)[-1].slot == 3
        assert dict(recovered_machine.state(node_id)) == {"k": "abc"}
    recovered_machine.assert_safety()


def test_leader_handoff_preserves_applied_prefix_and_fills_follower_gap() -> None:
    sim, cluster, machine = _cluster()
    old = cluster.node("n1")
    old_ballot = _lead(sim, cluster)
    old.propose(1, Append("k", "a"))
    sim.run()
    sim.block_link("n1", "n3")
    old.propose(2, Append("k", "b"))
    sim.run()
    assert cluster.chosen(2).value == Append("k", "b")
    assert [d.slot for d in machine.apply_ready("n1")] == [1, 2]
    assert [d.slot for d in machine.apply_ready("n3")] == [1]

    sim.crash("n1")
    sim.heal_link("n1", "n3")
    new = cluster.node("n2")
    new_ballot = _lead(sim, cluster, "n2")
    assert new_ballot > old_ballot
    new.propose(2, Append("k", "hijack"))  # Phase 1 must adopt the chosen value.
    new.propose(3, Append("k", "c"))
    sim.run()

    assert cluster.node("n3").learned_decision(2).value == Append("k", "b")
    # The application pass requested catch-up; n2 supplies its durable old-ballot
    # decision during Phase 1, before the new leader re-proposes the slot.
    assert cluster.node("n3").learned_decision(2).proposal == old_ballot
    assert [d.slot for d in machine.apply_ready("n3")] == [2, 3]
    assert [d.slot for d in machine.apply_ready("n2")] == [1, 2, 3]

    sim.restart("n1")
    assert machine.last_applied("n1") == 2
    assert machine.apply_ready("n1") == ()
    # Recovery and re-proposal both delivered slot 2; every node still holds one "b".
    assert dict(machine.state("n1")) == {"k": "ab"}
    for node_id in ("n2", "n3"):
        assert dict(machine.state(node_id)) == {"k": "abc"}
    assert sum(r.kind == "multipaxos-phase1-start" for r in sim.trace) == 2
    machine.assert_safety()


def test_learning_preserves_phase1_amortization() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    for slot in (1, 2, 3):
        leader.propose(slot, Put(f"k{slot}", "v"))
    sim.run()

    assert sum(r.kind == "multipaxos-phase1-start" for r in sim.trace) == 1
    assert sum(r.kind == "multipaxos-phase2-start" for r in sim.trace) == 3
    assert sum(r.kind == "multipaxos-learned" for r in sim.trace) == 9
    for node_id in NODES:
        assert machine.last_applied(node_id) == 0
        assert len(machine.apply_ready(node_id)) == 3
    machine.assert_safety()


def test_learning_is_opt_in_and_default_protocol_sends_no_decisions() -> None:
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES)
    leader = cluster.node("n1")
    leader.prepare_leadership()
    sim.run()
    leader.propose(1, "alpha")
    sim.run()

    assert cluster.chosen(1).value == "alpha"
    assert not any(
        isinstance(r.details.get("payload"), LeaderDecision) for r in sim.trace
    )
    assert all("multipaxos_learned" not in sim.persistent_state[n] for n in NODES)
    with pytest.raises(ValueError, match="learn_decisions=True"):
        MultiPaxosStateMachine(cluster)


def test_conflicting_runtime_decision_is_rejected() -> None:
    sim, cluster, _ = _cluster()
    leader = cluster.node("n1")
    ballot = _lead(sim, cluster)
    leader.propose(1, Put("k", "a"))
    sim.run()

    sim.send("n1", "n2", LeaderDecision(1, ballot, Put("k", "other")))
    with pytest.raises(PaxosSafetyViolation, match="conflicting decisions for slot 1"):
        sim.run()
    assert cluster.node("n2").learned_decision(1).value == Put("k", "a")


def _applied_run() -> Simulator:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Put("k", "a"))
    leader.propose(2, Put("k", "b"))
    sim.run()
    for node_id in NODES:
        machine.apply_ready(node_id)
    return sim


def test_reconstruction_rejects_applied_value_that_was_not_chosen() -> None:
    sim = _applied_run()
    applied = sim.persistent_state["n2"]["multipaxos_applied"]
    forged = SlotLearnedDecision(1, applied[0].proposal, Put("k", "forged"))
    sim.persistent_state["n2"]["multipaxos_applied"] = (forged, applied[1])

    with pytest.raises(PaxosSafetyViolation, match="without a matching durable learned"):
        _reconstruct(sim)


def test_reconstruction_rejects_learned_and_applied_value_without_quorum() -> None:
    sim = _applied_run()
    applied = sim.persistent_state["n2"]["multipaxos_applied"]
    forged = SlotLearnedDecision(2, applied[1].proposal, Put("k", "forged"))
    sim.persistent_state["n2"]["multipaxos_applied"] = (applied[0], forged)
    sim.persistent_state["n2"]["multipaxos_learned"][2] = forged

    with pytest.raises(PaxosSafetyViolation, match="without matching durable quorum"):
        _reconstruct(sim)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda applied: (applied[0], applied[0]), "applies slot 1 twice"),
        (lambda applied: (applied[1],), "skips slot 1"),
        (lambda applied: [*applied], "is invalid"),
    ),
)
def test_reconstruction_rejects_non_contiguous_applied_history(mutate, message) -> None:
    sim = _applied_run()
    applied = sim.persistent_state["n3"]["multipaxos_applied"]
    sim.persistent_state["n3"]["multipaxos_applied"] = mutate(applied)

    with pytest.raises(PaxosSafetyViolation, match=message):
        _reconstruct(sim)


def test_reconstruction_rejects_applied_slot_beyond_durable_learned() -> None:
    sim = _applied_run()
    del sim.persistent_state["n3"]["multipaxos_learned"][2]

    with pytest.raises(PaxosSafetyViolation, match="without a matching durable learned"):
        _reconstruct(sim)


@pytest.mark.parametrize(
    ("learned", "message"),
    (
        ({1: SlotLearnedDecision(2, ProposalNumber(1, "n1"), "x")}, "slot 1 is invalid"),
        ({1: SlotLearnedDecision(1, ProposalNumber(1, "outsider"), "x")}, "non-member"),
        ((), "learned decisions are invalid"),
    ),
)
def test_reconstruction_rejects_malformed_durable_learned_state(learned, message) -> None:
    sim = _applied_run()
    sim.persistent_state["n2"]["multipaxos_learned"] = learned

    with pytest.raises(PaxosSafetyViolation, match=message):
        _reconstruct(sim)


def test_safety_check_rejects_durable_prefix_changed_after_application() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Put("k", "a"))
    sim.run()
    machine.apply_ready("n2")

    sim.persistent_state["n2"]["multipaxos_applied"] = ()
    with pytest.raises(PaxosSafetyViolation, match="diverged from its durable applied"):
        machine.assert_safety()
    with pytest.raises(PaxosSafetyViolation, match="diverged from its durable applied"):
        machine.apply_ready("n2")


def test_safety_check_rejects_volatile_kv_content_divergence() -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    leader.propose(1, Append("k", "a"))
    leader.propose(2, Append("k", "b"))
    sim.run()
    machine.apply_ready("n2")
    assert dict(machine.state("n2")) == {"k": "ab"}
    machine.assert_safety()

    volatile = sim.volatile_state["n2"]["multipaxos_state_machine"]
    volatile["kv"]["k"] = "aab"
    assert volatile["applied_through"] == machine.last_applied("n2") == 2
    with pytest.raises(
        PaxosSafetyViolation,
        match="state machine for 'n2' diverged from its durable applied prefix",
    ):
        machine.assert_safety()


def _faulted_scenario() -> tuple[list, dict[str, tuple], dict[str, dict[str, str]], Simulator]:
    rules = (
        FaultRule(FaultAction.DUPLICATE, payload_type="LeaderDecision", extra_delay=3),
        FaultRule(FaultAction.DELAY, src="n1", dst="n3", extra_delay=4),
    )
    sim, cluster, machine = _cluster(FaultPlan(rules))
    old = cluster.node("n1")
    _lead(sim, cluster)
    old.propose(2, Put("k", "b"))
    old.propose(1, Put("k", "a"))
    sim.run(max_events=6)
    for node_id in NODES:
        machine.apply_ready(node_id)
    sim.crash("n1")
    sim.run()
    new = cluster.node("n2")
    _lead(sim, cluster, "n2")
    new.propose(1, "adopt")
    new.propose(2, "adopt")
    new.propose(3, Delete("k"))
    sim.run()
    sim.restart("n1")
    for node_id in NODES:
        machine.apply_ready(node_id)
    machine.assert_safety()
    applied = {node_id: machine.applied(node_id) for node_id in NODES}
    states = {node_id: dict(machine.state(node_id)) for node_id in NODES}
    return list(sim.trace), applied, states, sim


def test_faulted_scenario_replays_to_identical_trace_and_state() -> None:
    first_trace, first_applied, first_states, sim = _faulted_scenario()
    second_trace, second_applied, second_states, _ = _faulted_scenario()

    assert first_trace == second_trace
    assert first_applied == second_applied
    assert first_states == second_states
    assert _slots(first_applied["n2"]) == [
        (1, Put("k", "a")),
        (2, Put("k", "b")),
        (3, Delete("k")),
    ]

    # Replaying each durable prefix from empty state reproduces live state, and
    # a full reconstruction rebuilds the same machines from persistence alone.
    _, _, recovered = _reconstruct(sim)
    for node_id in NODES:
        assert MultiPaxosStateMachine.replay(first_applied[node_id]) == first_states[node_id]
        assert dict(recovered.state(node_id)) == first_states[node_id]


@pytest.mark.parametrize("order", tuple(itertools.permutations((1, 2, 3, 4))))
def test_every_proposal_order_applies_only_a_growing_contiguous_prefix(order) -> None:
    sim, cluster, machine = _cluster()
    leader = cluster.node("n1")
    _lead(sim, cluster)
    for slot in order:
        leader.propose(slot, Put("k", str(slot)))
    sim.crash("n3")

    progress = dict.fromkeys(NODES, 0)
    # Application passes now initiate bounded recovery episodes. Bound the driver
    # as well: continuously polling an unknown tail intentionally retries it.
    for step in range(256):
        for node_id in NODES:
            if sim.is_alive(node_id):
                machine.apply_ready(node_id)
            applied = machine.applied(node_id)
            assert [d.slot for d in applied] == list(range(1, len(applied) + 1))
            assert len(applied) >= progress[node_id]
            progress[node_id] = len(applied)
        if step == 4:
            sim.restart("n3")
        if all(machine.last_applied(node_id) == 4 for node_id in NODES):
            break
        sim.run(max_events=1)
    else:
        pytest.fail("bounded application/recovery driver did not converge")

    for node_id in NODES:
        assert machine.last_applied(node_id) == 4
        assert dict(machine.state(node_id)) == {"k": "4"}
        assert _applied_trace(sim, node_id) == [1, 2, 3, 4]
    # n3 recovers decisions missed while crashed without skipping its learned prefix.
    learned = {d.slot for d in cluster.node("n3").learned_decisions}
    contiguous = 0
    while contiguous + 1 in learned:
        contiguous += 1
    assert machine.last_applied("n3") == contiguous
    assert all(cluster.chosen(d.slot).value == d.value for d in machine.applied("n3"))
    machine.assert_safety()
