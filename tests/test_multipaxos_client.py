import copy

import pytest

from distlab.kv import ClientRequestConflict, Delete, Put
from distlab.linearizability import Get, SingleKeyKVLinearizabilityChecker
from distlab.multipaxos import LeaderDecision, MultiPaxosCluster
from distlab.multipaxos_client import ClientStatus, MultiPaxosClient
from distlab.multipaxos_demo import run_demo
from distlab.multipaxos_state_machine import Append, MultiPaxosRequest, MultiPaxosStateMachine
from distlab.paxos import PaxosSafetyViolation
from distlab.randomized_faults import FaultOpportunity, SeededFaultGenerator, SeededFaultSchedule
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator

NODES = ("n1", "n2", "n3")


def request(client, rid, operation):
    return MultiPaxosRequest(client, rid, f"{client}-{rid}", operation)


def ready():
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    client = MultiPaxosClient(machine)
    lead(sim, cluster, "n1")
    return sim, cluster, machine, client


def lead(sim, cluster, node):
    ballot = cluster.node(node).prepare_leadership()
    sim.run()
    assert cluster.node(node).has_leader_authority
    return ballot


def reconstruct(sim, client):
    new_sim = Simulator()
    new_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))
    cluster = MultiPaxosCluster(new_sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    rebuilt = MultiPaxosClient(machine, copy.deepcopy(client.journal))
    return new_sim, cluster, machine, rebuilt


def converge(sim, machine, through):
    for _ in range(12):
        for node in NODES:
            if sim.is_alive(node):
                machine.apply_ready(node)
        sim.run(max_events=256)
        machine.assert_safety()
        if all(machine.last_applied(n) >= through for n in NODES):
            assert len({tuple(machine.state(n).items()) for n in NODES}) == 1
            assert machine.client_requests("n1") == machine.client_requests("n2")
            assert machine.client_requests("n2") == machine.client_requests("n3")
            return
    pytest.fail("bounded catch-up did not converge")


def test_submit_chosen_applied_and_completion_are_distinct():
    sim, _, machine, client = ready()
    req = request("A", 1, Put("k", "v"))
    ticket = client.submit("n1", req)
    assert ticket.status == ClientStatus.SUBMITTED
    assert client.poll("n1", req, apply=False).status == ClientStatus.PENDING
    assert client.history.completed() == ()
    sim.run()
    assert client.poll("n1", req, apply=False).status == ClientStatus.CHOSEN
    assert machine.last_applied("n1") == 0
    assert machine.client_requests("n1") == {}
    assert client.poll("n1", req, acknowledge=False).status == ClientStatus.APPLIED
    assert client.history.completed() == ()
    done = client.poll("n1", req, expected_ballot=ticket.ballot)
    assert done.status == ClientStatus.COMPLETED
    assert done.slot == 1 and done.ballot == ticket.ballot
    assert len(client.history.completed()) == 1


def test_put_get_delete_get_produces_real_linearizable_history():
    _, _, machine, client = ready()
    operations = [Put("k", "v"), Get("k"), Delete("k"), Get("k")]
    answers = [
        client.execute("n1", request("A" if i % 2 == 0 else "B", i, op))
        for i, op in enumerate(operations)
    ]
    assert all(r.status == ClientStatus.COMPLETED for r in answers)
    assert [r.value for r in answers] == [None, "v", None, None]
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable
    assert dict(machine.state("n1")) == {}


def test_interleaved_put_retry_cannot_overwrite_a_later_write():
    _, cluster, machine, client = ready()
    original = request("A", 1, Put("k", "first"))
    assert client.execute("n1", original).status == ClientStatus.COMPLETED
    assert client.execute("n1", request("B", 1, Put("k", "later"))).status == ClientStatus.COMPLETED
    # Force the same client request into another chosen slot, bypassing transport
    # coalescing. A plain repeated Put would incorrectly erase B's later value.
    cluster.node("n1").propose(3, original)
    client.sim.run()
    machine.apply_ready("n1")
    assert dict(machine.state("n1")) == {"k": "later"}
    assert client.execute("n1", original).status == ClientStatus.COMPLETED
    assert len(client.history.invocations()) == len(client.history.completed()) == 2


def test_append_in_different_slots_duplicate_delivery_and_reconstruction_executes_once():
    sim, cluster, machine, client = ready()
    original = request("A", 1, Append("k", "x"))
    assert client.execute("n1", original).status == ClientStatus.COMPLETED
    sim.fault_plan = FaultPlan(
        (FaultRule(FaultAction.DUPLICATE, payload_type="LeaderDecision", extra_delay=3),)
    )
    cluster.node("n1").propose(2, Append("k", "y"))
    cluster.node("n1").propose(3, original)
    sim.run()
    converge(sim, machine, 3)
    assert dict(machine.state("n1")) == {"k": "xy"}
    sim, cluster, machine, client = reconstruct(sim, client)
    assert all(dict(machine.state(n)) == {"k": "xy"} for n in NODES)
    lead(sim, cluster, "n2")
    assert client.execute("n2", original).status == ClientStatus.COMPLETED
    assert dict(machine.state("n2")) == {"k": "xy"}
    assert client.history.invocations() == ()  # Append is not a checker operation.


@pytest.mark.parametrize(
    "different",
    [
        MultiPaxosRequest("A", 1, "A-1", Put("k", "changed")),
        MultiPaxosRequest("A", 1, "different-history", Put("k", "v")),
        MultiPaxosRequest("B", 1, "A-1", Put("k", "v")),
    ],
)
def test_identity_reuse_is_rejected_without_a_second_invocation(different):
    _, _, _, client = ready()
    assert client.execute("n1", request("A", 1, Put("k", "v"))).status == ClientStatus.COMPLETED
    before = client.history.invocations()
    assert client.submit("n1", different).status == ClientStatus.CONFLICT
    assert client.history.invocations() == before


def test_durable_conflict_fails_before_effect_or_applied_prefix_advances():
    sim, cluster, machine, client = ready()
    first = request("A", 1, Append("k", "x"))
    assert client.execute("n1", first).status == ClientStatus.COMPLETED
    prefix = machine.applied("n1")
    cluster.node("n1").propose(2, request("A", 1, Append("k", "BAD")))
    sim.run()
    with pytest.raises(ClientRequestConflict):
        machine.apply_ready("n1")
    assert machine.applied("n1") == prefix
    assert dict(machine.state("n1")) == {"k": "x"}


def test_get_result_is_captured_at_first_slot_not_completion_or_retry_time():
    sim, cluster, machine, client = ready()
    assert (
        client.execute("n1", request("A", 1, Put("k", "before"))).status == ClientStatus.COMPLETED
    )
    read = request("B", 1, Get("k"))
    client.submit("n1", read)
    sim.run()
    machine.apply_ready("n1")  # Read applied, response lost.
    assert client.execute("n1", request("A", 2, Put("k", "after"))).status == ClientStatus.COMPLETED
    cluster.node("n1").propose(4, read)
    sim.run()
    machine.apply_ready("n1")
    assert client.poll("n1", read).value == "before"
    sim, cluster, machine, client = reconstruct(sim, client)
    lead(sim, cluster, "n2")
    assert client.execute("n2", read).value == "before"
    assert dict(machine.state("n2")) == {"k": "after"}
    assert len(client.history.invocations()) == len(client.history.completed()) == 3
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_takeover_adoption_of_b_never_acknowledges_request_a_by_slot_number():
    sim, cluster, machine, client = ready()
    b = request("B", 1, Put("k", "B"))
    sim.partition(("n1",), ("n2", "n3"))
    client.submit("n1", b, slot=5)
    sim.run()
    sim.heal_partition(("n1",), ("n2", "n3"))
    lead(sim, cluster, "n2")
    a = request("A", 1, Put("k", "A"))
    client.submit("n2", a, slot=5)
    sim.run()
    assert cluster.node("n2").learned_decision(5).value == b
    assert client.poll("n2", a).status == ClientStatus.PENDING
    assert client.history.completed() == ()
    assert client.execute("n2", a).status == ClientStatus.COMPLETED
    assert client.poll("n2", b).status == ClientStatus.COMPLETED
    assert dict(machine.state("n2")) == {"k": "A"}
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_takeover_get_is_after_phase1_high_water_and_fills_accepted_none_holes():
    sim, cluster, machine, client = ready()
    sim.partition(("n3",), ("n1", "n2"))
    sim.fault_plan = FaultPlan(
        (FaultRule(FaultAction.DROP, src="n1", dst="n2", payload_type="LeaderDecision"),)
    )
    prior = request("A", 1, Put("k", "prior"))
    client.submit("n1", prior, slot=5)
    sim.run()
    assert client.execute("n1", prior).status == ClientStatus.COMPLETED
    sim.crash("n1")
    sim.fault_plan = FaultPlan()
    sim.heal_partition(("n3",), ("n1", "n2"))
    lead(sim, cluster, "n3")
    # n3 has neither accepted nor learned the completed write. Only n2's
    # transported Phase-1 promise reveals it. A local-prefix-only allocator
    # could put Get at slot 1 and incorrectly return None before slot 5.
    assert cluster.node("n3").accept_history == ()
    assert machine.last_applied("n3") == 0
    assert cluster.node("n3").learned_decisions == ()
    result = client.execute("n3", request("B", 1, Get("k")))
    assert result.status == ClientStatus.COMPLETED
    assert result.slot > 5 and result.value == "prior"
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_lost_ack_crash_retry_catchup_and_full_reconstruction():
    sim, cluster, machine, client = ready()
    first = request("A", 1, Put("k", "one"))
    assert client.execute("n1", first).status == ClientStatus.COMPLETED
    assert client.execute("n1", request("B", 1, Get("k"))).value == "one"
    sim.partition(("n3",), ("n1", "n2"))
    second = request("A", 2, Put("k", "two"))
    ticket = client.submit("n1", second)
    sim.run()
    assert client.poll("n1", second, acknowledge=False).status == ClientStatus.APPLIED
    sim.crash("n1")
    assert (
        client.poll("n1", second, expected_ballot=ticket.ballot).status
        == ClientStatus.LEADERSHIP_UNAVAILABLE
    )
    sim.heal_partition(("n3",), ("n1", "n2"))
    lead(sim, cluster, "n2")
    assert client.execute("n2", second).status == ClientStatus.COMPLETED
    sim.restart("n1")
    converge(sim, machine, ticket.slot)
    assert client.execute("n2", request("B", 2, Get("k"))).value == "two"
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable
    assert len(client.history.invocations()) == len(client.history.completed()) == 4
    sim, cluster, machine, client = reconstruct(sim, client)
    lead(sim, cluster, "n2")
    for _ in range(3):
        assert client.execute("n2", second).status == ClientStatus.COMPLETED
    assert all(dict(machine.state(n)) == {"k": "two"} for n in NODES)
    assert len(client.history.completed()) == 4
    machine.assert_safety()


def test_pending_request_survives_reconstruction_without_inventing_completion():
    sim, cluster, _, client = ready()
    sim.partition(("n1",), ("n2", "n3"))
    req = request("A", 1, Put("k", "v"))
    assert client.execute("n1", req, max_rounds=2).status == ClientStatus.QUORUM_UNAVAILABLE
    assert client.history.completed() == ()
    sim, cluster, _, client = reconstruct(sim, client)
    lead(sim, cluster, "n2")
    assert client.execute("n2", req).status == ClientStatus.COMPLETED
    assert len(client.history.invocations()) == 1


def test_lost_append_ack_takeover_rechooses_request_in_two_slots_but_effect_is_once():
    sim, cluster, machine, client = ready()
    sim.partition(("n3",), ("n1", "n2"))
    sim.fault_plan = FaultPlan(
        (FaultRule(FaultAction.DROP, src="n1", dst="n2", payload_type="LeaderDecision"),)
    )
    req = request("A", 1, Append("k", "x"))
    ticket = client.submit("n1", req)
    sim.run()
    assert client.poll("n1", req, acknowledge=False).status == ClientStatus.APPLIED
    assert cluster.node("n2").learned_decision(1) is None
    sim.crash("n1")
    sim.fault_plan = FaultPlan()
    sim.heal_partition(("n3",), ("n1", "n2"))
    lead(sim, cluster, "n2")
    result = client.execute("n2", req)
    assert result.status == ClientStatus.COMPLETED
    assert result.slot == 1
    assert result.authority_ballot == cluster.node("n2").leader_ballot
    assert result.ballot == machine.applied("n2")[0].proposal
    assert cluster.chosen(1).value == cluster.chosen(2).value == req
    assert (
        client.poll("n1", req, expected_ballot=ticket.ballot).status
        == ClientStatus.LEADERSHIP_UNAVAILABLE
    )
    sim.restart("n1")
    converge(sim, machine, 2)
    assert all(dict(machine.state(n)) == {"k": "x"} for n in NODES)
    assert all(machine.client_requests(n)[req.identity].slot == 1 for n in NODES)


def test_api_history_retains_an_incomplete_operation_without_fabricating_response():
    sim, _, _, client = ready()
    assert client.execute("n1", request("A", 1, Put("k", "known"))).status == ClientStatus.COMPLETED
    sim.partition(("n3",), ("n1", "n2"))
    pending = request("B", 1, Put("k", "unknown"))
    assert client.execute("n3", pending).status == ClientStatus.LEADERSHIP_UNAVAILABLE
    assert client.execute("n1", request("A", 2, Get("k"))).value == "known"
    assert len(client.history.invocations()) == 3
    assert len(client.history.completed()) == 2
    assert [i.operation_id for i in client.history.pending()] == [pending.operation_id]
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_gap_blocks_ack_until_peer_recovery_fills_missing_decision():
    sim, cluster, machine, client = ready()
    a = request("A", 1, Put("k", "v"))
    # Miss slot 1 at n1's learner while it remains a valid leader, using actual
    # decisions from a temporary n2 leader and a subsequent n1 Phase 1.
    lead(sim, cluster, "n2")
    sim.block_link("n2", "n1")
    client.submit("n2", a)
    sim.run()
    sim.heal_link("n2", "n1")
    lead(sim, cluster, "n1")
    b = request("B", 1, Get("k"))
    client.submit("n1", b)
    sim.run()
    assert machine.blocked_on("n1") == 1
    assert client.poll("n1", b).status == ClientStatus.CHOSEN
    assert client.history.completed() == ()
    sim.run()
    assert client.poll("n1", a).status == ClientStatus.COMPLETED
    assert client.poll("n1", b).value == "v"
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_follower_cannot_serve_stale_get_and_unknown_leadership_is_observable():
    sim, cluster, _, client = ready()
    sim.partition(("n3",), ("n1", "n2"))
    assert client.execute("n1", request("A", 1, Put("k", "fresh"))).status == ClientStatus.COMPLETED
    read = request("B", 1, Get("k"))
    assert client.execute("n3", read).status == ClientStatus.LEADERSHIP_UNAVAILABLE
    assert len(client.history.pending()) == 1
    sim.heal_partition(("n3",), ("n1", "n2"))
    lead(sim, cluster, "n3")
    assert client.execute("n3", read).value == "fresh"
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


def test_expired_ballot_ticket_and_forged_decision_cannot_complete():
    sim, cluster, machine, client = ready()
    req = request("A", 1, Put("k", "v"))
    ticket = client.submit("n1", req)
    sim.run()
    lead(sim, cluster, "n2")
    assert (
        client.poll("n1", req, expected_ballot=ticket.ballot).status
        == ClientStatus.LEADERSHIP_UNAVAILABLE
    )
    assert client.history.completed() == ()
    forged = request("B", 1, Put("k", "forged"))
    client.journal.register(forged)
    ballot = cluster.node("n2").leader_ballot
    sim.send("n2", "n2", LeaderDecision(2, ballot, forged))
    with pytest.raises(PaxosSafetyViolation, match="lacks matching durable quorum"):
        sim.run()
    assert client.poll("n2", forged).status == ClientStatus.PENDING
    assert machine.last_applied("n2") == 1
    assert client.history.completed() == ()


def test_overlapping_clients_have_one_history_invocation_per_request():
    sim, _, _, client = ready()
    a = request("A", 1, Put("k", "a"))
    b = request("B", 1, Put("k", "b"))
    client.submit("n1", a)
    client.submit("n1", b)
    client.submit("n1", a)
    sim.run()
    assert client.poll("n1", b).status == ClientStatus.COMPLETED
    assert client.poll("n1", a).status == ClientStatus.COMPLETED
    assert client.execute("n1", request("C", 1, Get("k"))).value == "b"
    assert len(client.history.invocations()) == 3
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable


@pytest.mark.parametrize("rid", [-1, True, "1", None])
def test_invalid_request_identifiers_are_rejected(rid):
    with pytest.raises(ValueError):
        MultiPaxosRequest("A", rid, "op", Put("k", "v"))


def test_negative_get_timing_mutation_is_detected_by_real_client_history(monkeypatch):
    sim, _, machine, client = ready()
    assert client.execute("n1", request("A", 1, Put("k", "v"))).status == ClientStatus.COMPLETED
    read = request("B", 1, Get("k"))
    client.submit("n1", read)
    sim.run()
    machine.apply_ready("n1")
    assert client.poll("n1", read).value == "v"
    assert client.execute("n1", request("A", 2, Put("k", "later"))).status == ClientStatus.COMPLETED
    # A plausible bad client reader returns current KV instead of the recorded
    # read-slot result. Mutate the service result source, not successful events.
    original = machine.client_requests
    from dataclasses import replace

    def bad_results(node):
        records = dict(original(node))
        records[read.identity] = replace(records[read.identity], result=machine.state(node)["k"])
        return records

    monkeypatch.setattr(machine, "client_requests", bad_results)
    with pytest.raises(AssertionError, match="reconstructed client result changed"):
        client.poll("n1", read)


def test_negative_dedup_mutation_produces_a_non_linearizable_client_history(monkeypatch):
    import distlab.multipaxos_state_machine as implementation

    sim, cluster, machine, client = ready()
    a = request("A", 1, Put("k", "a"))
    assert client.execute("n1", a).status == ClientStatus.COMPLETED
    assert client.execute("n1", request("B", 1, Put("k", "b"))).status == ClientStatus.COMPLETED
    original = implementation._apply_decision

    def bad_dedup(state, requests, decision):
        command = decision.value
        if isinstance(command, MultiPaxosRequest) and command.identity in requests:
            implementation.apply_multipaxos_command(state, command.operation)
        original(state, requests, decision)

    monkeypatch.setattr(implementation, "_apply_decision", bad_dedup)
    cluster.node("n1").propose(3, a)
    sim.run()
    machine.apply_ready("n1")
    result = client.execute("n1", request("B", 2, Get("k")))
    assert result.value == "a"  # Actual bad read produced by the mutated API path.
    assert not SingleKeyKVLinearizabilityChecker().check(client.history).linearizable
    assert len(client.history.invocations()) == 3


def test_client_recovers_multiple_windows_before_acknowledging_high_slot():
    sim, _, machine, client = ready()
    req = request("A", 1, Put("k", "high"))
    client.submit("n1", req, slot=40)
    sim.run()
    assert client.poll("n1", req).status == ClientStatus.CHOSEN
    assert client.history.completed() == ()
    result = client.execute("n1", req, max_rounds=12, events_per_round=128)
    assert result.status == ClientStatus.COMPLETED
    assert result.slot == 40
    assert machine.last_applied("n1") == 40
    assert [d.slot for d in machine.applied("n1")] == list(range(1, 41))
    assert dict(machine.state("n1")) == {"k": "high"}


def faulted_client_run(schedule):
    sim, _, machine, client = ready()
    sim.fault_plan = schedule.to_fault_plan()
    operations = [
        request("A", 1, Put("k", "a")),
        request("B", 1, Get("k")),
        request("A", 2, Put("k", "b")),
        request("B", 2, Get("k")),
    ]
    results = []
    for req in operations:
        result = client.execute("n1", req)
        if result.status != ClientStatus.COMPLETED:
            sim.fault_plan = FaultPlan()
            result = client.execute("n1", req)
        assert result.status == ClientStatus.COMPLETED
        results.append(result)
    assert SingleKeyKVLinearizabilityChecker().check(client.history).linearizable
    converge(sim, machine, results[-1].slot)
    assert any(r.kind == "send" and r.details["action"] != "deliver" for r in sim.trace)
    return sim.trace, results, client.history.completed(), dict(machine.state("n1"))


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_fixed_seed_client_fault_schedules_replay_exactly(seed):
    opportunities = tuple(
        FaultOpportunity(src, dst, ordinal, payload)
        for src in NODES
        for dst in NODES
        if src != dst
        for ordinal in range(1, 80)
        for payload in (
            "LeaderAccept",
            "LeaderAccepted",
            "LeaderDecision",
            "DecisionRecoveryRequest",
            "DecisionRecoveryResponse",
        )
    )
    schedule = SeededFaultGenerator(
        drop_rate=0.03,
        delay_rate=0.1,
        duplicate_rate=0.1,
        max_extra_delay=3,
    ).compile(seed, opportunities)
    replay = SeededFaultSchedule.from_json(schedule.to_json())
    assert faulted_client_run(schedule) == faulted_client_run(replay)


def test_executable_two_client_crash_catchup_demo_replays_exactly():
    first = run_demo()
    assert first == run_demo()
    assert first["linearizability"]["linearizable"]
    assert first["history_invocations"] == first["history_completions"] == 6
    assert [r["slot"] for r in first["log"]] == list(range(1, 8))
    assert first["log"][5]["first_effect_slot"] == 3
    retry = next(r for r in first["responses"] if r["step"] == "same A request retried on n2")
    assert retry["ballot"]["round"] == 1
    assert retry["authority_ballot"]["round"] == 2
    assert all(r["kv"] == {"k": "gamma"} for r in first["replicas"].values())
