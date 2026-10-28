"""Observed lost-Ack histories versus hypothetical completion-extension witnesses.

Run python -m distlab.pending_history_demo [scenario] for actual protocol evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict

from .kv import Put
from .linearizability import Get, SingleKeyKVLinearizabilityChecker
from .multipaxos import MultiPaxosCluster
from .multipaxos_client import ClientStatus, MultiPaxosClient
from .multipaxos_state_machine import MultiPaxosRequest, MultiPaxosStateMachine
from .randomized_faults import SeededFaultSchedule
from .simulator import Simulator

SCENARIOS = ("basic", "takeover", "restart", "catchup", "reconstruct", "retry", "unchosen")


def run_demo(scenario: str = "basic", faults: SeededFaultSchedule | None = None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    nodes = ("n1", "n2", "n3")
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, nodes, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    client = MultiPaxosClient(machine)
    cluster.node("n1").prepare_leadership()
    sim.run()
    if faults is not None:
        sim.fault_plan = faults.to_fault_plan()
    if scenario == "catchup":
        sim.partition(("n3",), ("n1", "n2"))
    write = MultiPaxosRequest("A", 1, "A-put", Put("k", "v"))
    for _ in range(8):
        client.submit("n1", write)
        sim.run(max_events=128)
        applied = client.poll("n1", write, acknowledge=False)
        if applied.status == ClientStatus.APPLIED:
            break
    else:
        raise AssertionError("lost-Ack write did not apply within the scenario budget")
    assert client.history.completed() == ()
    assert [i.operation_id for i in client.history.pending()] == ["A-put"]

    leader = "n1"
    old_trace = []
    unchosen = None
    if scenario == "takeover":
        leader = "n2"
        cluster.node(leader).prepare_leadership()
        sim.run(max_events=256)
    elif scenario == "restart":
        sim.crash("n1")
        sim.run(max_events=128)
        sim.restart("n1")
        leader = "n2"
        cluster.node(leader).prepare_leadership()
        sim.run(max_events=256)
    elif scenario == "catchup":
        assert cluster.node("n3").learned_decision(1) is None
        sim.heal_partition(("n3",), ("n1", "n2"))
    elif scenario == "reconstruct":
        old_trace = list(sim.trace)
        replacement = Simulator(fault_plan=sim.fault_plan)
        replacement.persistent_state.update(copy.deepcopy(sim.persistent_state))
        sim = replacement
        cluster = MultiPaxosCluster(sim, nodes, learn_decisions=True)
        machine = MultiPaxosStateMachine(cluster)
        client = MultiPaxosClient(machine, copy.deepcopy(client.journal))
        leader = "n2"
        cluster.node(leader).prepare_leadership()
        sim.run(max_events=256)
    elif scenario == "unchosen":
        sim.partition(("n1",), ("n2", "n3"))
        unchosen = MultiPaxosRequest("C", 1, "C-unchosen", Put("k", "unused"))
        assert client.submit("n1", unchosen).status == ClientStatus.SUBMITTED
        sim.run(max_events=128)  # Accepted only at isolated n1, without a quorum.
        assert cluster.chosen(2) is None
        leader = "n2"
        cluster.node(leader).prepare_leadership()
        sim.run(max_events=256)

    read = MultiPaxosRequest("B", 1, "B-get", Get("k"))
    answer = client.execute(leader, read, max_rounds=8, events_per_round=128)
    assert answer.status == ClientStatus.COMPLETED and answer.value == "v"
    if scenario == "catchup":
        for _ in range(8):
            for node in nodes:
                machine.apply_ready(node)
            sim.run(max_events=256)
            if machine.last_applied("n3") >= answer.slot:
                break
        assert machine.last_applied("n3") >= answer.slot
        assert machine.client_requests("n3")[write.identity].request == write
    machine.assert_safety()

    history = client.history
    snapshot = copy.deepcopy(vars(history))
    trace_before = list(sim.trace)
    checker = SingleKeyKVLinearizabilityChecker()
    legacy = checker.check(history)
    extension = checker.check_with_pending_extensions(history)
    assert not legacy.linearizable and extension.linearizable
    assert vars(history) == snapshot
    assert sim.trace == trace_before
    assert [c.operation_id for c in history.completed()] == ["B-get"]
    assert extension.order == ("A-put", "B-get")
    assert extension.included_pending == ("A-put",)
    if unchosen is not None:
        assert extension.omitted_pending == ("C-unchosen",)
        assert all(d.value != unchosen for d in cluster.chosen_slots())

    evidence = {
        "scenario": scenario,
        "observed": {
            "invocations": [asdict(i) for i in history.invocations()],
            "completions": [asdict(c.completion) for c in history.completed()],
            "pending": [i.operation_id for i in history.pending()],
        },
        "legacy_result": asdict(legacy),
        "hypothetical_extension": asdict(extension),
        "history_unchanged": vars(history) == snapshot,
        "protocol_evidence": {
            "write_slot": applied.slot,
            "write_ballot": asdict(applied.ballot),
            "write_status": applied.status.value,
            "write_chosen": cluster.chosen(applied.slot).value == write,
            "read_slot": answer.slot,
            "read_ballot": asdict(answer.ballot),
            "read_authority_ballot": asdict(answer.authority_ballot),
            "read_status": answer.status.value,
            "read_value": answer.value,
            "unchosen_request": unchosen.operation_id if unchosen is not None else None,
            "unchosen_has_quorum": (
                any(d.value == unchosen for d in cluster.chosen_slots())
                if unchosen is not None
                else None
            ),
            "unchosen_has_effect": (
                any(unchosen.identity in machine.client_requests(n) for n in nodes)
                if unchosen is not None
                else None
            ),
            "applied_through": {n: machine.last_applied(n) for n in nodes},
        },
    }
    if scenario == "retry":
        assert client.execute(leader, write).status == ClientStatus.COMPLETED
        actual = checker.check_with_pending_extensions(history)
        assert actual.included_pending == ()
        assert len(history.invocations()) == len(history.completed()) == 2
        evidence["after_actual_retry"] = {
            "observed_completions": [asdict(c.completion) for c in history.completed()],
            "extension": asdict(actual),
            "legacy_result": asdict(checker.check(history)),
        }
    trace = json.dumps([asdict(r) for r in (*old_trace, *sim.trace)], sort_keys=True)
    evidence["fault_actions"] = dict(
        sorted(
            Counter(
                r.details["action"]
                for r in (*old_trace, *sim.trace)
                if r.kind == "send" and r.details["action"] != "deliver"
            ).items()
        )
    )
    evidence["trace_sha256"] = hashlib.sha256(trace.encode()).hexdigest()
    return evidence


if __name__ == "__main__":
    print(json.dumps(run_demo(sys.argv[1] if len(sys.argv) > 1 else "basic"), indent=2))
