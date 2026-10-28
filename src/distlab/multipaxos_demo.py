"""Run with python -m distlab.multipaxos_demo; all responses come from the API."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

from .kv import Put
from .linearizability import Get, SingleKeyKVLinearizabilityChecker
from .multipaxos import MultiPaxosCluster
from .multipaxos_client import ClientResult, ClientStatus, MultiPaxosClient
from .multipaxos_state_machine import MultiPaxosRequest, MultiPaxosStateMachine
from .simulator import Simulator


def run_demo() -> dict:
    nodes = ("n1", "n2", "n3")
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, nodes, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    client = MultiPaxosClient(machine)
    responses = []

    def observe(label: str, result: ClientResult) -> None:
        assert result.status in (
            ClientStatus.SUBMITTED,
            ClientStatus.APPLIED,
            ClientStatus.COMPLETED,
            ClientStatus.LEADERSHIP_UNAVAILABLE,
        )
        chosen = cluster.chosen(result.slot) if result.slot is not None else None
        responses.append(
            {
                "step": label,
                "client_id": result.request.client_id,
                "request_id": result.request.request_id,
                "operation_id": result.request.operation_id,
                "node": result.node_id,
                "slot": result.slot,
                "status": result.status.value,
                "ballot": asdict(result.ballot) if result.ballot is not None else None,
                "authority_ballot": (
                    asdict(result.authority_ballot) if result.authority_ballot is not None else None
                ),
                "value": result.value,
                "chosen": chosen is not None and chosen.value == result.request,
                "applied": any(d.value == result.request for d in machine.applied(result.node_id)),
            }
        )

    def settle(through: int) -> None:
        for _ in range(12):
            for node in nodes:
                machine.apply_ready(node)
            sim.run(max_events=256)
            if all(machine.last_applied(n) >= through for n in nodes):
                machine.assert_safety()
                return
        raise AssertionError("demo failed to converge within its event budget")

    cluster.node("n1").prepare_leadership()
    sim.run()
    a1 = MultiPaxosRequest("A", 1, "A-put-1", Put("k", "alpha"))
    b1 = MultiPaxosRequest("B", 1, "B-get-1", Get("k"))
    a2 = MultiPaxosRequest("A", 2, "A-put-2", Put("k", "beta"))
    observe("A writes", client.execute("n1", a1))
    observe("B reads alpha", client.execute("n1", b1))
    sim.partition(("n3",), ("n1", "n2"))
    ticket = client.submit("n1", a2)
    observe("A submits before crash", ticket)
    sim.run(max_events=256)
    observe("A applied but acknowledgement lost", client.poll("n1", a2, acknowledge=False))
    sim.crash("n1")
    observe("old leader cannot acknowledge", client.poll("n1", a2, expected_ballot=ticket.ballot))
    sim.heal_partition(("n3",), ("n1", "n2"))
    cluster.node("n2").prepare_leadership()
    sim.run()
    observe("same A request retried on n2", client.execute("n2", a2))
    sim.restart("n1")
    settle(3)
    observe("B reads beta", client.execute("n2", MultiPaxosRequest("B", 2, "B-get-2", Get("k"))))
    observe(
        "B interleaves write",
        client.execute(
            "n2",
            MultiPaxosRequest("B", 3, "B-put-3", Put("k", "gamma")),
        ),
    )
    cluster.node("n2").propose(6, a2)  # Chosen twice; the earlier effect must stay unique.
    sim.run()
    settle(6)
    observe("duplicate A cannot erase gamma", client.execute("n2", a2))
    observe(
        "B reads gamma",
        client.execute(
            "n2",
            MultiPaxosRequest("B", 4, "B-get-4", Get("k")),
        ),
    )
    settle(7)

    # Full runtime reconstruction retains server persistence and the separate
    # client-owned journal. Server KV/dedup/results are rebuilt from applied slots.
    original_trace = list(sim.trace)
    recovered = Simulator()
    recovered.persistent_state.update(copy.deepcopy(sim.persistent_state))
    sim = recovered
    cluster = MultiPaxosCluster(sim, nodes, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    client = MultiPaxosClient(machine, copy.deepcopy(client.journal))
    cluster.node("n2").prepare_leadership()
    sim.run()
    observe("retry after full reconstruction", client.execute("n2", a2))
    machine.assert_safety()
    verdict = SingleKeyKVLinearizabilityChecker().check(client.history)
    assert verdict.linearizable
    assert len(client.history.invocations()) == len(client.history.completed()) == 6
    assert all(dict(machine.state(n)) == {"k": "gamma"} for n in nodes)
    assert machine.client_requests("n1") == machine.client_requests("n2")
    assert machine.client_requests("n2") == machine.client_requests("n3")

    log = []
    for decision in cluster.chosen_slots():
        command = decision.value
        log.append(
            {
                "slot": decision.slot,
                "ballot": asdict(decision.proposal),
                "client_id": command.client_id,
                "request_id": command.request_id,
                "operation_id": command.operation_id,
                "chosen": True,
                "applied_on": [n for n in nodes if machine.last_applied(n) >= decision.slot],
                "first_effect_slot": machine.client_requests("n2")[command.identity].slot,
            }
        )
    trace = json.dumps([asdict(r) for r in (*original_trace, *sim.trace)], sort_keys=True)
    return {
        "responses": responses,
        "log": log,
        "replicas": {
            n: {
                "kv": dict(machine.state(n)),
                "applied_through": machine.last_applied(n),
                "dedup": [asdict(r) for r in machine.client_requests(n).values()],
            }
            for n in nodes
        },
        "history_invocations": len(client.history.invocations()),
        "history_completions": len(client.history.completed()),
        "linearizability": asdict(verdict),
        "full_reconstruction": True,
        "trace_sha256": hashlib.sha256(trace.encode()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
