from __future__ import annotations

import pytest

from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import (
    FaultAction,
    FaultPlan,
    FaultRule,
    Message,
    ScenarioAction,
    Simulator,
)


def _record_payloads(sim: Simulator, message: Message) -> None:
    sim.volatile_state[message.dst].setdefault("received", []).append(message.payload)


def _registered_simulator(*, fault_plan: FaultPlan | None = None) -> Simulator:
    sim = Simulator(fault_plan=fault_plan)
    for node in ("a", "b"):
        sim.register(node, _record_payloads)
    return sim


def test_directional_link_delay_only_slows_selected_direction_until_cleared() -> None:
    sim = _registered_simulator()

    sim.set_link_delay("a", "b", 4)
    sim.send("a", "b", "slow")
    sim.send("b", "a", "reverse")
    sim.run()

    deliveries = [record for record in sim.trace if record.kind == "deliver"]
    assert [(record.time, record.details["payload"]) for record in deliveries] == [
        (1, "reverse"),
        (5, "slow"),
    ]

    sim.clear_link_delay("a", "b")
    sim.send("a", "b", "normal")
    sim.run()

    assert sim.trace[-1].kind == "deliver"
    assert sim.trace[-1].time == 6
    assert sim.trace[-1].details["payload"] == "normal"


def test_directional_link_delay_composes_with_per_message_delay_rule() -> None:
    plan = FaultPlan(
        (
            FaultRule(
                FaultAction.DELAY,
                src="a",
                dst="b",
                ordinal=1,
                extra_delay=3,
            ),
        )
    )
    sim = _registered_simulator(fault_plan=plan)

    sim.set_link_delay("a", "b", 4)
    sim.send("a", "b", "combined", delay=2)
    sim.run()

    delivery = next(record for record in sim.trace if record.kind == "deliver")
    assert delivery.time == 9
    assert delivery.details["payload"] == "combined"


def test_directional_link_delay_scenario_replays_exact_trace() -> None:
    actions = (
        ScenarioAction.set_link_delay("a", "b", extra_delay=4),
        ScenarioAction.send("a", "b", "slow"),
        ScenarioAction.send("b", "a", "reverse"),
        ScenarioAction.run(),
        ScenarioAction.clear_link_delay("a", "b"),
        ScenarioAction.send("a", "b", "normal"),
    )

    first = _registered_simulator().run_scenario(actions)
    second = _registered_simulator().run_scenario(actions)

    assert first == second
    assert [record.kind for record in first].count("set-link-delay") == 1
    assert [record.kind for record in first].count("clear-link-delay") == 1


def test_delayed_vote_responses_defer_majority_without_breaking_raft_safety() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    safety = RaftSafetyHarness(cluster)
    n1 = cluster.node("n1")

    sim.set_link_delay("n2", "n1", 10)
    sim.set_link_delay("n3", "n1", 10)
    n1.start_election()
    sim.run(max_events=2)

    assert n1.role is RaftRole.CANDIDATE
    assert cluster.leaders_by_term == {}
    safety.checkpoint()

    sim.clear_link_delay("n2", "n1")
    sim.clear_link_delay("n3", "n1")
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.LEADER
    assert cluster.leaders_by_term == {2: "n1"}
    safety.checkpoint()


def test_directional_link_delay_rejects_invalid_configuration() -> None:
    sim = _registered_simulator()

    for src, dst, extra_delay in (
        ("a", "a", 1),
        ("a", "missing", 1),
        ("missing", "b", 1),
        ("a", "b", 0),
        ("a", "b", -1),
    ):
        with pytest.raises(ValueError):
            sim.set_link_delay(src, dst, extra_delay)
