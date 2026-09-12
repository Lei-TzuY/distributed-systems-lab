from __future__ import annotations

from dataclasses import dataclass

from distlab import FaultAction, FaultPlan, FaultRule, ScenarioAction, Simulator
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness


@dataclass(frozen=True, slots=True)
class Request:
    value: str


@dataclass(frozen=True, slots=True)
class Response:
    value: str


def _record(sim: Simulator, message) -> None:
    sim.volatile_state[message.dst].setdefault("received", []).append(message.payload)


def test_fault_rule_can_target_one_protocol_message_type() -> None:
    plan = FaultPlan((FaultRule(FaultAction.DROP, src="a", dst="b", payload_type="Response"),))
    sim = Simulator(fault_plan=plan)
    sim.register("b", _record)

    sim.send("a", "b", Request("request"))
    sim.send("a", "b", Response("response"))
    sim.send("a", "b", Request("followup"))
    sim.run()

    assert sim.volatile_state["b"]["received"] == [Request("request"), Request("followup")]
    assert [record.kind for record in sim.trace].count("drop") == 1


def test_protocol_type_fault_plan_replays_exact_trace() -> None:
    actions = (
        ScenarioAction.send("a", "b", Request("first")),
        ScenarioAction.send("a", "b", Response("drop-me")),
        ScenarioAction.send("a", "b", Request("last")),
    )
    plan = FaultPlan((FaultRule(FaultAction.DROP, payload_type="Response"),))

    traces = []
    for _ in range(2):
        sim = Simulator(fault_plan=plan)
        sim.register("b", lambda _sim, _message: None)
        traces.append(sim.run_scenario(actions))

    assert traces[0] == traces[1]
    assert [record.kind for record in traces[0]].count("drop") == 1


def test_selective_vote_response_loss_blocks_election_without_blocking_link() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DROP,
                    dst="n1",
                    payload_type="RequestVoteResponse",
                ),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    safety = RaftSafetyHarness(cluster)
    n1 = cluster.node("n1")

    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.CANDIDATE
    assert cluster.leaders_by_term == {}
    vote_drops = [record for record in sim.trace if record.kind == "drop"]
    assert len(vote_drops) == 2
    assert all(type(record.details["payload"]).__name__ == "RequestVoteResponse" for record in vote_drops)
    safety.checkpoint()

    sim.fault_plan = FaultPlan()
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.LEADER
    assert cluster.leaders_by_term == {2: "n1"}
    safety.checkpoint()


def test_fault_rule_rejects_empty_protocol_type_selector() -> None:
    try:
        FaultRule(FaultAction.DROP, payload_type="")
    except ValueError as exc:
        assert "payload_type must be non-empty" in str(exc)
    else:
        raise AssertionError("empty payload_type selector must be rejected")
