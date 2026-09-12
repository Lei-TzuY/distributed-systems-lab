from __future__ import annotations

from distlab.raft import RaftCluster, RaftRole
from distlab.simulator import Message, ScenarioAction, Simulator


def _record_payloads(sim: Simulator, message: Message) -> None:
    sim.volatile_state[message.dst].setdefault("received", []).append(message.payload)


def _registered_simulator() -> Simulator:
    sim = Simulator()
    for node in ("a", "b", "c"):
        sim.register(node, _record_payloads)
    return sim


def test_partition_drops_cross_group_messages_until_healed() -> None:
    sim = _registered_simulator()

    sim.partition(("a",), ("b", "c"))
    sim.send("a", "b", "blocked-a-b")
    sim.send("c", "a", "blocked-c-a")
    sim.send("b", "c", "same-side")
    sim.run()

    assert sim.volatile_state["a"].get("received", []) == []
    assert sim.volatile_state["b"].get("received", []) == []
    assert sim.volatile_state["c"]["received"] == ["same-side"]
    assert [record.kind for record in sim.trace].count("partition-drop") == 2

    sim.heal_partition(("a",), ("b", "c"))
    sim.send("a", "b", "healed-a-b")
    sim.send("c", "a", "healed-c-a")
    sim.run()

    assert sim.volatile_state["a"]["received"] == ["healed-c-a"]
    assert sim.volatile_state["b"]["received"] == ["healed-a-b"]


def test_partition_scenario_replays_exact_trace() -> None:
    actions = (
        ScenarioAction.partition(("a",), ("b", "c")),
        ScenarioAction.send("a", "b", "blocked"),
        ScenarioAction.send("b", "c", "local"),
        ScenarioAction.run(),
        ScenarioAction.heal_partition(("a",), ("b", "c")),
        ScenarioAction.send("a", "b", "healed"),
    )

    first = _registered_simulator().run_scenario(actions)
    second = _registered_simulator().run_scenario(actions)

    assert first == second
    assert [record.kind for record in first].count("partition-drop") == 1
    assert [record.kind for record in first].count("partition") == 1
    assert [record.kind for record in first].count("heal-partition") == 1


def test_partition_prevents_raft_majority_until_healed() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    n1 = cluster.node("n1")

    sim.partition(("n1",), ("n2", "n3"))
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.CANDIDATE
    assert cluster.leaders_by_term == {}
    assert len([record for record in sim.trace if record.kind == "partition-drop"]) == 2

    sim.heal_partition(("n1",), ("n2", "n3"))
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.LEADER
    assert cluster.leaders_by_term == {2: "n1"}
    cluster.assert_log_matching()


def test_partition_rejects_invalid_groups() -> None:
    sim = _registered_simulator()

    for left, right in (
        ((), ("b",)),
        (("a",), ()),
        (("a", "a"), ("b",)),
        (("a",), ("a", "b")),
        (("a",), ("missing",)),
    ):
        try:
            sim.partition(left, right)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid partition {left!r}/{right!r} was accepted")


def test_directional_link_block_only_drops_one_way_until_healed() -> None:
    sim = _registered_simulator()

    sim.block_link("a", "b")
    sim.send("a", "b", "blocked-a-b")
    sim.send("b", "a", "reverse-still-open")
    sim.run()

    assert sim.volatile_state["a"]["received"] == ["reverse-still-open"]
    assert sim.volatile_state["b"].get("received", []) == []
    assert [record.kind for record in sim.trace].count("partition-drop") == 1
    assert [record.kind for record in sim.trace].count("block-link") == 1

    sim.heal_link("a", "b")
    sim.send("a", "b", "healed-a-b")
    sim.run()

    assert sim.volatile_state["b"]["received"] == ["healed-a-b"]
    assert [record.kind for record in sim.trace].count("heal-link") == 1


def test_directional_link_scenario_replays_exact_trace() -> None:
    actions = (
        ScenarioAction.block_link("a", "b"),
        ScenarioAction.send("a", "b", "blocked"),
        ScenarioAction.send("b", "a", "reverse"),
        ScenarioAction.run(),
        ScenarioAction.heal_link("a", "b"),
        ScenarioAction.send("a", "b", "healed"),
    )

    first = _registered_simulator().run_scenario(actions)
    second = _registered_simulator().run_scenario(actions)

    assert first == second
    assert [record.kind for record in first].count("partition-drop") == 1
    assert [record.kind for record in first].count("block-link") == 1
    assert [record.kind for record in first].count("heal-link") == 1


def test_directional_vote_response_loss_prevents_raft_majority_until_healed() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    n1 = cluster.node("n1")

    sim.block_link("n2", "n1")
    sim.block_link("n3", "n1")
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.CANDIDATE
    assert cluster.leaders_by_term == {}
    assert len([record for record in sim.trace if record.kind == "partition-drop"]) == 2

    sim.heal_link("n2", "n1")
    sim.heal_link("n3", "n1")
    n1.start_election()
    sim.run()

    assert n1.role is RaftRole.LEADER
    assert cluster.leaders_by_term == {2: "n1"}
    cluster.assert_election_safety()
    cluster.assert_log_matching()
    cluster.assert_leader_completeness()
    cluster.assert_state_machine_safety()


def test_directional_link_rejects_invalid_endpoints() -> None:
    sim = _registered_simulator()

    for src, dst in (("a", "a"), ("a", "missing"), ("missing", "b")):
        try:
            sim.block_link(src, dst)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid directional link {src!r}->{dst!r} was accepted")
