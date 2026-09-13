import pytest

from distlab.link_fault_schedule import (
    LinkFaultAction,
    LinkFaultKind,
    SeededLinkFaultGenerator,
    SeededLinkFaultSchedule,
)
from distlab.simulator import Simulator


def _recording_handler(deliveries: list[tuple[int, str, str, object]]):
    def handle(sim: Simulator, message) -> None:
        deliveries.append((sim.time, message.src, message.dst, message.payload))

    return handle


def test_link_fault_schedule_round_trips_and_replays_exactly() -> None:
    schedule = SeededLinkFaultSchedule(
        seed=17,
        actions=(
            LinkFaultAction(
                action_id="link-fault-000001",
                kind=LinkFaultKind.SET_DELAY,
                src="n1",
                dst="n2",
                before_action_index=0,
                extra_delay=4,
            ),
            LinkFaultAction(
                action_id="link-fault-000002",
                kind=LinkFaultKind.CLEAR_DELAY,
                src="n1",
                dst="n2",
                before_action_index=1,
            ),
        ),
    )

    decoded = SeededLinkFaultSchedule.from_json(schedule.to_json())
    assert decoded == schedule

    traces = []
    deliveries_by_run = []
    for _ in range(2):
        deliveries: list[tuple[int, str, str, object]] = []
        sim = Simulator()
        sim.register("n1", _recording_handler(deliveries))
        sim.register("n2", _recording_handler(deliveries))

        sim.run_scenario(decoded.actions_before(0))
        sim.send("n1", "n2", "slow")
        sim.run()
        sim.run_scenario(decoded.actions_before(1))
        sim.send("n1", "n2", "normal")
        sim.run()

        traces.append(sim.trace)
        deliveries_by_run.append(deliveries)

    assert traces[0] == traces[1]
    assert deliveries_by_run[0] == deliveries_by_run[1]
    assert deliveries_by_run[0] == [
        (5, "n1", "n2", "slow"),
        (6, "n1", "n2", "normal"),
    ]


def test_directional_link_schedule_preserves_reverse_direction() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator()
    sim.register("n1", _recording_handler(deliveries))
    sim.register("n2", _recording_handler(deliveries))
    schedule = SeededLinkFaultSchedule(
        seed=1,
        actions=(
            LinkFaultAction(
                action_id="link-fault-000001",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
        ),
    )

    sim.run_scenario(schedule.actions_before(0))
    sim.send("n1", "n2", "blocked")
    sim.send("n2", "n1", "reverse")
    sim.run()

    assert deliveries == [(1, "n2", "n1", "reverse")]
    assert any(record.kind == "partition-drop" for record in sim.trace)


def test_seeded_generator_is_order_independent_and_persistable() -> None:
    generator_a = SeededLinkFaultGenerator(
        nodes=("n3", "n1", "n2"),
        block_rate=0.35,
        heal_rate=0.2,
        delay_rate=0.3,
        clear_delay_rate=0.15,
        max_extra_delay=5,
    )
    generator_b = SeededLinkFaultGenerator(
        nodes=("n1", "n2", "n3"),
        block_rate=0.35,
        heal_rate=0.2,
        delay_rate=0.3,
        clear_delay_rate=0.15,
        max_extra_delay=5,
    )

    first = generator_a.compile(seed=20260913, boundary_count=40)
    second = generator_b.compile(seed=20260913, boundary_count=40)

    assert first == second
    assert first.actions
    assert SeededLinkFaultSchedule.from_json(first.to_json()) == first
    assert all(action.src != action.dst for action in first.actions)
    assert [action.before_action_index for action in first.actions] == sorted(
        action.before_action_index for action in first.actions
    )


def test_link_fault_action_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="distinct"):
        LinkFaultAction("bad", LinkFaultKind.BLOCK, "n1", "n1", 0)
    with pytest.raises(ValueError, match="positive"):
        LinkFaultAction("bad", LinkFaultKind.SET_DELAY, "n1", "n2", 0)
    with pytest.raises(ValueError, match="only valid"):
        LinkFaultAction(
            "bad",
            LinkFaultKind.BLOCK,
            "n1",
            "n2",
            0,
            extra_delay=1,
        )
    with pytest.raises(ValueError, match="ordered"):
        SeededLinkFaultSchedule(
            seed=1,
            actions=(
                LinkFaultAction("late", LinkFaultKind.BLOCK, "n1", "n2", 2),
                LinkFaultAction("early", LinkFaultKind.HEAL, "n1", "n2", 1),
            ),
        )


def test_seeded_generator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SeededLinkFaultGenerator(nodes=("n1",))
    with pytest.raises(ValueError, match="sum to at most 1"):
        SeededLinkFaultGenerator(
            nodes=("n1", "n2"),
            block_rate=0.6,
            delay_rate=0.5,
        )
    with pytest.raises(ValueError, match="positive"):
        SeededLinkFaultGenerator(nodes=("n1", "n2"), max_extra_delay=0)
    with pytest.raises(ValueError, match="non-negative"):
        SeededLinkFaultGenerator(nodes=("n1", "n2")).compile(1, -1)
