import pytest

from distlab import (
    FaultAction,
    FaultOpportunity,
    ScenarioAction,
    SeededFaultGenerator,
    SeededFaultSchedule,
    Simulator,
)


def _opportunities() -> tuple[FaultOpportunity, ...]:
    return tuple(FaultOpportunity("a", "b", ordinal) for ordinal in range(1, 9))


def _actions() -> tuple[ScenarioAction, ...]:
    return tuple(ScenarioAction.send("a", "b", ordinal) for ordinal in range(1, 9))


def test_same_seed_compiles_to_identical_explicit_schedule() -> None:
    generator = SeededFaultGenerator(
        drop_rate=0.25,
        delay_rate=0.25,
        duplicate_rate=0.25,
        max_extra_delay=4,
    )

    first = generator.compile(1729, _opportunities())
    second = generator.compile(1729, tuple(reversed(_opportunities())))

    assert first == second
    assert first.to_json() == second.to_json()
    assert all(rule.action is not FaultAction.DELIVER for rule in first.rules)


def test_persisted_schedule_replays_without_using_seed_generation() -> None:
    generator = SeededFaultGenerator(
        drop_rate=0.25,
        delay_rate=0.25,
        duplicate_rate=0.25,
        max_extra_delay=4,
    )
    generated = generator.compile(8086, _opportunities())
    persisted = generated.to_json()
    replayed = SeededFaultSchedule.from_json(persisted)

    traces = []
    deliveries = []
    for schedule in (generated, replayed):
        delivered: list[tuple[int, object]] = []
        sim = Simulator(fault_plan=schedule.to_fault_plan())
        sim.register(
            "b",
            lambda current, message, output=delivered: output.append(
                (current.time, message.payload)
            ),
        )
        traces.append(sim.run_scenario(_actions()))
        deliveries.append(delivered)

    assert replayed == generated
    assert traces[0] == traces[1]
    assert deliveries[0] == deliveries[1]


def test_different_seeds_are_reproducible_independent_trials() -> None:
    generator = SeededFaultGenerator(
        drop_rate=0.34,
        delay_rate=0.33,
        duplicate_rate=0.33,
        max_extra_delay=7,
    )

    seed_one = generator.compile(1, _opportunities())
    seed_two = generator.compile(2, _opportunities())

    assert seed_one == generator.compile(1, _opportunities())
    assert seed_two == generator.compile(2, _opportunities())
    assert seed_one.rules != seed_two.rules


def test_generated_rules_target_only_declared_message_ordinals() -> None:
    opportunities = (
        FaultOpportunity("n1", "n2", 2),
        FaultOpportunity("n1", "n3", 4),
    )
    schedule = SeededFaultGenerator(
        drop_rate=1,
        delay_rate=0,
        duplicate_rate=0,
    ).compile(5, opportunities)

    assert [(rule.src, rule.dst, rule.ordinal) for rule in schedule.rules] == [
        ("n1", "n2", 2),
        ("n1", "n3", 4),
    ]


def test_invalid_generator_and_opportunities_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum"):
        SeededFaultGenerator(drop_rate=0.5, delay_rate=0.5, duplicate_rate=0.1)
    with pytest.raises(ValueError, match="positive"):
        SeededFaultGenerator(max_extra_delay=0)
    with pytest.raises(ValueError, match="positive"):
        FaultOpportunity("a", "b", 0)
    with pytest.raises(ValueError, match="unique"):
        SeededFaultGenerator().compile(
            1,
            (FaultOpportunity("a", "b", 1), FaultOpportunity("a", "b", 1)),
        )


def test_schedule_deserialization_rejects_unknown_version() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SeededFaultSchedule.from_json('{"version":2,"seed":1,"rules":[]}')
