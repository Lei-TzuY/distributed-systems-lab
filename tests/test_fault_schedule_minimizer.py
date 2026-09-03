from distlab.fault_schedule_minimizer import NonLinearizableFaultScheduleMinimizer
from distlab.randomized_faults import (
    FaultOpportunity,
    SeededFaultGenerator,
    SeededFaultSchedule,
)
from distlab.randomized_workload import (
    SeededClientWorkloadGenerator,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner
from distlab.simulator import FaultAction, FaultRule


def _failing_inputs() -> tuple[SeededClientWorkloadSchedule, SeededFaultSchedule]:
    workload = SeededClientWorkloadGenerator(
        clients=("client",),
        nodes=("n1", "n2"),
        keys=("x",),
        values=("one",),
        put_rate=0.5,
        delete_rate=0.0,
    ).compile(2, 2)
    faults = SeededFaultGenerator(
        drop_rate=1.0,
        delay_rate=0.0,
        duplicate_rate=0.0,
    ).compile(
        2,
        (
            FaultOpportunity("n1", "n2", 2),
            FaultOpportunity("n1", "n2", 3),
        ),
    )
    return workload, faults


def test_minimizer_removes_redundant_rules_and_returns_1_minimal_failure() -> None:
    workload, faults = _failing_inputs()
    augmented = SeededFaultSchedule(
        seed=faults.seed,
        rules=(
            *faults.rules,
            FaultRule(FaultAction.DROP, src="n1", dst="n2", ordinal=999),
        ),
    )

    minimized = NonLinearizableFaultScheduleMinimizer().minimize(workload, augmented)

    assert 2 in minimized.removed_original_indices
    replay = ReplicatedKVScenarioRunner(workload, minimized.schedule).run()
    assert not replay.linearizability.linearizable

    for position in range(len(minimized.schedule.rules)):
        candidate = SeededFaultSchedule(
            seed=minimized.schedule.seed,
            rules=(
                minimized.schedule.rules[:position]
                + minimized.schedule.rules[position + 1 :]
            ),
        )
        assert ReplicatedKVScenarioRunner(workload, candidate).run().linearizability.linearizable


def test_minimizer_is_deterministic() -> None:
    workload, faults = _failing_inputs()

    first = NonLinearizableFaultScheduleMinimizer().minimize(workload, faults)
    second = NonLinearizableFaultScheduleMinimizer().minimize(workload, faults)

    assert second == first


def test_minimizer_rejects_linearizable_baseline() -> None:
    workload, faults = _failing_inputs()
    no_faults = SeededFaultSchedule(seed=faults.seed, rules=())

    try:
        NonLinearizableFaultScheduleMinimizer().minimize(workload, no_faults)
    except ValueError as exc:
        assert "non-linearizable" in str(exc)
    else:
        raise AssertionError("linearizable baseline must be rejected")
