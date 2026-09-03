from distlab.randomized_faults import (
    FaultOpportunity,
    SeededFaultGenerator,
    SeededFaultSchedule,
)
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadGenerator,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner
from distlab.workload_minimizer import NonLinearizableClientWorkloadMinimizer


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


def test_minimizer_removes_redundant_actions_and_returns_1_minimal_failure() -> None:
    workload, faults = _failing_inputs()
    noise = ClientWorkloadAction(
        operation_id="noise",
        client_id="client",
        node_id="n1",
        kind=ClientOperationKind.GET,
        key="x",
    )
    augmented = SeededClientWorkloadSchedule(
        seed=workload.seed,
        actions=(noise, *workload.actions),
    )

    minimized = NonLinearizableClientWorkloadMinimizer().minimize(augmented, faults)

    assert 0 in minimized.removed_original_indices
    assert minimized.schedule.actions == workload.actions
    replay = ReplicatedKVScenarioRunner(minimized.schedule, faults).run()
    assert not replay.linearizability.linearizable

    for position in range(len(minimized.schedule.actions)):
        candidate = SeededClientWorkloadSchedule(
            seed=minimized.schedule.seed,
            actions=(
                minimized.schedule.actions[:position]
                + minimized.schedule.actions[position + 1 :]
            ),
        )
        assert ReplicatedKVScenarioRunner(candidate, faults).run().linearizability.linearizable


def test_minimizer_is_deterministic() -> None:
    workload, faults = _failing_inputs()

    first = NonLinearizableClientWorkloadMinimizer().minimize(workload, faults)
    second = NonLinearizableClientWorkloadMinimizer().minimize(workload, faults)

    assert second == first


def test_minimizer_preserves_original_index_partition() -> None:
    workload, faults = _failing_inputs()

    minimized = NonLinearizableClientWorkloadMinimizer().minimize(workload, faults)

    partition = minimized.kept_original_indices + minimized.removed_original_indices
    assert len(set(partition)) == len(partition)
    assert set(partition) == set(range(len(workload.actions)))
    assert minimized.schedule.actions == tuple(
        workload.actions[index] for index in minimized.kept_original_indices
    )


def test_minimizer_rejects_linearizable_baseline() -> None:
    workload, _ = _failing_inputs()
    no_faults = SeededFaultSchedule(seed=workload.seed, rules=())

    try:
        NonLinearizableClientWorkloadMinimizer().minimize(workload, no_faults)
    except ValueError as exc:
        assert "non-linearizable" in str(exc)
    else:
        raise AssertionError("linearizable baseline must be rejected")
