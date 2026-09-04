from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleSchedule,
)
from distlab.lifecycle_minimizer import NonLinearizableLifecycleScheduleMinimizer
from distlab.randomized_faults import (
    FaultOpportunity,
    SeededFaultGenerator,
)
from distlab.randomized_workload import SeededClientWorkloadGenerator


def _stale_read_inputs():
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


def test_minimizer_removes_redundant_crash_restart_pair_deterministically() -> None:
    workload, faults = _stale_read_inputs()
    lifecycle = SeededLifecycleSchedule(
        seed=2,
        actions=(
            NodeLifecycleAction(
                action_id="crash-n3",
                node_id="n3",
                kind=NodeLifecycleKind.CRASH,
                before_action_index=2,
            ),
            NodeLifecycleAction(
                action_id="restart-n3",
                node_id="n3",
                kind=NodeLifecycleKind.RESTART,
                before_action_index=2,
            ),
        ),
    )

    first = NonLinearizableLifecycleScheduleMinimizer().minimize(
        workload,
        faults,
        lifecycle,
    )
    second = NonLinearizableLifecycleScheduleMinimizer().minimize(
        workload,
        faults,
        lifecycle,
    )

    assert first == second
    assert first.schedule.actions == ()
    assert first.kept_original_indices == ()
    assert first.removed_original_indices == (0, 1)


def test_minimizer_preserves_original_projection_and_is_one_minimal() -> None:
    workload, faults = _stale_read_inputs()
    lifecycle = SeededLifecycleSchedule(
        seed=2,
        actions=(
            NodeLifecycleAction(
                action_id="redundant-crash",
                node_id="n3",
                kind=NodeLifecycleKind.CRASH,
                before_action_index=2,
            ),
        ),
    )

    reduction = NonLinearizableLifecycleScheduleMinimizer().minimize(
        workload,
        faults,
        lifecycle,
    )

    assert reduction.schedule.actions == tuple(
        lifecycle.actions[index] for index in reduction.kept_original_indices
    )
    assert set(reduction.kept_original_indices) | set(
        reduction.removed_original_indices
    ) == set(range(len(lifecycle.actions)))
    assert set(reduction.kept_original_indices).isdisjoint(
        reduction.removed_original_indices
    )
    assert reduction.schedule.actions == ()
