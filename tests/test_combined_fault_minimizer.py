from distlab.combined_fault_minimizer import NonLinearizableCombinedFaultScheduleMinimizer
from distlab.lifecycle import NodeLifecycleAction, NodeLifecycleKind, SeededLifecycleSchedule
from distlab.link_fault_schedule import LinkFaultAction, LinkFaultKind, SeededLinkFaultSchedule
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def test_combined_minimizer_preserves_one_joint_failure_witness() -> None:
    workload = SeededClientWorkloadSchedule(
        seed=41,
        actions=(
            ClientWorkloadAction(
                operation_id="put",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="get",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.GET,
                key="x",
            ),
        ),
    )
    faults = SeededFaultSchedule(seed=41, rules=())
    lifecycle = SeededLifecycleSchedule(
        seed=41,
        actions=(
            NodeLifecycleAction(
                action_id="stop-n3",
                node_id="n3",
                kind=next(iter(NodeLifecycleKind)),
                before_action_index=2,
            ),
        ),
    )
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="block",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="heal",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=2,
            ),
        ),
    )

    reduction = NonLinearizableCombinedFaultScheduleMinimizer().minimize(
        workload,
        faults,
        lifecycle,
        link_faults,
    )

    assert reduction.lifecycle.actions == ()
    assert reduction.kept_lifecycle_original_indices == ()
    assert reduction.removed_lifecycle_original_indices == (0,)
    assert [action.action_id for action in reduction.link_faults.actions] == ["block"]
    assert reduction.kept_link_fault_original_indices == (0,)
    assert reduction.removed_link_fault_original_indices == (1,)

    replay = ReplicatedKVScenarioRunner(
        workload,
        faults,
        lifecycle=reduction.lifecycle,
        link_faults=reduction.link_faults,
    ).run()
    assert not replay.linearizability.linearizable


def test_combined_minimizer_rejects_seed_drift() -> None:
    workload = SeededClientWorkloadSchedule(seed=7, actions=())
    faults = SeededFaultSchedule(seed=7, rules=())
    lifecycle = SeededLifecycleSchedule.empty(7)
    link_faults = SeededLinkFaultSchedule(seed=8, actions=())

    try:
        NonLinearizableCombinedFaultScheduleMinimizer().minimize(
            workload,
            faults,
            lifecycle,
            link_faults,
        )
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("combined minimizer must reject schedule seed drift")
