import pytest

from distlab.joint_scenario_minimizer import NonLinearizableJointScenarioMinimizer
from distlab.lifecycle import NodeLifecycleAction, NodeLifecycleKind, SeededLifecycleSchedule
from distlab.link_fault_schedule import LinkFaultAction, LinkFaultKind, SeededLinkFaultSchedule
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)


def _inputs(seed: int = 43):
    workload = SeededClientWorkloadSchedule(
        seed=seed,
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
    faults = SeededFaultSchedule(seed=seed, rules=())
    lifecycle = SeededLifecycleSchedule(
        seed=seed,
        actions=(
            NodeLifecycleAction(
                action_id="redundant-stop",
                node_id="n3",
                kind=next(iter(NodeLifecycleKind)),
                before_action_index=2,
            ),
        ),
    )
    link_faults = SeededLinkFaultSchedule(
        seed=seed,
        actions=(
            LinkFaultAction(
                action_id="block",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="redundant-heal",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=2,
            ),
        ),
    )
    return workload, faults, lifecycle, link_faults


def test_joint_scenario_minimization_reaches_stable_fixed_point() -> None:
    reduction = NonLinearizableJointScenarioMinimizer().minimize(*_inputs())
    rerun = NonLinearizableJointScenarioMinimizer().minimize(
        reduction.workload,
        reduction.faults,
        reduction.lifecycle,
        reduction.link_faults,
    )

    assert rerun.workload == reduction.workload
    assert rerun.faults == reduction.faults
    assert rerun.lifecycle == reduction.lifecycle
    assert rerun.link_faults == reduction.link_faults
    assert rerun.removed_workload_original_indices == ()
    assert rerun.removed_fault_original_indices == ()
    assert rerun.removed_lifecycle_original_indices == ()
    assert rerun.removed_link_fault_original_indices == ()


def test_joint_scenario_minimization_rejects_seed_drift() -> None:
    workload, faults, lifecycle, link_faults = _inputs()
    drifted = SeededFaultSchedule(seed=99, rules=faults.rules)

    with pytest.raises(ValueError, match="seed"):
        NonLinearizableJointScenarioMinimizer().minimize(
            workload,
            drifted,
            lifecycle,
            link_faults,
        )
