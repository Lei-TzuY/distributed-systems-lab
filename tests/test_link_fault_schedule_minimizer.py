import pytest

from distlab.link_fault_schedule import (
    LinkFaultAction,
    LinkFaultKind,
    SeededLinkFaultSchedule,
)
from distlab.link_fault_schedule_minimizer import NonLinearizableLinkFaultScheduleMinimizer
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def _workload() -> SeededClientWorkloadSchedule:
    return SeededClientWorkloadSchedule(
        seed=41,
        actions=(
            ClientWorkloadAction(
                operation_id="write",
                client_id="writer",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="stale-read",
                client_id="reader",
                node_id="n2",
                kind=ClientOperationKind.GET,
                key="x",
            ),
        ),
    )


def _action(action_id: str, kind: LinkFaultKind, boundary: int) -> LinkFaultAction:
    return LinkFaultAction(
        action_id=action_id,
        kind=kind,
        src="n1",
        dst="n2",
        before_action_index=boundary,
    )


def test_minimizer_removes_irrelevant_link_transitions_and_is_one_minimal() -> None:
    workload = _workload()
    faults = SeededFaultSchedule(seed=41, rules=())
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            _action("block", LinkFaultKind.BLOCK, 0),
            _action("heal-after-history", LinkFaultKind.HEAL, 2),
        ),
    )

    reduction = NonLinearizableLinkFaultScheduleMinimizer().minimize(
        workload,
        faults,
        link_faults,
    )

    assert reduction.kept_original_indices == (0,)
    assert reduction.removed_original_indices == (1,)
    assert tuple(action.action_id for action in reduction.schedule.actions) == ("block",)

    replay = ReplicatedKVScenarioRunner(
        workload,
        faults,
        link_faults=SeededLinkFaultSchedule.from_json(reduction.schedule.to_json()),
    ).run()
    assert not replay.linearizability.linearizable

    without_remaining = ReplicatedKVScenarioRunner(
        workload,
        faults,
        link_faults=SeededLinkFaultSchedule.empty(41),
    ).run()
    assert without_remaining.linearizability.linearizable


def test_minimizer_rejects_linearizable_baseline() -> None:
    workload = _workload()
    with pytest.raises(ValueError, match="requires a non-linearizable scenario"):
        NonLinearizableLinkFaultScheduleMinimizer().minimize(
            workload,
            SeededFaultSchedule(seed=41, rules=()),
            SeededLinkFaultSchedule.empty(41),
        )
