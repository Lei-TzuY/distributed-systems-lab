import pytest

from distlab.link_fault_schedule import (
    LinkFaultAction,
    LinkFaultKind,
    SeededLinkFaultSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def _stale_read_workload() -> SeededClientWorkloadSchedule:
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


def test_directional_link_schedule_executes_in_kv_scenario_and_replays_exactly() -> None:
    workload = _stale_read_workload()
    faults = SeededFaultSchedule(seed=41, rules=())
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="block-leader-to-n2",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
        ),
    )

    first = ReplicatedKVScenarioRunner(
        workload,
        faults,
        link_faults=link_faults,
    ).run()
    second = ReplicatedKVScenarioRunner(
        SeededClientWorkloadSchedule.from_json(workload.to_json()),
        SeededFaultSchedule.from_json(faults.to_json()),
        link_faults=SeededLinkFaultSchedule.from_json(link_faults.to_json()),
    ).run()

    assert first.snapshots["n1"] == {"x": "one"}
    assert first.snapshots["n2"] == {}
    assert first.snapshots["n3"] == {"x": "one"}
    assert not first.linearizability.linearizable
    assert first.history.invocations() == second.history.invocations()
    assert first.history.completed() == second.history.completed()
    assert first.trace == second.trace
    assert [record.kind for record in first.trace].count("scenario-link-fault") == 1
    assert [record.kind for record in first.trace].count("partition-drop") >= 1


def test_link_schedule_can_heal_before_later_workload_boundary() -> None:
    workload = _stale_read_workload()
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="block-leader-to-n2",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="heal-leader-to-n2",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=1,
            ),
        ),
    )

    result = ReplicatedKVScenarioRunner(
        workload,
        SeededFaultSchedule(seed=41, rules=()),
        link_faults=link_faults,
    ).run()

    kinds = [record.kind for record in result.trace]
    assert kinds.count("scenario-link-fault") == 2
    assert kinds.index("block-link") < kinds.index("heal-link")


def test_runner_rejects_unknown_link_fault_endpoint() -> None:
    workload = _stale_read_workload()
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="unknown-endpoint",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="ghost",
                before_action_index=0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="link fault schedule references unknown nodes"):
        ReplicatedKVScenarioRunner(
            workload,
            SeededFaultSchedule(seed=41, rules=()),
            link_faults=link_faults,
        )


def test_runner_rejects_link_fault_boundary_past_workload() -> None:
    workload = _stale_read_workload()
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="late-block",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=3,
            ),
        ),
    )

    with pytest.raises(ValueError, match="link fault action references a workload boundary"):
        ReplicatedKVScenarioRunner(
            workload,
            SeededFaultSchedule(seed=41, rules=()),
            link_faults=link_faults,
        )
