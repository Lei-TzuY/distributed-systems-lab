import json

from distlab.link_fault_campaign import (
    LinkFaultCampaignFailureArtifact,
    SeededLinkFaultScenarioCampaign,
)
from distlab.link_fault_schedule import (
    LinkFaultAction,
    LinkFaultKind,
    SeededLinkFaultGenerator,
    SeededLinkFaultSchedule,
)
from distlab.randomized_faults import SeededFaultGenerator, SeededFaultSchedule
from distlab.randomized_workload import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadGenerator,
    SeededClientWorkloadSchedule,
)
from distlab.scenario_runner import ReplicatedKVScenarioRunner


def _stale_read_inputs():
    workload = SeededClientWorkloadSchedule(
        seed=41,
        actions=(
            ClientWorkloadAction(
                operation_id="op-000001",
                client_id="client",
                node_id="n1",
                kind=ClientOperationKind.PUT,
                key="x",
                value="one",
                request_id=1,
            ),
            ClientWorkloadAction(
                operation_id="op-000002",
                client_id="client",
                node_id="n2",
                kind=ClientOperationKind.GET,
                key="x",
            ),
        ),
    )
    faults = SeededFaultSchedule(seed=41, rules=())
    link_faults = SeededLinkFaultSchedule(
        seed=41,
        actions=(
            LinkFaultAction(
                action_id="block-replication",
                kind=LinkFaultKind.BLOCK,
                src="n1",
                dst="n2",
                before_action_index=0,
            ),
            LinkFaultAction(
                action_id="irrelevant-heal",
                kind=LinkFaultKind.HEAL,
                src="n1",
                dst="n2",
                before_action_index=2,
            ),
        ),
    )
    return workload, faults, link_faults


def test_link_fault_failure_artifact_round_trips_minimizes_and_replays() -> None:
    workload, faults, link_faults = _stale_read_inputs()
    result = ReplicatedKVScenarioRunner(
        workload,
        faults,
        link_faults=link_faults,
    ).run()
    assert not result.linearizability.linearizable

    failure = LinkFaultCampaignFailureArtifact.capture(
        workload,
        faults,
        link_faults,
        result,
    )
    encoded = failure.to_json()
    restored = LinkFaultCampaignFailureArtifact.from_json(encoded)

    assert restored.to_json() == encoded
    assert json.loads(encoded)["version"] == 1
    assert restored.kept_link_fault_action_indices == (0,)
    assert restored.removed_link_fault_action_indices == (1,)
    assert tuple(action.action_id for action in restored.minimized_link_faults.actions) == (
        "block-replication",
    )
    replay = restored.replay()
    assert not replay.linearizability.linearizable
    assert replay.snapshots["n1"] == {"x": "one"}
    assert replay.snapshots["n2"] == {}


def test_link_fault_failure_artifact_rejects_index_projection_mismatch() -> None:
    workload, faults, link_faults = _stale_read_inputs()
    result = ReplicatedKVScenarioRunner(
        workload,
        faults,
        link_faults=link_faults,
    ).run()
    failure = LinkFaultCampaignFailureArtifact.capture(
        workload,
        faults,
        link_faults,
        result,
    )
    raw = json.loads(failure.to_json())
    raw["kept_link_fault_action_indices"] = [1]
    raw["removed_link_fault_action_indices"] = [0]

    try:
        LinkFaultCampaignFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "minimized link faults" in str(exc)
    else:
        raise AssertionError("mismatched minimized link fault projection must be rejected")


def test_seeded_link_fault_campaign_runs_multiple_replayable_schedules() -> None:
    campaign = SeededLinkFaultScenarioCampaign(
        workload_generator=SeededClientWorkloadGenerator(
            clients=("client",),
            nodes=("n1",),
            keys=("x",),
            values=("one", "two"),
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=0.0,
            delay_rate=0.0,
            duplicate_rate=0.0,
        ),
        link_fault_generator=SeededLinkFaultGenerator(
            nodes=("n1", "n2", "n3"),
            block_rate=0.0,
            heal_rate=0.0,
            delay_rate=0.0,
            clear_delay_rate=0.0,
        ),
        fault_opportunities=(),
        operation_count=3,
    )

    result = campaign.run((7, 11))

    assert result.attempted_seeds == (7, 11)
    assert result.failure is None
