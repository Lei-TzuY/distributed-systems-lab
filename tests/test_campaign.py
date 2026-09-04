import json

from distlab.campaign import CampaignFailureArtifact, SeededScenarioCampaign
from distlab.lifecycle import SeededLifecycleGenerator
from distlab.randomized_faults import FaultOpportunity, SeededFaultGenerator
from distlab.randomized_workload import SeededClientWorkloadGenerator


def _stale_read_campaign() -> SeededScenarioCampaign:
    return SeededScenarioCampaign(
        workload_generator=SeededClientWorkloadGenerator(
            clients=("client",),
            nodes=("n1", "n2"),
            keys=("x",),
            values=("one",),
            put_rate=0.5,
            delete_rate=0.0,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=1.0,
            delay_rate=0.0,
            duplicate_rate=0.0,
        ),
        fault_opportunities=(
            FaultOpportunity("n1", "n2", 2),
            FaultOpportunity("n1", "n2", 3),
        ),
        operation_count=2,
    )


def test_campaign_captures_first_failure_and_exactly_replays_artifact() -> None:
    campaign = _stale_read_campaign()

    result = campaign.run((2, 37))

    assert result.attempted_seeds == (2,)
    assert result.failure is not None
    assert result.failure.minimized_operation_ids == ("op-000001", "op-000002")
    assert result.failure.minimized_workload.actions == tuple(
        result.failure.workload.actions[index]
        for index in result.failure.kept_workload_action_indices
    )
    assert set(result.failure.kept_workload_action_indices) | set(
        result.failure.removed_workload_action_indices
    ) == set(range(len(result.failure.workload.actions)))
    assert result.failure.minimized_faults.rules == tuple(
        result.failure.faults.rules[index]
        for index in result.failure.kept_fault_rule_indices
    )
    assert set(result.failure.kept_fault_rule_indices) | set(
        result.failure.removed_fault_rule_indices
    ) == set(range(len(result.failure.faults.rules)))
    assert result.failure.lifecycle.actions == ()
    replay = result.failure.replay()
    assert not replay.linearizability.linearizable
    assert replay.snapshots["n1"] == {"x": "one"}
    assert replay.snapshots["n2"] == {}


def test_failure_artifact_json_round_trip_is_canonical_and_replayable() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None

    encoded = failure.to_json()
    restored = CampaignFailureArtifact.from_json(encoded)
    raw = json.loads(encoded)

    assert restored.to_json() == encoded
    assert raw["version"] == 4
    assert raw["minimized_workload"] == json.loads(failure.minimized_workload.to_json())
    assert raw["kept_workload_action_indices"] == list(
        failure.kept_workload_action_indices
    )
    assert raw["removed_workload_action_indices"] == list(
        failure.removed_workload_action_indices
    )
    assert raw["minimized_faults"] == json.loads(failure.minimized_faults.to_json())
    assert raw["kept_fault_rule_indices"] == list(failure.kept_fault_rule_indices)
    assert raw["removed_fault_rule_indices"] == list(failure.removed_fault_rule_indices)
    assert raw["lifecycle"] == json.loads(failure.lifecycle.to_json())
    assert restored.trace_json == failure.trace_json
    assert restored.replay().trace


def test_failure_artifact_accepts_version_three_as_empty_lifecycle() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None
    raw = json.loads(failure.to_json())
    raw["version"] = 3
    raw.pop("lifecycle")

    restored = CampaignFailureArtifact.from_json(json.dumps(raw))

    assert restored.lifecycle.actions == ()
    assert restored.replay().trace


def test_campaign_failure_persists_and_replays_lifecycle_schedule() -> None:
    campaign = SeededScenarioCampaign(
        workload_generator=SeededClientWorkloadGenerator(
            clients=("client",),
            nodes=("n1", "n2"),
            keys=("x",),
            values=("one",),
            put_rate=0.5,
            delete_rate=0.0,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=1.0,
            delay_rate=0.0,
            duplicate_rate=0.0,
        ),
        fault_opportunities=(
            FaultOpportunity("n1", "n2", 2),
            FaultOpportunity("n1", "n2", 3),
        ),
        operation_count=2,
        lifecycle_generator=SeededLifecycleGenerator(
            nodes=("n3",),
            crash_rate=0.5,
            restart_rate=0.0,
        ),
    )

    failure = campaign.run((2,)).failure

    assert failure is not None
    assert len(failure.lifecycle.actions) == 1
    assert failure.lifecycle.actions[0].node_id == "n3"
    assert failure.lifecycle.actions[0].before_action_index == 2
    assert failure.minimized_workload == failure.workload
    assert failure.minimized_faults == failure.faults
    assert failure.removed_workload_action_indices == ()
    assert failure.removed_fault_rule_indices == ()
    encoded = failure.to_json()
    restored = CampaignFailureArtifact.from_json(encoded)
    assert restored.lifecycle == failure.lifecycle
    replay = restored.replay()
    assert not replay.linearizability.linearizable
    assert any(record.kind == "scenario-lifecycle" for record in replay.trace)


def test_failure_artifact_rejects_workload_index_partition_mismatch() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None
    raw = json.loads(failure.to_json())
    raw["kept_workload_action_indices"] = raw["kept_workload_action_indices"][:-1]
    raw["removed_workload_action_indices"] = []

    try:
        CampaignFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "partition" in str(exc)
    else:
        raise AssertionError("incomplete workload index partition must be rejected")


def test_failure_artifact_rejects_workload_projection_mismatch() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None
    raw = json.loads(failure.to_json())
    raw["minimized_workload"]["actions"] = []

    try:
        CampaignFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "minimized workload" in str(exc)
    else:
        raise AssertionError("mismatched minimized workload must be rejected")


def test_failure_artifact_rejects_fault_index_partition_mismatch() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None
    raw = json.loads(failure.to_json())
    raw["removed_fault_rule_indices"] = []

    try:
        CampaignFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "partition" in str(exc)
    else:
        raise AssertionError("incomplete fault index partition must be rejected")


def test_failure_artifact_rejects_lifecycle_projection_minimization() -> None:
    campaign = SeededScenarioCampaign(
        workload_generator=SeededClientWorkloadGenerator(
            clients=("client",),
            nodes=("n1", "n2"),
            keys=("x",),
            values=("one",),
            put_rate=0.5,
            delete_rate=0.0,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=1.0,
            delay_rate=0.0,
            duplicate_rate=0.0,
        ),
        fault_opportunities=(
            FaultOpportunity("n1", "n2", 2),
            FaultOpportunity("n1", "n2", 3),
        ),
        operation_count=2,
        lifecycle_generator=SeededLifecycleGenerator(
            nodes=("n3",),
            crash_rate=0.5,
            restart_rate=0.0,
        ),
    )
    failure = campaign.run((2,)).failure
    assert failure is not None
    raw = json.loads(failure.to_json())
    raw["kept_fault_rule_indices"] = raw["kept_fault_rule_indices"][:-1]
    raw["removed_fault_rule_indices"] = [len(failure.faults.rules) - 1]
    raw["minimized_faults"]["rules"] = raw["minimized_faults"]["rules"][:-1]

    try:
        CampaignFailureArtifact.from_json(json.dumps(raw))
    except ValueError as exc:
        assert "preserve the full fault schedule" in str(exc)
    else:
        raise AssertionError("lifecycle artifact fault minimization must be rejected")


def test_campaign_without_failure_records_every_attempted_seed() -> None:
    campaign = SeededScenarioCampaign(
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
        fault_opportunities=(),
        operation_count=4,
    )

    result = campaign.run((1, 2, 3))

    assert result.attempted_seeds == (1, 2, 3)
    assert result.failure is None


def test_campaign_rejects_duplicate_seeds() -> None:
    campaign = _stale_read_campaign()

    try:
        campaign.run((2, 2))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate campaign seeds must be rejected")
