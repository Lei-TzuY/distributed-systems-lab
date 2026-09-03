import json

from distlab.campaign import CampaignFailureArtifact, SeededScenarioCampaign
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
    replay = result.failure.replay()
    assert not replay.linearizability.linearizable
    assert replay.snapshots["n1"] == {"x": "one"}
    assert replay.snapshots["n2"] == {}


def test_failure_artifact_json_round_trip_is_canonical_and_replayable() -> None:
    failure = _stale_read_campaign().run((2,)).failure
    assert failure is not None

    encoded = failure.to_json()
    restored = CampaignFailureArtifact.from_json(encoded)

    assert restored.to_json() == encoded
    assert json.loads(encoded)["version"] == 1
    assert restored.trace_json == failure.trace_json
    assert restored.replay().trace


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
