from distlab.combined_fault_campaign import SeededCombinedFaultScenarioCampaign
from distlab.lifecycle import SeededLifecycleGenerator
from distlab.link_fault_schedule import SeededLinkFaultGenerator
from distlab.randomized_faults import SeededFaultGenerator
from distlab.randomized_workload import SeededClientWorkloadGenerator


def _campaign() -> SeededCombinedFaultScenarioCampaign:
    return SeededCombinedFaultScenarioCampaign(
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
        lifecycle_generator=SeededLifecycleGenerator(
            nodes=("n1", "n2", "n3"),
            crash_rate=0.0,
            restart_rate=0.0,
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


def test_seeded_combined_fault_campaign_runs_multiple_replayable_schedules() -> None:
    result = _campaign().run((7, 11))

    assert result.attempted_seeds == (7, 11)
    assert result.failure is None


def test_seeded_combined_fault_campaign_rejects_duplicate_seeds() -> None:
    try:
        _campaign().run((7, 7))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate campaign seeds must be rejected")
