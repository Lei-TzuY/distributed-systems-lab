from distlab.combined_fault_campaign import SeededCombinedFaultScenarioCampaign
from distlab.joint_combined_fault_artifact import JointCombinedFaultFailureArtifact
from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleGenerator,
    SeededLifecycleSchedule,
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


class _WorkloadGenerator:
    def compile(self, seed: int, operation_count: int) -> SeededClientWorkloadSchedule:
        assert operation_count == 2
        return SeededClientWorkloadSchedule(
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


class _FaultGenerator:
    def compile(self, seed: int, opportunities) -> SeededFaultSchedule:
        assert opportunities == ()
        return SeededFaultSchedule(seed=seed, rules=())


class _LifecycleGenerator:
    def compile(self, seed: int, action_count: int) -> SeededLifecycleSchedule:
        assert action_count == 2
        return SeededLifecycleSchedule(
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


class _LinkFaultGenerator:
    def compile(self, seed: int, action_count: int) -> SeededLinkFaultSchedule:
        assert action_count == 2
        return SeededLinkFaultSchedule(
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


def test_combined_fault_campaign_publishes_joint_minimized_failure_evidence() -> None:
    campaign = SeededCombinedFaultScenarioCampaign(
        workload_generator=_WorkloadGenerator(),
        fault_generator=_FaultGenerator(),
        lifecycle_generator=_LifecycleGenerator(),
        link_fault_generator=_LinkFaultGenerator(),
        fault_opportunities=(),
        operation_count=2,
    )

    result = campaign.run((43,))

    assert result.attempted_seeds == (43,)
    assert isinstance(result.failure, JointCombinedFaultFailureArtifact)
    assert result.failure.minimized_lifecycle.actions == ()
    assert result.failure.kept_link_fault_action_indices == (0,)
    assert result.failure.removed_link_fault_action_indices == (1,)
    result.failure.replay()
