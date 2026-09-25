from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleGenerator,
    SeededLifecycleSchedule,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction,
    MultiPaxosActionKind,
    MultiPaxosScenarioOutcome,
    MultiPaxosScenarioRunner,
    MultiPaxosTrialArtifact,
    SeededMultiPaxosActionGenerator,
    SeededMultiPaxosActionSchedule,
    SeededMultiPaxosCampaign,
    multipaxos_fault_opportunities,
)
from distlab.randomized_faults import SeededFaultGenerator, SeededFaultSchedule
from distlab.simulator import FaultAction, FaultRule


def _no_faults(seed: int) -> SeededFaultSchedule:
    return SeededFaultSchedule(seed=seed, rules=())


def test_seeded_multipaxos_actions_compile_deterministically() -> None:
    generator = SeededMultiPaxosActionGenerator(
        node_ids=("n1", "n2", "n3"),
        values=("alpha", "beta"),
        leadership_change_rate=0.3,
        max_events_after=6,
    )

    first = generator.compile(1729, 8)
    second = generator.compile(1729, 8)

    assert first == second
    assert SeededMultiPaxosActionSchedule.from_json(first.to_json()) == first
    assert first.actions[0].kind is MultiPaxosActionKind.PREPARE
    proposal_slots = [
        action.slot
        for action in first.actions
        if action.kind is MultiPaxosActionKind.PROPOSE
    ]
    assert proposal_slots == list(range(1, len(proposal_slots) + 1))


def test_explicit_stable_leader_scenario_amortizes_phase1_and_replays() -> None:
    actions = SeededMultiPaxosActionSchedule(
        seed=11,
        actions=(
            MultiPaxosAction(
                "prepare-n1",
                MultiPaxosActionKind.PREPARE,
                "n1",
                events_after=4,
                round_number=1,
            ),
            MultiPaxosAction(
                "slot-1",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=4,
                slot=1,
                value="alpha",
            ),
            MultiPaxosAction(
                "slot-2",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=4,
                slot=2,
                value="beta",
            ),
        ),
    )

    first = MultiPaxosScenarioRunner(actions, _no_faults(11)).run()
    second = MultiPaxosScenarioRunner(actions, _no_faults(11)).run()

    assert first == second
    assert first.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert [(item.slot, item.value) for item in first.chosen] == [
        (1, "alpha"),
        (2, "beta"),
    ]
    assert sum(
        record.kind == "multipaxos-phase1-start" for record in first.trace
    ) == 1
    assert sum(
        record.kind == "multipaxos-phase2-start" for record in first.trace
    ) == 2


def test_takeover_adopts_durable_value_and_artifact_replays_exactly() -> None:
    actions = SeededMultiPaxosActionSchedule(
        seed=23,
        actions=(
            MultiPaxosAction(
                "prepare-old",
                MultiPaxosActionKind.PREPARE,
                "n1",
                events_after=4,
                round_number=1,
            ),
            MultiPaxosAction(
                "accept-old",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=4,
                slot=1,
                value="old",
            ),
            MultiPaxosAction(
                "prepare-new",
                MultiPaxosActionKind.PREPARE,
                "n2",
                events_after=4,
                round_number=2,
            ),
            MultiPaxosAction(
                "retry-slot",
                MultiPaxosActionKind.PROPOSE,
                "n2",
                events_after=4,
                slot=1,
                value="new",
            ),
        ),
    )
    faults = SeededFaultSchedule(
        seed=23,
        rules=(
            FaultRule(
                FaultAction.DROP,
                src="n2",
                dst="n1",
                ordinal=1,
                payload_type="LeaderAccepted",
            ),
            FaultRule(
                FaultAction.DROP,
                src="n3",
                dst="n1",
                ordinal=1,
                payload_type="LeaderAccepted",
            ),
        ),
    )

    result = MultiPaxosScenarioRunner(actions, faults).run()
    artifact = MultiPaxosTrialArtifact.capture(actions, faults, result)
    restored = MultiPaxosTrialArtifact.from_json(artifact.to_json())

    assert result.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert [(item.slot, item.value) for item in result.chosen] == [(1, "old")]
    assert restored == artifact
    assert restored.replay() == result


def test_no_phase1_quorum_is_incomplete_not_safety_failure() -> None:
    actions = SeededMultiPaxosActionSchedule(
        seed=31,
        actions=(
            MultiPaxosAction(
                "prepare",
                MultiPaxosActionKind.PREPARE,
                "n1",
                events_after=4,
                round_number=1,
            ),
            MultiPaxosAction(
                "proposal",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=2,
                slot=1,
                value="alpha",
            ),
        ),
    )
    faults = SeededFaultSchedule(
        seed=31,
        rules=(
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst="n2",
                ordinal=1,
                payload_type="LeaderPrepare",
            ),
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst="n3",
                ordinal=1,
                payload_type="LeaderPrepare",
            ),
        ),
    )

    result = MultiPaxosScenarioRunner(actions, faults).run()
    artifact = MultiPaxosTrialArtifact.capture(actions, faults, result)

    assert result.outcome is MultiPaxosScenarioOutcome.INCOMPLETE
    assert result.chosen == ()
    assert result.violation is None
    assert [
        record
        for record in result.trace
        if record.kind == "scenario-multipaxos-propose-rejected"
    ]
    assert artifact.replay() == result


def test_crash_restart_and_reprepare_are_part_of_exact_replay() -> None:
    actions = SeededMultiPaxosActionSchedule(
        seed=41,
        actions=(
            MultiPaxosAction(
                "prepare-first",
                MultiPaxosActionKind.PREPARE,
                "n1",
                events_after=4,
            ),
            MultiPaxosAction(
                "crashed-proposal",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=0,
                slot=1,
                value="discarded",
            ),
            MultiPaxosAction(
                "prepare-after-restart",
                MultiPaxosActionKind.PREPARE,
                "n1",
                events_after=4,
            ),
            MultiPaxosAction(
                "safe-proposal",
                MultiPaxosActionKind.PROPOSE,
                "n1",
                events_after=4,
                slot=1,
                value="safe",
            ),
        ),
    )
    lifecycle = SeededLifecycleSchedule(
        seed=41,
        actions=(
            NodeLifecycleAction(
                "crash-n1",
                "n1",
                NodeLifecycleKind.CRASH,
                before_action_index=1,
            ),
            NodeLifecycleAction(
                "restart-n1",
                "n1",
                NodeLifecycleKind.RESTART,
                before_action_index=2,
            ),
        ),
    )

    result = MultiPaxosScenarioRunner(
        actions,
        _no_faults(41),
        lifecycle=lifecycle,
    ).run()
    artifact = MultiPaxosTrialArtifact.capture(
        actions,
        _no_faults(41),
        result,
        lifecycle=lifecycle,
    )

    assert result.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert [(item.slot, item.value) for item in result.chosen] == [(1, "safe")]
    assert [
        record
        for record in result.trace
        if record.kind == "scenario-multipaxos-action-skipped"
    ]
    assert MultiPaxosTrialArtifact.from_json(artifact.to_json()).replay() == result


def test_multipaxos_fault_opportunities_cover_protocol_messages() -> None:
    opportunities = multipaxos_fault_opportunities(
        ("n1", "n2", "n3"),
        max_message_ordinal=2,
    )

    assert len(opportunities) == 3 * 2 * 2 * 4
    assert {
        opportunity.payload_type
        for opportunity in opportunities
        if opportunity.src == "n1"
        and opportunity.dst == "n2"
        and opportunity.ordinal == 1
    } == {"LeaderPrepare", "LeaderPromise", "LeaderAccept", "LeaderAccepted"}


def test_seeded_multipaxos_campaign_preserves_safety_and_replays_every_trial() -> None:
    campaign = SeededMultiPaxosCampaign(
        action_generator=SeededMultiPaxosActionGenerator(
            node_ids=("n1", "n2", "n3"),
            values=("alpha", "beta", "gamma"),
            leadership_change_rate=0.3,
            max_events_after=6,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=0.04,
            delay_rate=0.04,
            duplicate_rate=0.04,
            max_extra_delay=3,
        ),
        action_count=8,
        node_ids=("n1", "n2", "n3"),
        max_message_ordinal=10,
        final_event_budget=128,
    )

    result = campaign.run((101, 202, 303, 404))

    assert result.attempted_seeds == (101, 202, 303, 404)
    assert result.safety_failure is None
    assert len(result.trials) == 4
    for artifact in result.trials:
        replay = MultiPaxosTrialArtifact.from_json(artifact.to_json()).replay()
        assert replay.outcome is not MultiPaxosScenarioOutcome.SAFETY_VIOLATION
        assert replay.violation is None


def test_seeded_multipaxos_lifecycle_campaign_is_exactly_replayable() -> None:
    campaign = SeededMultiPaxosCampaign(
        action_generator=SeededMultiPaxosActionGenerator(
            node_ids=("n1", "n2", "n3"),
            leadership_change_rate=0.4,
            max_events_after=6,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=0.02,
            delay_rate=0.02,
            duplicate_rate=0.02,
        ),
        lifecycle_generator=SeededLifecycleGenerator(
            nodes=("n1", "n2", "n3"),
            crash_rate=0.2,
            restart_rate=0.2,
        ),
        action_count=10,
        node_ids=("n1", "n2", "n3"),
        max_message_ordinal=8,
        final_event_budget=128,
    )

    result = campaign.run((919,))

    assert result.safety_failure is None
    assert len(result.trials) == 1
    artifact = result.trials[0]
    replay = MultiPaxosTrialArtifact.from_json(artifact.to_json()).replay()
    assert replay.outcome is artifact.outcome
    assert replay.chosen == artifact.chosen
