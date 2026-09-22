import json

from distlab.lifecycle import NodeLifecycleAction, NodeLifecycleKind, SeededLifecycleSchedule
from distlab.paxos import ProposalNumber
from distlab.paxos_campaign import (
    PaxosProposalAction,
    PaxosScenarioOutcome,
    PaxosScenarioRunner,
    PaxosTrialArtifact,
    SeededPaxosCampaign,
    SeededPaxosProposalGenerator,
    SeededPaxosProposalSchedule,
    paxos_fault_opportunities,
)
from distlab.randomized_faults import SeededFaultGenerator, SeededFaultSchedule
from distlab.simulator import FaultAction, FaultRule


def _no_faults(seed: int) -> SeededFaultSchedule:
    return SeededFaultSchedule(seed=seed, rules=())


def test_seeded_paxos_proposals_compile_deterministically() -> None:
    generator = SeededPaxosProposalGenerator(
        node_ids=("n1", "n2", "n3"),
        values=("alpha", "beta"),
        max_events_after=3,
    )

    first = generator.compile(1729, 6)
    second = generator.compile(1729, 6)

    assert first == second
    assert SeededPaxosProposalSchedule.from_json(first.to_json()) == first
    assert len(first.actions) == 6
    assert all(action.proposer_id in {"n1", "n2", "n3"} for action in first.actions)
    assert all(action.value in {"alpha", "beta"} for action in first.actions)
    assert all(0 <= action.events_after <= 3 for action in first.actions)


def test_explicit_paxos_scenario_replays_concurrent_proposals_without_rng() -> None:
    proposals = SeededPaxosProposalSchedule(
        seed=11,
        actions=(
            PaxosProposalAction("p1", "n1", "alpha", events_after=0, round_number=1),
            PaxosProposalAction("p2", "n2", "beta", events_after=0, round_number=1),
        ),
    )

    first = PaxosScenarioRunner(proposals, _no_faults(11)).run()
    second = PaxosScenarioRunner(proposals, _no_faults(11)).run()

    assert first == second
    assert first.outcome is PaxosScenarioOutcome.CHOSEN
    assert first.chosen is not None
    assert first.chosen.proposal == ProposalNumber(1, "n2")
    assert first.chosen.value == "beta"
    assert first.violation is None


def test_trial_artifact_round_trip_replays_chosen_trace_exactly() -> None:
    proposals = SeededPaxosProposalSchedule(
        seed=23,
        actions=(
            PaxosProposalAction("p1", "n1", "alpha", events_after=0),
            PaxosProposalAction("p2", "n2", "beta", events_after=2),
        ),
    )
    faults = SeededFaultSchedule(
        seed=23,
        rules=(
            FaultRule(
                FaultAction.DUPLICATE,
                src="n1",
                dst="n3",
                ordinal=1,
                payload_type="Prepare",
                extra_delay=2,
            ),
        ),
    )
    result = PaxosScenarioRunner(proposals, faults).run()
    artifact = PaxosTrialArtifact.capture(proposals, faults, result)

    restored = PaxosTrialArtifact.from_json(artifact.to_json())
    replay = restored.replay()

    assert restored == artifact
    assert replay == result
    assert replay.outcome is PaxosScenarioOutcome.CHOSEN
    assert json.loads(restored.trace_json)


def test_incomplete_quorum_is_replayable_evidence_not_safety_failure() -> None:
    proposals = SeededPaxosProposalSchedule(
        seed=31,
        actions=(PaxosProposalAction("p1", "n1", "alpha"),),
    )
    faults = SeededFaultSchedule(
        seed=31,
        rules=(
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst="n2",
                ordinal=1,
                payload_type="Prepare",
            ),
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst="n3",
                ordinal=1,
                payload_type="Prepare",
            ),
        ),
    )

    result = PaxosScenarioRunner(proposals, faults).run()
    artifact = PaxosTrialArtifact.capture(proposals, faults, result)

    assert result.outcome is PaxosScenarioOutcome.INCOMPLETE
    assert result.chosen is None
    assert result.violation is None
    assert artifact.replay() == result


def test_crash_restart_lifecycle_is_part_of_exact_paxos_replay() -> None:
    proposals = SeededPaxosProposalSchedule(
        seed=41,
        actions=(
            PaxosProposalAction("p1", "n1", "alpha", events_after=1),
            PaxosProposalAction("p2", "n2", "beta", events_after=2),
        ),
    )
    lifecycle = SeededLifecycleSchedule(
        seed=41,
        actions=(
            NodeLifecycleAction(
                "crash-n3",
                "n3",
                NodeLifecycleKind.CRASH,
                before_action_index=0,
            ),
            NodeLifecycleAction(
                "restart-n3",
                "n3",
                NodeLifecycleKind.RESTART,
                before_action_index=1,
            ),
        ),
    )
    result = PaxosScenarioRunner(
        proposals,
        _no_faults(41),
        lifecycle=lifecycle,
    ).run()
    artifact = PaxosTrialArtifact.capture(
        proposals,
        _no_faults(41),
        result,
        lifecycle=lifecycle,
    )

    restored = PaxosTrialArtifact.from_json(artifact.to_json())

    assert restored.lifecycle == lifecycle
    assert restored.replay() == result
    assert [
        record.kind
        for record in result.trace
        if record.kind == "scenario-paxos-lifecycle"
    ] == ["scenario-paxos-lifecycle", "scenario-paxos-lifecycle"]


def test_paxos_fault_opportunities_cover_protocol_types_per_link_ordinal() -> None:
    opportunities = paxos_fault_opportunities(
        ("n1", "n2", "n3"),
        max_message_ordinal=2,
    )

    assert len(opportunities) == 3 * 2 * 2 * 5
    assert {
        opportunity.payload_type
        for opportunity in opportunities
        if opportunity.src == "n1"
        and opportunity.dst == "n2"
        and opportunity.ordinal == 1
    } == {"Prepare", "Promise", "AcceptRequest", "Accepted", "Learn"}


def test_seeded_paxos_campaign_preserves_safety_and_replays_every_trial() -> None:
    campaign = SeededPaxosCampaign(
        proposal_generator=SeededPaxosProposalGenerator(
            node_ids=("n1", "n2", "n3"),
            values=("alpha", "beta", "gamma"),
            max_events_after=2,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=0.05,
            delay_rate=0.05,
            duplicate_rate=0.05,
            max_extra_delay=3,
        ),
        proposal_count=4,
        node_ids=("n1", "n2", "n3"),
        max_message_ordinal=8,
        final_event_budget=96,
    )

    result = campaign.run((101, 202, 303, 404))

    assert result.attempted_seeds == (101, 202, 303, 404)
    assert result.safety_failure is None
    assert len(result.trials) == 4
    for artifact in result.trials:
        replay = PaxosTrialArtifact.from_json(artifact.to_json()).replay()
        assert replay.outcome is not PaxosScenarioOutcome.SAFETY_VIOLATION
        assert replay.violation is None


def test_seeded_paxos_campaign_with_lifecycle_faults_is_exactly_replayable() -> None:
    from distlab.lifecycle import SeededLifecycleGenerator

    campaign = SeededPaxosCampaign(
        proposal_generator=SeededPaxosProposalGenerator(
            node_ids=("n1", "n2", "n3"),
            max_events_after=1,
        ),
        fault_generator=SeededFaultGenerator(
            drop_rate=0.02,
            delay_rate=0.02,
            duplicate_rate=0.02,
        ),
        lifecycle_generator=SeededLifecycleGenerator(
            nodes=("n1", "n2", "n3"),
            crash_rate=0.25,
            restart_rate=0.25,
        ),
        proposal_count=5,
        node_ids=("n1", "n2", "n3"),
        max_message_ordinal=6,
        final_event_budget=64,
    )

    result = campaign.run((919,))

    assert result.safety_failure is None
    assert len(result.trials) == 1
    artifact = result.trials[0]
    assert PaxosTrialArtifact.from_json(artifact.to_json()).replay().outcome is artifact.outcome
