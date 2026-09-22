import distlab


def test_schedule_generation_and_reduction_are_public() -> None:
    expected = {
        "ClientWorkloadMinimizationResult",
        "LeaderGenerationGuard",
        "LifecycleScheduleMinimizationResult",
        "NodeLifecycleAction",
        "NodeLifecycleKind",
        "NonLinearizableClientWorkloadMinimizer",
        "NonLinearizableLifecycleScheduleMinimizer",
        "PaxosCluster",
        "PaxosSafetyHarness",
        "PaxosScenarioRunner",
        "PaxosTrialArtifact",
        "SeededPaxosCampaign",
        "SeededPaxosProposalGenerator",
        "ProposalNumber",
        "SeededLifecycleGenerator",
        "SeededLifecycleSchedule",
    }

    assert expected <= set(distlab.__all__)
    for name in expected:
        assert getattr(distlab, name) is not None
