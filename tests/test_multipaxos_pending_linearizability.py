import pytest

from distlab.pending_history_demo import SCENARIOS, run_demo
from distlab.randomized_faults import (
    FaultOpportunity,
    SeededFaultGenerator,
    SeededFaultSchedule,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_client_lost_ack_histories_admit_extensions_without_new_responses(scenario):
    report = run_demo(scenario)
    observed = report["observed"]
    assert [c["operation_id"] for c in observed["completions"]] == ["B-get"]
    assert "A-put" in observed["pending"]
    assert not report["legacy_result"]["linearizable"]
    assert report["hypothetical_extension"] == {
        "linearizable": True,
        "order": ("A-put", "B-get"),
        "included_pending": ("A-put",),
        "omitted_pending": ("C-unchosen",) if scenario == "unchosen" else (),
    }
    assert report["history_unchanged"]
    evidence = report["protocol_evidence"]
    assert evidence["write_chosen"]
    assert evidence["write_status"] == "applied_not_acknowledged"
    assert evidence["write_slot"] == 1 and evidence["read_slot"] == 2
    assert evidence["read_status"] == "completed" and evidence["read_value"] == "v"
    if scenario == "catchup":
        assert evidence["applied_through"]["n3"] == 2
    if scenario == "unchosen":
        assert evidence["unchosen_has_quorum"] is False
        assert evidence["unchosen_has_effect"] is False
    if scenario == "retry":
        actual = report["after_actual_retry"]
        assert len(actual["observed_completions"]) == 2
        assert actual["extension"]["included_pending"] == ()
        assert actual["legacy_result"]["linearizable"]


def test_crash_catchup_and_reconstruction_evidence_replays_exactly():
    for scenario in ("restart", "catchup", "reconstruct", "unchosen"):
        assert run_demo(scenario) == run_demo(scenario)


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_fixed_seed_message_fault_schedules_replay_the_same_pending_history(seed):
    nodes = ("n1", "n2", "n3")
    opportunities = tuple(
        FaultOpportunity(src, dst, ordinal, payload)
        for src in nodes
        for dst in nodes
        if src != dst
        for ordinal in range(1, 40)
        for payload in (
            "LeaderAccept",
            "LeaderAccepted",
            "LeaderDecision",
            "DecisionRecoveryRequest",
            "DecisionRecoveryResponse",
        )
    )
    schedule = SeededFaultGenerator(
        drop_rate=0.03,
        delay_rate=0.1,
        duplicate_rate=0.1,
        max_extra_delay=3,
    ).compile(seed, opportunities)
    restored = SeededFaultSchedule.from_json(schedule.to_json())
    assert schedule == restored
    first = run_demo("takeover", schedule)
    assert sum(first["fault_actions"].values()) > 0
    assert first == run_demo("takeover", restored)
    assert first["hypothetical_extension"]["included_pending"] == ("A-put",)
    assert [c["operation_id"] for c in first["observed"]["completions"]] == ["B-get"]
