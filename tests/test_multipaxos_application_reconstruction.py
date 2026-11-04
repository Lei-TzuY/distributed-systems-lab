"""Opt-in full Simulator reconstruction in application-aware fault campaigns."""

import json

import pytest

from distlab.multipaxos_application_campaign import (
    MultiPaxosApplicationArtifact,
    MultiPaxosApplicationScenarioRunner,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction,
    MultiPaxosActionKind,
    MultiPaxosReplayMismatch,
    MultiPaxosScenarioOutcome,
    SeededMultiPaxosActionSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule
from distlab.simulator import FaultAction, FaultRule

NODES = ("n1", "n2", "n3")


def actions():
    action = MultiPaxosAction
    kind = MultiPaxosActionKind
    return SeededMultiPaxosActionSchedule(17, (
        action("lead-n1", kind.PREPARE, "n1", events_after=30),
        action("a-put", kind.PROPOSE, "n1", slot=1,
               value="put:a:alpha", events_after=40),
        action("lead-n2", kind.PREPARE, "n2", events_after=40),
        action("b-get", kind.PROPOSE, "n2", slot=2,
               value="get:a", events_after=40),
        action("c-put", kind.PROPOSE, "n2", slot=3,
               value="put:b:beta", events_after=40),
        action("d-get", kind.PROPOSE, "n2", slot=4,
               value="get:b", events_after=40),
    ))


def runner(*, reconstruction_boundary=2, withheld_acks=("a-put",), faults=None,
           schedule=None):
    return MultiPaxosApplicationScenarioRunner(
        schedule or actions(),
        faults if faults is not None else SeededFaultSchedule(17, ()),
        reconstruction_boundary=reconstruction_boundary,
        withheld_acks=withheld_acks,
    )


def markers(result):
    return [
        record for record in json.loads(result.trace_json)
        if record["kind"] == "multipaxos-full-reconstruction"
    ]


def test_actual_full_reconstruction_then_takeover_log_reads_and_pending_ack():
    implementation = runner()
    result = implementation.run()
    assert result.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert result.violation is None and result.linearizable is True
    assert result.completed == (
        ("b-get", "alpha"), ("c-put", None), ("d-get", "beta"),
    )
    assert result.pending == ("a-put",)
    assert all(slots == (1, 2, 3, 4) for _, slots in result.applied_slots)
    assert all(
        kv == (("a", "alpha"), ("b", "beta"))
        for _, kv in result.kv
    )
    assert [r["details"] for r in markers(result)] == [
        {"boundary": 2, "phase": "stop"},
        {"boundary": 2, "phase": "start"},
    ]
    raw = json.loads(result.trace_json)
    assert any("LeaderPrepare" in json.dumps(x) for x in raw)
    assert implementation.run().to_json() == result.to_json()


def test_reconstruction_artifact_contains_checkpoint_and_replays_byte_identically():
    implementation = runner()
    result = implementation.run()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, result)
    encoded = artifact.to_json()
    assert json.loads(encoded)["reconstruction_boundary"] == 2
    restored = MultiPaxosApplicationArtifact.from_json(encoded)
    assert restored.reconstruction_boundary == 2
    assert restored.to_json() == encoded
    assert restored.replay().to_json() == result.to_json()
    assert MultiPaxosApplicationArtifact.from_json(encoded).replay() == result


def test_rebuilding_at_zero_is_equivalent_to_fresh_boot_and_no_fabrication():
    schedule = actions()
    initial = MultiPaxosApplicationScenarioRunner(
        schedule, SeededFaultSchedule(17, ()),
    ).run()
    rebuilt = runner(
        reconstruction_boundary=0, withheld_acks=(),
    ).run()
    assert rebuilt.completed == initial.completed
    assert rebuilt.pending == initial.pending
    assert rebuilt.kv == initial.kv
    assert rebuilt.applied_slots == initial.applied_slots
    assert rebuilt.linearizable == initial.linearizable is True
    assert [r["details"]["phase"] for r in markers(rebuilt)] == ["stop", "start"]


def test_rebuilding_at_end_keeps_completed_client_journal_and_durable_projection():
    r = runner(reconstruction_boundary=len(actions().actions), withheld_acks=())
    result = r.run()
    assert result.pending == ()
    assert result.completed == (
        ("a-put", None), ("b-get", "alpha"),
        ("c-put", None), ("d-get", "beta"),
    )
    assert result.linearizable is True
    assert all(slots == (1, 2, 3, 4) for _, slots in result.applied_slots)
    assert MultiPaxosApplicationArtifact.capture(r, result).replay() == result


def test_missing_post_rebuild_phase1_must_not_publish_read():
    original = actions().actions
    schedule = SeededMultiPaxosActionSchedule(17, (
        original[0],
        original[1],
        MultiPaxosAction("no-new-leader-read", MultiPaxosActionKind.PROPOSE,
                         "n1", slot=2, value="get:a", events_after=30),
    ))
    result = runner(schedule=schedule, withheld_acks=()).run()
    assert result.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert result.completed == (("a-put", None),)
    assert result.pending == ("no-new-leader-read",)
    assert result.linearizable is True
    assert all(slots == (1,) for _, slots in result.applied_slots)
    assert any("leadership_unavailable" in json.dumps(x)
               for x in json.loads(result.trace_json))


def test_lossy_peer_decision_and_full_reconstruction_remain_replayable():
    faults = SeededFaultSchedule(17, (
        FaultRule(FaultAction.DROP, src="n1", dst="n3",
                  payload_type="LeaderDecision"),
    ))
    implementation = runner(faults=faults)
    result = implementation.run()
    assert result.linearizable and result.violation is None
    assert result.pending == ("a-put",)
    assert all(slots == (1, 2, 3, 4) for _, slots in result.applied_slots)
    serialized = MultiPaxosApplicationArtifact.capture(implementation, result).to_json()
    assert json.loads(serialized)["faults"]["rules"][0]["ordinal"] is None
    assert MultiPaxosApplicationArtifact.from_json(serialized).replay() == result
    assert "drop" in result.trace_json
    assert "DecisionRecoveryRequest" in result.trace_json
    assert "DecisionRecoveryResponse" in result.trace_json


def test_append_after_full_reconstruction_preserves_prefix_but_no_wrong_oracle():
    kind = MultiPaxosActionKind
    act = MultiPaxosAction
    schedule = SeededMultiPaxosActionSchedule(17, (
        act("first-lead", kind.PREPARE, "n1", events_after=30),
        act("initial-append", kind.PROPOSE, "n1", slot=1,
            value="append:k:one", events_after=40),
        act("new-lead", kind.PREPARE, "n2", events_after=40),
        act("next-append", kind.PROPOSE, "n2", slot=2,
            value="append:k:two", events_after=40),
    ))
    result = runner(schedule=schedule, withheld_acks=()).run()
    assert result.linearizable is None
    assert all(state == (("k", "onetwo"),) for _, state in result.kv)
    assert all(slots == (1, 2) for _, slots in result.applied_slots)


@pytest.mark.parametrize("bad", [-1, True, False, 7, 1.5, "2"])
def test_invalid_reconstruction_boundaries_fail_before_execution(bad):
    with pytest.raises(ValueError, match="reconstruction_boundary"):
        runner(reconstruction_boundary=bad)


@pytest.mark.parametrize("bad", [-1, True, 999, "3"])
def test_invalid_artifact_checkpoint_rejected(bad):
    implementation = runner()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, implementation.run())
    raw = json.loads(artifact.to_json())
    raw["reconstruction_boundary"] = bad
    with pytest.raises(ValueError, match="artifact"):
        MultiPaxosApplicationArtifact.from_json(json.dumps(raw))


def test_tampered_checkpoint_cannot_replay_other_checkpoint_evidence():
    implementation = runner()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, implementation.run())
    raw = json.loads(artifact.to_json())
    raw["reconstruction_boundary"] = 1
    alternate = MultiPaxosApplicationArtifact.from_json(json.dumps(raw))
    with pytest.raises(MultiPaxosReplayMismatch):
        alternate.replay()


def test_default_artifact_omits_checkpoint_key_and_remains_legacy_compatible():
    normal = runner(reconstruction_boundary=None, withheld_acks=())
    result = normal.run()
    encoded = MultiPaxosApplicationArtifact.capture(normal, result).to_json()
    assert "reconstruction_boundary" not in json.loads(encoded)
    assert MultiPaxosApplicationArtifact.from_json(encoded).to_json() == encoded
