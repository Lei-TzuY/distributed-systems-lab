"""Real application-aware Multi-Paxos campaigns with exact deterministic evidence."""

import json

import pytest

from distlab.lifecycle import (
    NodeLifecycleAction,
    NodeLifecycleKind,
    SeededLifecycleSchedule,
)
from distlab.multipaxos_application_campaign import (
    MultiPaxosApplicationArtifact,
    MultiPaxosApplicationScenarioRunner,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction,
    MultiPaxosActionKind,
    MultiPaxosScenarioOutcome,
    SeededMultiPaxosActionGenerator,
    SeededMultiPaxosActionSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule
from distlab.simulator import FaultAction, FaultRule

NODES = ("n1", "n2", "n3")


def actions(seed=73):
    return SeededMultiPaxosActionSchedule(
        seed=seed,
        actions=(
            MultiPaxosAction("lead", MultiPaxosActionKind.PREPARE, "n1", events_after=20),
            MultiPaxosAction(
                "a-put", MultiPaxosActionKind.PROPOSE, "n1", slot=1,
                value="put:k:hello", events_after=30,
            ),
            MultiPaxosAction(
                "b-get", MultiPaxosActionKind.PROPOSE, "n1", slot=2,
                value="get:k", events_after=30,
            ),
        ),
    )


def runner(schedule=None, faults=None, **options):
    schedule = schedule or actions()
    return MultiPaxosApplicationScenarioRunner(
        schedule,
        faults or SeededFaultSchedule(seed=schedule.seed, rules=()),
        **options,
    )


def test_put_get_real_completed_history_and_canonical_trace():
    a = runner().run()
    b = runner().run()
    assert a == b and a.to_json() == b.to_json()
    assert a.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert a.violation is None and a.linearizable is True
    assert a.completed == (("a-put", None), ("b-get", "hello"))
    assert a.pending == ()
    assert all(s == (1, 2) for _, s in a.applied_slots)
    assert all(state == (("k", "hello"),) for _, state in a.kv)
    raw = json.loads(a.to_json())
    assert raw["version"] == 1
    assert any(x["kind"] == "multipaxos-apply" for x in raw["trace"])
    assert json.dumps(raw, sort_keys=True, separators=(",", ":")) == a.to_json()


def test_lost_ack_remains_pending_but_completed_get_linearizes():
    result = runner(withheld_acks=("a-put",)).run()
    assert result.violation is None
    assert result.completed == (("b-get", "hello"),)
    assert result.pending == ("a-put",)
    assert result.linearizable is True
    assert all(state == (("k", "hello"),) for _, state in result.kv)
    assert runner(withheld_acks=("a-put",)).run().to_json() == result.to_json()


def test_follower_lost_decision_recovers_through_actual_peer_transport():
    faults = SeededFaultSchedule(
        seed=73,
        rules=(
            FaultRule(
                FaultAction.DROP, src="n1", dst="n3",
                payload_type="LeaderDecision",
            ),
        ),
    )
    result = runner(faults=faults).run()
    assert result.violation is None and result.linearizable is True
    assert result.completed == (("a-put", None), ("b-get", "hello"))
    assert all(slots == (1, 2) for _, slots in result.applied_slots)
    assert all(kv == (("k", "hello"),) for _, kv in result.kv)
    kinds = [r["kind"] for r in json.loads(result.trace_json)]
    assert "drop" in kinds and "multipaxos-apply" in kinds
    assert "DecisionRecoveryRequest" in result.trace_json
    assert "DecisionRecoveryResponse" in result.trace_json
    assert result.to_json() == runner(faults=faults).run().to_json()


def test_crash_restart_explicit_schedule_replays_and_durable_prefix_survives():
    lifecycle = SeededLifecycleSchedule(
        seed=73,
        actions=(
            NodeLifecycleAction("crash-n3", "n3", NodeLifecycleKind.CRASH, 2),
            NodeLifecycleAction("restart-n3", "n3", NodeLifecycleKind.RESTART, 3),
        ),
    )
    result = runner(lifecycle=lifecycle).run()
    assert result.outcome is MultiPaxosScenarioOutcome.PROGRESS
    assert result.linearizable is True
    assert all(slots == (1, 2) for _, slots in result.applied_slots)
    assert result.to_json() == runner(lifecycle=lifecycle).run().to_json()


def test_seed_compiled_schedules_replay_exactly_with_application_enabled():
    generator = SeededMultiPaxosActionGenerator(
        node_ids=NODES,
        values=("put:k:alpha", "put:k:beta", "get:k", "delete:k"),
        leadership_change_rate=0.3,
        max_events_after=6,
    )
    for seed in (7, 11, 23, 41):
        schedule = generator.compile(seed, 12)
        faults = SeededFaultSchedule(seed=seed, rules=())
        runner_a = runner(schedule, faults, application_rounds=4)
        first = runner_a.run()
        hydrated = SeededMultiPaxosActionSchedule.from_json(schedule.to_json())
        second = runner(hydrated, faults, application_rounds=4).run()
        assert first == second and first.to_json() == second.to_json()
        assert first.outcome is not MultiPaxosScenarioOutcome.SAFETY_VIOLATION


@pytest.mark.parametrize("bad_value", ["alpha", "get:", "put:", "delete:", "append:k"])
def test_invalid_application_schedule_rejected_before_execution(bad_value):
    schedule = SeededMultiPaxosActionSchedule(
        seed=13,
        actions=(MultiPaxosAction(
            "bad", MultiPaxosActionKind.PROPOSE, "n1",
            slot=1, value=bad_value,
        ),),
    )
    with pytest.raises(ValueError):
        runner(schedule)


def test_withheld_ack_must_reference_existing_proposal():
    with pytest.raises(ValueError):
        runner(withheld_acks=("not-an-action",))
    with pytest.raises(ValueError):
        runner(withheld_acks=("a-put", "a-put"))


def test_append_runs_but_is_not_falsely_checked_by_put_delete_get_oracle():
    schedule = SeededMultiPaxosActionSchedule(
        seed=13,
        actions=(
            MultiPaxosAction("lead", MultiPaxosActionKind.PREPARE, "n1", events_after=20),
            MultiPaxosAction("append", MultiPaxosActionKind.PROPOSE, "n1", slot=1,
                             value="append:k:!", events_after=30),
        ),
    )
    result = runner(schedule).run()
    assert result.linearizable is None
    assert all(state == (("k", "!"),) for _, state in result.kv)
    assert all(slots == (1,) for _, slots in result.applied_slots)


def test_application_artifact_roundtrip_and_replay_detects_tampering():
    from distlab.multipaxos_application_campaign import MultiPaxosApplicationArtifact
    from distlab.multipaxos_campaign import MultiPaxosReplayMismatch

    implementation = runner(withheld_acks=("a-put",))
    result = implementation.run()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, result)
    encoded = artifact.to_json()
    restored = MultiPaxosApplicationArtifact.from_json(encoded)
    assert encoded == restored.to_json()
    assert restored.replay() == result
    modified = json.loads(encoded)
    modified["report"]["completed"] = [["b-get", "forged"]]
    bad = MultiPaxosApplicationArtifact.from_json(
        json.dumps(modified, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(MultiPaxosReplayMismatch):
        bad.replay()


def test_no_phase1_quorum_is_incomplete_not_successful_ack():
    faults = SeededFaultSchedule(
        seed=73,
        rules=(
            FaultRule(FaultAction.DROP, src="n1", dst="n2", payload_type="LeaderPrepare"),
            FaultRule(FaultAction.DROP, src="n1", dst="n3", payload_type="LeaderPrepare"),
        ),
    )
    result = runner(faults=faults).run()
    assert result.outcome is MultiPaxosScenarioOutcome.INCOMPLETE
    assert result.completed == ()
    assert set(result.pending) == {"a-put", "b-get"}
    assert all(slots == () for _, slots in result.applied_slots)
    assert all(state == () for _, state in result.kv)
    assert result.violation is None
    assert runner(faults=faults).run() == result


def test_unsupported_multiple_keys_refuse_false_single_key_verdict():
    schedule = SeededMultiPaxosActionSchedule(
        seed=13,
        actions=(
            MultiPaxosAction("a", MultiPaxosActionKind.PROPOSE, "n1", slot=1,
                             value="put:a:v"),
            MultiPaxosAction("b", MultiPaxosActionKind.PROPOSE, "n1", slot=2,
                             value="get:b"),
        ),
    )
    with pytest.raises(ValueError, match="single-key"):
        runner(schedule)


def test_wildcard_dropped_decision_artifact_roundtrip_exact_replay():
    """Real peer recovery from dropped decisions must be a replayable artifact."""
    faults = SeededFaultSchedule(
        seed=73,
        rules=(
            FaultRule(
                FaultAction.DROP,
                src="n1",
                dst="n3",
                payload_type="LeaderDecision",
            ),
        ),
    )
    implementation = runner(faults=faults)
    original = implementation.run()
    assert original.violation is None and original.linearizable is True
    assert original.completed == (("a-put", None), ("b-get", "hello"))
    assert all(slots == (1, 2) for _, slots in original.applied_slots)
    assert all(kv == (("k", "hello"),) for _, kv in original.kv)
    trace = original.trace_json
    assert '"kind":"drop"' in trace
    assert "DecisionRecoveryRequest" in trace
    assert "DecisionRecoveryResponse" in trace

    artifact = MultiPaxosApplicationArtifact.capture(implementation, original)
    encoded = artifact.to_json()
    assert json.loads(encoded)["faults"]["rules"][0]["ordinal"] is None
    recovered = MultiPaxosApplicationArtifact.from_json(encoded)
    assert recovered.faults == faults
    assert recovered.to_json() == encoded
    assert recovered.replay().to_json() == original.to_json()


@pytest.mark.parametrize("selectors", [
    {"src": None, "dst": "n3", "ordinal": None, "payload_type": "LeaderDecision"},
    {"src": "n1", "dst": None, "ordinal": None, "payload_type": "LeaderDecision"},
    {"src": "n1", "dst": "n3", "ordinal": None, "payload_type": None},
    {"src": None, "dst": None, "ordinal": None, "payload_type": None},
    {"src": "n1", "dst": "n3", "ordinal": 2, "payload_type": "LeaderDecision"},
])
def test_application_artifact_nullable_fault_selectors_roundtrip(selectors):
    rule = FaultRule(FaultAction.DELAY, extra_delay=1, **selectors)
    schedule = SeededFaultSchedule(seed=73, rules=(rule,))
    implementation = runner(faults=schedule)
    original = implementation.run()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, original)
    encoded = artifact.to_json()
    restored = MultiPaxosApplicationArtifact.from_json(encoded)
    assert restored.faults == schedule
    assert restored.to_json() == encoded
    assert restored.replay() == original


@pytest.mark.parametrize("field,bad_value", [
    ("ordinal", True),
    ("ordinal", 0),
    ("ordinal", -1),
    ("ordinal", "1"),
    ("src", False),
    ("dst", 8),
    ("payload_type", 0),
    ("payload_type", ""),
    ("extra_delay", True),
    ("extra_delay", -1),
])
def test_application_artifact_rejects_invalid_fault_selectors(field, bad_value):
    implementation = runner()
    artifact = MultiPaxosApplicationArtifact.capture(implementation, implementation.run())
    raw = json.loads(artifact.to_json())
    raw["faults"]["rules"] = [
        {"action": "drop", "src": None, "dst": None, "ordinal": None,
         "payload_type": None, "extra_delay": 0}
    ]
    raw["faults"]["rules"][0][field] = bad_value
    with pytest.raises(ValueError, match="artifact"):
        MultiPaxosApplicationArtifact.from_json(json.dumps(raw))
