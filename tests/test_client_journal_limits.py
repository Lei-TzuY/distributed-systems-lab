"""Bounded hostile-input recovery for opt-in portable client journal checkpoints."""

import copy
import json
import random

import pytest

from distlab.client_journal_checkpoint import (
    CheckpointLimits,
    CheckpointResourceLimitExceeded,
)
from distlab.kv import Delete, Put
from distlab.linearizability import Get
from distlab.multipaxos_application_campaign import (
    MultiPaxosApplicationArtifact,
    MultiPaxosApplicationScenarioRunner,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction,
    MultiPaxosActionKind,
    SeededMultiPaxosActionSchedule,
)
from distlab.multipaxos_client import ClientJournal
from distlab.multipaxos_state_machine import MultiPaxosRequest
from distlab.randomized_faults import SeededFaultSchedule


def source():
    journal = ClientJournal()
    operations = (
        MultiPaxosRequest("A", 0, "write-a", Put("a", "雪")),
        MultiPaxosRequest("B", 2, "read-a", Get("a")),
        MultiPaxosRequest("C", 4, "delete-b", Delete("b")),
    )
    for request in operations:
        journal.register(request)
    journal.history.attach_client_request_id("write-a", 0)
    journal.history.respond("read-a", "雪")
    return journal


def limits(**kwargs):
    base = {
        "max_bytes": 128_000,
        "max_requests": 100,
        "max_events": 200,
        "max_depth": 8,
    }
    base.update(kwargs)
    return CheckpointLimits(**base)


def test_bounded_import_restores_same_observed_state_and_exact_v1_bytes():
    original = source()
    data = original.to_json()
    prior = copy.deepcopy(vars(original.history))
    restored = ClientJournal.from_json(data, limits=limits())
    assert restored is not original
    assert restored.history is not original.history
    assert restored.to_json() == data
    assert vars(restored.history) == vars(original.history) == prior
    assert list(restored.requests.items()) == list(original.requests.items())
    assert tuple(x.operation_id for x in restored.history.pending()) == (
        "write-a", "delete-b"
    )


def test_opt_in_byte_budget_exact_boundary_and_unicode_bytes():
    data = source().to_json()
    byte_size = len(data.encode("utf-8"))
    assert ClientJournal.from_json(data, limits=limits(max_bytes=byte_size)).to_json() == data
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_bytes"):
        ClientJournal.from_json(data, limits=limits(max_bytes=byte_size - 1))
    unicode_data = json.dumps(json.loads(data), ensure_ascii=False)
    assert len(unicode_data.encode("utf-8")) > len(unicode_data)
    exact = len(unicode_data.encode("utf-8"))
    assert ClientJournal.from_json(unicode_data, limits=limits(max_bytes=exact))
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_bytes"):
        ClientJournal.from_json(unicode_data, limits=limits(max_bytes=exact - 1))


def test_request_and_event_count_limits_reject_before_object_construction():
    data = source().to_json()
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_requests"):
        ClientJournal.from_json(data, limits=limits(max_requests=2))
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_events"):
        ClientJournal.from_json(data, limits=limits(max_events=3))
    assert ClientJournal.from_json(
        data, limits=limits(max_requests=3, max_events=4)
    ).to_json() == data


@pytest.mark.parametrize("collection", ["abandoned", "client_request_ids"])
def test_auxiliary_collections_cannot_bypass_request_budget(collection):
    raw = json.loads(source().to_json())
    raw[collection] = ["a", "b", "c", "d"] if collection == "abandoned" else [
        {"operation_id": f"x{i}", "request_id": i} for i in range(4)
    ]
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_requests"):
        ClientJournal.from_json(json.dumps(raw), limits=limits(max_requests=3))


@pytest.mark.parametrize("bad", [
    0, -1, True, False, "10", 1.2, None
])
def test_limit_objects_refuse_invalid_budget_values(bad):
    for field in ("max_bytes", "max_requests", "max_events", "max_depth"):
        with pytest.raises(ValueError, match=field):
            limits(**{field: bad})


def test_import_rejects_invalid_limits_object():
    with pytest.raises(TypeError):
        ClientJournal.from_json(source().to_json(), limits={"max_bytes": 100})


def test_json_depth_budget_ignores_braces_and_quotes_inside_string():
    raw = json.loads(source().to_json())
    raw["requests"][0]["operation"]["value"] = '{[][]}"\\\\{:}'
    data = json.dumps(raw, ensure_ascii=False)
    assert ClientJournal.from_json(data, limits=limits(max_depth=4))
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_depth"):
        ClientJournal.from_json(data, limits=limits(max_depth=3))


def test_deeply_nested_malformed_json_rejected_by_resource_limit():
    data = "[" * 100 + "0" + "]" * 100
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_depth"):
        ClientJournal.from_json(data, limits=limits(max_depth=7))


def test_large_preexisting_input_rejected_without_json_parser(monkeypatch):
    import distlab.client_journal_checkpoint as codec

    def forbidden(*args, **kwargs):
        raise AssertionError("JSON parser must not be called")

    monkeypatch.setattr(codec.json, "loads", forbidden)
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_bytes"):
        ClientJournal.from_json("a" * 500, limits=limits(max_bytes=64))


def test_limits_do_not_disable_existing_schema_rejections():
    raw = json.loads(source().to_json())
    raw["requests"][0]["operation"]["type"] = "pickle"
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(json.dumps(raw), limits=limits())


def test_unbounded_legacy_api_explicitly_keeps_former_semantics():
    data = source().to_json()
    assert ClientJournal.from_json(data).to_json() == data


def campaign():
    A, K = MultiPaxosAction, MultiPaxosActionKind
    schedule = SeededMultiPaxosActionSchedule(151, (
        A("leader", K.PREPARE, "n1", events_after=24),
        A("pending-write", K.PROPOSE, "n1", slot=1,
          value="put:a:alpha", events_after=24),
        A("leader-next", K.PREPARE, "n2", events_after=24),
        A("read-a", K.PROPOSE, "n2", slot=2,
          value="get:a", events_after=24),
        A("write-b", K.PROPOSE, "n2", slot=3,
          value="put:b:beta", events_after=24),
        A("read-b", K.PROPOSE, "n2", slot=4,
          value="get:b", events_after=24),
    ))
    return schedule, SeededFaultSchedule(151, ())


def test_bounded_json_reconstruction_uses_same_real_server_evidence():
    schedule, faults = campaign()
    args = dict(
        reconstruction_boundary=2,
        journal_checkpoint=True,
        withheld_acks=("pending-write",),
    )
    default = MultiPaxosApplicationScenarioRunner(schedule, faults, **args)
    bounded = MultiPaxosApplicationScenarioRunner(
        schedule, faults, journal_limits=limits(), **args
    )
    before, after = default.run(), bounded.run()
    assert before == after
    assert after.linearizable is True and after.violation is None
    assert after.pending == ("pending-write",)
    assert after.completed == (
        ("read-a", "alpha"), ("write-b", None), ("read-b", "beta")
    )
    assert all(slots == (1, 2, 3, 4) for _, slots in after.applied_slots)
    artifact = MultiPaxosApplicationArtifact.capture(bounded, after)
    text = artifact.to_json()
    assert json.loads(text)["journal_limits"] == {
        "max_bytes": 128_000,
        "max_requests": 100,
        "max_events": 200,
        "max_depth": 8,
    }
    restored = MultiPaxosApplicationArtifact.from_json(text)
    assert restored.to_json() == text and restored.replay() == after
    assert "journal_limits" not in json.loads(
        MultiPaxosApplicationArtifact.capture(default, before).to_json()
    )


def test_bounded_reconstruction_fails_closed_instead_of_acknowledging():
    schedule, faults = campaign()
    runner = MultiPaxosApplicationScenarioRunner(
        schedule, faults, reconstruction_boundary=2,
        journal_checkpoint=True, journal_limits=limits(max_bytes=80),
        withheld_acks=("pending-write",)
    )
    with pytest.raises(CheckpointResourceLimitExceeded, match="max_bytes"):
        runner.run()


@pytest.mark.parametrize("bad", [
    {}, {"max_bytes": True}, {"max_bytes": -1},
    {"max_bytes": 10, "max_events": 1.5},
])
def test_checkpoint_artifact_rejects_forged_limits(bad):
    schedule, faults = campaign()
    runner = MultiPaxosApplicationScenarioRunner(
        schedule, faults, reconstruction_boundary=2,
        journal_checkpoint=True, withheld_acks=("pending-write",)
    )
    raw = json.loads(MultiPaxosApplicationArtifact.capture(runner, runner.run()).to_json())
    raw["journal_limits"] = bad
    with pytest.raises(ValueError, match="artifact"):
        MultiPaxosApplicationArtifact.from_json(json.dumps(raw))


@pytest.mark.parametrize("mode", [
    {"journal_checkpoint": False},
    {"reconstruction_boundary": None, "journal_checkpoint": True},
])
def test_checkpoint_limits_rejected_without_valid_json_reconstruction(mode):
    schedule, faults = campaign()
    with pytest.raises(ValueError, match="journal_limits"):
        MultiPaxosApplicationScenarioRunner(
            schedule, faults, journal_limits=limits(), **mode
        )


def test_deterministic_randomized_small_valid_checkpoints():
    rng = random.Random(20261105)
    for seed in range(48):
        journal = ClientJournal()
        n = rng.randint(0, 15)
        for i in range(n):
            kind = rng.choice(("put", "get", "delete"))
            key = rng.choice(("a", "b", "c"))
            operation = (
                Put(key, str(rng.randrange(8))) if kind == "put"
                else Get(key) if kind == "get" else Delete(key)
            )
            req = MultiPaxosRequest("seed", seed * 100 + i, f"{seed}-{i}", operation)
            journal.register(req)
            if rng.random() < 0.5:
                journal.history.respond(req.operation_id, None)
        data = journal.to_json()
        assert ClientJournal.from_json(data, limits=limits()).to_json() == data
