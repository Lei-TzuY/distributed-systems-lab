"""Portable client observations remain separate from durable server evidence."""

import copy
import json

import pytest

from distlab.kv import Delete, Put
from distlab.linearizability import Get
from distlab.multikey_linearizability import MultiKeyKVLinearizabilityChecker
from distlab.multipaxos import MultiPaxosCluster
from distlab.multipaxos_application_campaign import (
    MultiPaxosApplicationArtifact,
    MultiPaxosApplicationScenarioRunner,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction,
    MultiPaxosActionKind,
    MultiPaxosReplayMismatch,
    SeededMultiPaxosActionSchedule,
)
from distlab.multipaxos_client import ClientJournal, ClientStatus, MultiPaxosClient
from distlab.multipaxos_state_machine import Append, MultiPaxosRequest, MultiPaxosStateMachine
from distlab.randomized_faults import SeededFaultSchedule
from distlab.simulator import Simulator

NODES = ("n1", "n2", "n3")


def request(cid, rid, operation):
    return MultiPaxosRequest(cid, rid, f"{cid}-{rid}", operation)


def mixed_journal():
    journal = ClientJournal()
    requests = (
        request("A", 0, Put("a", "alpha")),
        request("B", 7, Put("b", "βeta:\n")),
        request("A", 1, Get("a")),
        request("C", 1, Append("a", "!")),
        request("B", 8, Delete("b")),
        request("B", 9, Get("b")),
        request("A", 2, Get("a")),
    )
    for item in requests:
        journal.register(item)
    journal.history.attach_client_request_id("A-0", 0)
    journal.history.attach_client_request_id("B-7", 7)
    # Response order deliberately differs from invocation order; the completed
    # write's request-ID metadata has not yet been explicitly retired.
    journal.history.respond("A-1", "alpha")
    journal.history.respond("B-7")
    journal.history.respond("B-8")
    journal.history.abandon("B-9")
    return journal


def simple_raw():
    journal = ClientJournal()
    journal.register(request("A", 1, Put("a", "alpha")))
    journal.register(request("B", 1, Get("a")))
    journal.history.respond("B-1", "alpha")
    return json.loads(journal.to_json())


def test_roundtrip_preserves_identity_event_order_pending_append_and_terminal_metadata():
    original = mixed_journal()
    before = copy.deepcopy(vars(original.history))
    encoded = original.to_json()
    assert encoded == original.to_json()
    assert encoded == json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":"))
    restored = ClientJournal.from_json(encoded)
    assert restored is not original and restored.history is not original.history
    assert restored.to_json() == encoded
    assert list(restored.requests.items()) == list(original.requests.items())
    assert all(restored.requests[k] is not v for k, v in original.requests.items())
    assert vars(restored.history) == vars(original.history) == before
    assert restored.history.is_abandoned("B-9")
    assert restored.history.client_request_id("A-0") == 0
    assert restored.history.client_request_id("B-7") == 7
    completions = sorted(restored.history.completed(), key=lambda item: item.completion.sequence)
    assert [r.operation_id for r in completions] == ["A-1", "B-7", "B-8"]
    assert "C-1" in restored.requests
    assert all(r.operation_id != "C-1" for r in restored.history.invocations())
    assert [r.operation_id for r in restored.history.pending()] == ["A-0", "B-9", "A-2"]


def test_empty_checkpoint_has_one_canonical_representation():
    encoded = ClientJournal().to_json()
    assert (
        encoded == '{"abandoned":[],"client_request_ids":[],"events":[],"requests":[],"version":1}'
    )
    assert ClientJournal.from_json(encoded).to_json() == encoded


@pytest.mark.parametrize("version", [0, 2, True, False, 1.0, "1", None])
def test_unknown_or_malformed_versions_fail_closed(version):
    raw = simple_raw()
    raw["version"] = version
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", True),
        ("request_id", -1),
        ("request_id", 1.5),
        ("request_id", "1"),
        ("client_id", ""),
        ("client_id", 4),
        ("operation_id", False),
        ("operation", {"type": "unknown", "key": "a"}),
        ("operation", {"type": "put", "key": "a", "value": False}),
        ("operation", {"type": "get", "key": 1}),
        ("operation", {"type": "append", "key": "a", "suffix": None}),
        ("operation", {"type": "get", "key": "a", "value": "extra"}),
    ],
)
def test_invalid_request_fields_fail_closed(field, value):
    raw = simple_raw()
    raw["requests"][0][field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(json.dumps(raw))


@pytest.mark.parametrize("kind", ["identical", "identity", "operation_id", "contents"])
def test_duplicate_or_conflicting_requests_are_rejected(kind):
    raw = simple_raw()
    added = copy.deepcopy(raw["requests"][0])
    if kind == "identity":
        added["operation_id"] = "another-operation"
    elif kind == "operation_id":
        added["request_id"] = 99
    elif kind == "contents":
        added["operation"]["value"] = "tampered"
    raw["requests"].append(added)
    with pytest.raises(ValueError, match="identity"):
        ClientJournal.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "corrupt",
    [
        "duplicate-completion",
        "completion-without-invocation",
        "duplicate-invocation",
        "missing-invocation",
        "unknown-operation",
        "bool-sequence",
        "float-sequence",
        "out-of-order",
        "wrong-result-type",
        "extra-event-field",
        "wrong-request-order",
        "append-in-history",
        "unknown-event-type",
    ],
)
def test_invalid_observed_events_are_rejected(corrupt):
    raw = simple_raw()
    if corrupt == "duplicate-completion":
        raw["events"].append(
            {"sequence": 3, "type": "complete", "operation_id": "B-1", "result": "alpha"}
        )
    elif corrupt == "completion-without-invocation":
        raw["events"][0] = {
            "sequence": 0,
            "type": "complete",
            "operation_id": "A-1",
            "result": None,
        }
    elif corrupt == "duplicate-invocation":
        raw["events"][1]["operation_id"] = "A-1"
    elif corrupt == "missing-invocation":
        raw["events"] = []
    elif corrupt == "unknown-operation":
        raw["events"][0]["operation_id"] = "not-registered"
    elif corrupt == "bool-sequence":
        raw["events"][0]["sequence"] = False
    elif corrupt == "float-sequence":
        raw["events"][0]["sequence"] = 0.0
    elif corrupt == "out-of-order":
        raw["events"][:2] = reversed(raw["events"][:2])
    elif corrupt == "wrong-result-type":
        raw["events"][2]["result"] = True
    elif corrupt == "extra-event-field":
        raw["events"][0]["operation"] = {"type": "put", "key": "a", "value": "conflicting"}
    elif corrupt == "wrong-request-order":
        raw["requests"].reverse()
    elif corrupt == "append-in-history":
        raw["requests"][0]["operation"] = {"type": "append", "key": "a", "suffix": "!"}
    else:
        raw["events"][0]["type"] = "submit"
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "corrupt",
    [
        "unknown-root",
        "missing-root",
        "not-list",
        "extra-request",
        "duplicate-abandonment",
        "completed-abandonment",
        "metadata-bool",
        "metadata-conflict",
        "duplicate-metadata",
    ],
)
def test_unsupported_fields_and_inconsistent_metadata_are_rejected(corrupt):
    raw = simple_raw()
    if corrupt == "unknown-root":
        raw["pickle"] = "unsupported"
    elif corrupt == "missing-root":
        del raw["abandoned"]
    elif corrupt == "not-list":
        raw["events"] = {}
    elif corrupt == "extra-request":
        raw["requests"][0]["ballot"] = 42
    elif corrupt == "duplicate-abandonment":
        raw["events"] = raw["events"][:2]
        raw["abandoned"] = ["B-1", "B-1"]
    elif corrupt == "completed-abandonment":
        raw["abandoned"] = ["B-1"]
    else:
        raw["client_request_ids"] = [{"operation_id": "A-1", "request_id": 1}]
        if corrupt == "metadata-bool":
            raw["client_request_ids"][0]["request_id"] = True
        elif corrupt == "metadata-conflict":
            raw["client_request_ids"][0]["request_id"] = 2
        else:
            raw["client_request_ids"] *= 2
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    "encoded",
    [
        None,
        b"{}",
        "not json",
        "[]",
        "null",
        '{"version":1,"version":1}',
        '{"version":NaN}',
        '{"version":Infinity}',
    ],
)
def test_malformed_json_and_duplicate_fields_fail_closed(encoded):
    with pytest.raises(ValueError, match="checkpoint"):
        ClientJournal.from_json(encoded)


def test_nested_duplicate_fields_are_rejected():
    encoded = json.dumps(simple_raw()).replace(
        '"request_id": 1', '"request_id": 1, "request_id": 1', 1
    )
    with pytest.raises(ValueError, match="duplicate JSON"):
        ClientJournal.from_json(encoded)


def test_export_does_not_silently_discard_inconsistent_history():
    journal = ClientJournal()
    journal.history.invoke("unregistered", "A", Put("a", "v"))
    with pytest.raises(ValueError, match="history conflicts"):
        journal.to_json()
    journal = mixed_journal()
    journal.history._unsupported_metadata = True
    with pytest.raises(ValueError, match="not representable"):
        journal.to_json()


def ready():
    sim = Simulator()
    cluster = MultiPaxosCluster(sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    client = MultiPaxosClient(machine)
    cluster.node("n1").prepare_leadership()
    sim.run()
    return sim, cluster, machine, client


def rebuild(sim, encoded):
    new_sim = Simulator()
    new_sim.persistent_state.update(copy.deepcopy(sim.persistent_state))
    cluster = MultiPaxosCluster(new_sim, NODES, learn_decisions=True)
    machine = MultiPaxosStateMachine(cluster)
    return new_sim, cluster, machine, MultiPaxosClient(machine, ClientJournal.from_json(encoded))


def test_completed_get_lost_ack_multikey_takeover_retry_and_follower_catchup():
    sim, cluster, _machine, client = ready()
    put_a = request("A", 1, Put("a", "alpha"))
    put_b = request("B", 1, Put("b", "beta"))
    get_a = request("A", 2, Get("a"))
    held = request("A", 3, Put("a", "new"))
    for item in (put_a, put_b, get_a):
        assert client.execute("n1", item).status == ClientStatus.COMPLETED
    assert client.history.completed()[-1].completion.result == "alpha"
    sim.partition(("n3",), ("n1", "n2"))
    client.submit("n1", held)
    sim.run()
    assert client.poll("n1", held, acknowledge=False).status == ClientStatus.APPLIED
    assert cluster.node("n3").learned_decision(4) is None
    before = copy.deepcopy(vars(client.history))
    encoded = client.journal.to_json()
    assert vars(client.history) == before
    sim.crash("n1")
    restored_sim, restored, rebuilt, restored_client = rebuild(sim, encoded)
    assert restored_sim is not sim and restored_client.journal is not client.journal
    assert vars(restored_client.history) == before
    assert restored_client.poll("n2", held).status == ClientStatus.LEADERSHIP_UNAVAILABLE
    restored.node("n2").prepare_leadership()
    restored_sim.run()
    assert restored_client.execute("n2", held).status == ClientStatus.COMPLETED
    assert len(restored_client.history.invocations()) == 4
    assert restored_client.execute("n2", get_a).value == "alpha"  # frozen old Get, current a=new
    assert len(restored_client.history.invocations()) == 4
    conflicting = MultiPaxosRequest("A", 3, held.operation_id, Put("a", "wrong"))
    assert restored_client.submit("n2", conflicting).status == ClientStatus.CONFLICT
    fresh_get = request("B", 2, Get("b"))
    assert restored_client.execute("n2", fresh_get).value == "beta"
    delete_a = request("A", 4, Delete("a"))
    assert restored_client.execute("n2", delete_a).status == ClientStatus.COMPLETED
    assert restored_client.execute("n2", request("A", 5, Get("a"))).value is None
    for _ in range(4):
        for node in NODES:
            rebuilt.apply_ready(node)
        restored_sim.run()
    for node in NODES:
        assert dict(rebuilt.state(node)) == {"b": "beta"}
        assert rebuilt.last_applied(node) == 7
        assert rebuilt.client_requests(node) == rebuilt.client_requests("n2")
    assert (
        MultiKeyKVLinearizabilityChecker()
        .check_with_pending_extensions(restored_client.history)
        .linearizable
    )
    rebuilt.assert_safety()


def test_append_identity_survives_checkpoint_and_duplicate_slot_executes_once():
    sim, _cluster, _machine, client = ready()
    put = request("A", 1, Put("k", "start"))
    append = request("A", 2, Append("k", "!"))
    assert client.execute("n1", put).status == ClientStatus.COMPLETED
    client.submit("n1", append)
    sim.run()
    assert client.poll("n1", append, acknowledge=False).status == ClientStatus.APPLIED
    restored_sim, restored, rebuilt, new_client = rebuild(sim, client.journal.to_json())
    assert list(new_client.journal.requests) == [put.operation_id, append.operation_id]
    assert [r.operation_id for r in new_client.history.invocations()] == [put.operation_id]
    restored.node("n2").prepare_leadership()
    restored_sim.run()
    assert new_client.execute("n2", append).status == ClientStatus.COMPLETED
    restored.node("n2").propose(3, append)
    restored_sim.run()
    for node in NODES:
        rebuilt.apply_ready(node)
        assert dict(rebuilt.state(node)) == {"k": "start!"}
        assert rebuilt.client_requests(node)[append.identity].slot == 2
        assert [r.slot for r in rebuilt.applied(node)] == [1, 2, 3]
    rebuilt.assert_safety()


def test_forged_checkpoint_completion_cannot_bypass_server_ack_gate():
    sim, _, machine, _ = ready()
    req = request("A", 1, Put("a", "unproven"))
    journal = ClientJournal()
    journal.register(req)
    raw = json.loads(journal.to_json())
    # Syntactically valid client claim, with no chosen/applied server evidence.
    raw["events"].append(
        {"sequence": 1, "type": "complete", "operation_id": req.operation_id, "result": None}
    )
    restored = ClientJournal.from_json(json.dumps(raw))
    client = MultiPaxosClient(machine, restored)
    assert len(client.history.completed()) == 1
    assert client.poll("n1", req, apply=False).status == ClientStatus.PENDING
    assert machine.last_applied("n1") == 0 and machine.client_requests("n1") == {}
    assert all(r.kind != "multipaxos-chosen" for r in sim.trace)


def test_forged_get_result_cannot_replace_the_real_applied_result():
    sim, _, machine, client = ready()
    assert client.execute("n1", request("A", 1, Put("a", "real"))).status == ClientStatus.COMPLETED
    read = request("A", 2, Get("a"))
    assert client.execute("n1", read).value == "real"
    raw = json.loads(client.journal.to_json())
    next(
        e
        for e in raw["events"]
        if e["type"] == "complete" and e["operation_id"] == read.operation_id
    )["result"] = "forged"
    forged = MultiPaxosClient(machine, ClientJournal.from_json(json.dumps(raw)))
    with pytest.raises(AssertionError, match="reconstructed client result changed"):
        forged.poll("n1", read)
    assert dict(machine.state("n1")) == {"a": "real"}
    sim.run()


def scenario():
    commands = (("put-a", "put:a:alpha"), ("get-a", "get:a"), ("held-b", "put:b:beta"))
    act, kind = MultiPaxosAction, MultiPaxosActionKind
    actions = [act("lead-old", kind.PREPARE, "n1", events_after=24)]
    actions += [
        act(name, kind.PROPOSE, "n1", slot=i, value=value, events_after=24)
        for i, (name, value) in enumerate(commands, 1)
    ]
    actions += [
        act("lead-new", kind.PREPARE, "n2", events_after=24),
        act("get-b", kind.PROPOSE, "n2", slot=4, value="get:b", events_after=24),
        act("delete-a", kind.PROPOSE, "n2", slot=5, value="delete:a", events_after=24),
        act("get-empty", kind.PROPOSE, "n2", slot=6, value="get:a", events_after=24),
    ]
    return SeededMultiPaxosActionSchedule(4017, tuple(actions))


def test_json_reconstruction_uses_fresh_journal_and_preserves_default_results(monkeypatch):
    actions = scenario()
    options = dict(reconstruction_boundary=4, withheld_acks=("held-b",))
    original_import = ClientJournal.from_json
    imported = []

    def tracked(cls, encoded):
        journal = original_import(encoded)
        imported.append(journal)
        return journal

    monkeypatch.setattr(ClientJournal, "from_json", classmethod(tracked))
    default = MultiPaxosApplicationScenarioRunner(actions, SeededFaultSchedule(4017, ()), **options)
    expected = default.run()
    assert imported == []  # unchanged default deepcopy path
    portable = MultiPaxosApplicationScenarioRunner(
        actions, SeededFaultSchedule(4017, ()), journal_checkpoint=True, **options
    )
    result = portable.run()
    assert len(imported) == 1
    assert [r.operation_id for r in imported[0].history.pending()] == ["held-b"]
    assert result.to_json() == expected.to_json()
    assert result.completed == (
        ("put-a", None),
        ("get-a", "alpha"),
        ("get-b", "beta"),
        ("delete-a", None),
        ("get-empty", None),
    )
    assert result.pending == ("held-b",) and result.linearizable
    artifact = MultiPaxosApplicationArtifact.capture(portable, result)
    encoded = artifact.to_json()
    assert json.loads(encoded)["journal_checkpoint"] is True
    restored = MultiPaxosApplicationArtifact.from_json(encoded)
    assert restored.to_json() == encoded and restored.replay() == result
    assert len(imported) == 2 and imported[0] is not imported[1]
    assert "journal_checkpoint" not in json.loads(
        MultiPaxosApplicationArtifact.capture(default, expected).to_json()
    )
    tampered = json.loads(encoded)
    tampered["report"]["completed"][0][1] = "wrong"
    with pytest.raises(MultiPaxosReplayMismatch):
        MultiPaxosApplicationArtifact.from_json(json.dumps(tampered)).replay()


@pytest.mark.parametrize("value", [None, 0, 1, "json", []])
def test_invalid_checkpoint_modes_are_rejected(value):
    with pytest.raises(ValueError, match="journal_checkpoint"):
        MultiPaxosApplicationScenarioRunner(
            scenario(),
            SeededFaultSchedule(4017, ()),
            reconstruction_boundary=4,
            journal_checkpoint=value,
        )


def test_json_mode_requires_a_reconstruction_boundary():
    with pytest.raises(ValueError, match="reconstruction_boundary"):
        MultiPaxosApplicationScenarioRunner(
            scenario(), SeededFaultSchedule(4017, ()), journal_checkpoint=True
        )


def test_artifact_rejects_nonboolean_checkpoint_mode():
    runner = MultiPaxosApplicationScenarioRunner(scenario(), SeededFaultSchedule(4017, ()))
    raw = json.loads(MultiPaxosApplicationArtifact.capture(runner, runner.run()).to_json())
    raw["journal_checkpoint"] = 1
    with pytest.raises(ValueError, match="artifact"):
        MultiPaxosApplicationArtifact.from_json(json.dumps(raw))
