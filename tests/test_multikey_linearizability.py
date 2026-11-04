"""Independent-key locality, pending completion, and deterministic safety probes."""

import copy
import itertools
import random

import pytest

from distlab.kv import Delete, Put
from distlab.linearizability import (
    Get,
    InvalidHistory,
    LinearizabilitySearchExhausted,
    OperationHistory,
    SingleKeyKVLinearizabilityChecker,
)
from distlab.multikey_linearizability import MultiKeyKVLinearizabilityChecker


def build_history(*events):
    history = OperationHistory()
    for event in events:
        if event[0] == "invoke":
            _, identity, client, operation = event
            history.invoke(identity, client, operation)
        else:
            _, identity, response = event
            history.respond(identity, response)
    return history


def test_two_keys_have_independent_completed_observed_values():
    history = build_history(
        ("invoke", "a1", "A", Put("a", "alpha")),
        ("invoke", "b1", "B", Put("b", "beta")),
        ("response", "b1", None),
        ("response", "a1", None),
        ("invoke", "a2", "A", Get("a")),
        ("invoke", "b2", "B", Get("b")),
        ("response", "b2", "beta"),
        ("response", "a2", "alpha"),
    )
    result = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert result.linearizable
    assert tuple(key for key, _ in result.by_key) == ("a", "b")
    assert result.by_key[0][1].order == ("a1", "a2")
    assert result.by_key[1][1].order == ("b1", "b2")
    assert result == MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(history)


def test_pending_write_on_one_key_and_independent_write_on_other():
    history = build_history(
        ("invoke", "a-put", "A", Put("a", "alpha")),
        ("invoke", "b-put", "B", Put("b", "beta")),
        ("response", "b-put", None),
        ("invoke", "a-get", "C", Get("a")),
        ("response", "a-get", "alpha"),
        ("invoke", "b-get", "D", Get("b")),
        ("response", "b-get", "beta"),
    )
    before = copy.deepcopy(vars(history))
    result = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert result.linearizable
    assert result.by_key[0][1].included_pending == ("a-put",)
    assert result.by_key[1][1].included_pending == ()
    assert set(item.operation_id for item in history.pending()) == {"a-put"}
    assert vars(history) == before


def test_incomplete_delete_may_explain_absence_only_on_its_key():
    h = build_history(
        ("invoke", "bput", "A", Put("b", "B")),
        ("response", "bput", None),
        ("invoke", "aput", "A", Put("a", "A")),
        ("response", "aput", None),
        ("invoke", "bdelete", "B", Delete("b")),
        ("invoke", "bget", "C", Get("b")),
        ("response", "bget", None),
        ("invoke", "aget", "C", Get("a")),
        ("response", "aget", "A"),
    )
    result = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    assert result.linearizable
    assert dict(result.by_key)["b"].included_pending == ("bdelete",)


def test_cross_key_value_cannot_repair_a_stale_read():
    h = build_history(
        ("invoke", "write-a", "A", Put("a", "secret")),
        ("response", "write-a", None),
        ("invoke", "read-b", "B", Get("b")),
        ("response", "read-b", "secret"),
    )
    result = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    assert result.linearizable is False
    assert dict(result.by_key)["a"].linearizable
    assert dict(result.by_key)["b"].linearizable is False


def test_unrelated_pending_write_cannot_fabricate_value_on_another_key():
    h = build_history(
        ("invoke", "w", "A", Put("a", "v")),
        ("invoke", "r", "B", Get("b")),
        ("response", "r", "v"),
    )
    r = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    assert not r.linearizable
    assert dict(r.by_key)["a"].linearizable
    assert not dict(r.by_key)["b"].linearizable


def test_original_single_key_checker_still_rejects_multi_key_inputs():
    h = build_history(
        ("invoke", "a", "A", Put("a", "v")),
        ("response", "a", None),
        ("invoke", "b", "B", Get("b")),
        ("response", "b", None),
    )
    with pytest.raises(InvalidHistory, match="single-key"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    assert MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(h).linearizable


def test_one_key_result_is_identical_to_legacy_pending_extension():
    h = build_history(
        ("invoke", "a", "A", Put("k", "hello")),
        ("invoke", "b", "B", Get("k")),
        ("response", "b", "hello"),
    )
    one = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    many = MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(h)
    assert many.linearizable == one.linearizable
    assert many.by_key == (("k", one),)


def test_empty_and_pending_reads_do_not_fabricate_completions():
    checker = MultiKeyKVLinearizabilityChecker()
    assert checker.check_with_pending_extensions(OperationHistory()).by_key == ()
    h = build_history(("invoke", "a", "A", Get("b")))
    r = checker.check_with_pending_extensions(h)
    assert r.linearizable
    assert r.by_key[0][1].order == ()
    assert r.by_key[0][1].omitted_pending == ("a",)
    assert not h.completed()


@pytest.mark.parametrize("kwargs", [
    {"max_operations": 0}, {"max_operations": True}, {"max_operations": 65},
    {"max_keys": 0}, {"max_keys": False}, {"max_keys": 65},
    {"max_states_per_key": 0}, {"max_states_per_key": True},
])
def test_invalid_search_limits_fail_closed(kwargs):
    with pytest.raises(ValueError):
        MultiKeyKVLinearizabilityChecker().check_with_pending_extensions(
            OperationHistory(), **kwargs
        )


def test_total_operations_and_key_count_limits_exhaust_separately():
    h = build_history(
        ("invoke", "a", "A", Get("a")),
        ("invoke", "b", "B", Get("b")),
    )
    checker = MultiKeyKVLinearizabilityChecker()
    with pytest.raises(LinearizabilitySearchExhausted, match="max_operations"):
        checker.check_with_pending_extensions(h, max_operations=1)
    with pytest.raises(LinearizabilitySearchExhausted, match="max_keys"):
        checker.check_with_pending_extensions(h, max_keys=1)
    h2 = build_history(
        ("invoke", "w", "A", Put("a", "v")),
        ("invoke", "r", "B", Get("a")),
        ("response", "r", "v"),
    )
    with pytest.raises(LinearizabilitySearchExhausted, match="max_states"):
        checker.check_with_pending_extensions(h2, max_states_per_key=1)


def reference_bruteforce_global(history):
    """Oracle separate from key projection and DFS: enumerate global extensions."""
    invocations = history.invocations()
    actual = {x.operation_id: x.completion for x in history.completed()}
    mandatory = [inv for inv in invocations if inv.operation_id in actual]
    optional = [
        inv for inv in invocations
        if inv.operation_id not in actual and isinstance(inv.operation, (Put, Delete))
    ]
    for n in range(len(optional) + 1):
        for chosen in itertools.combinations(optional, n):
            for order in itertools.permutations((*mandatory, *chosen)):
                positions = {inv.operation_id: i for i, inv in enumerate(order)}
                if any(
                    completion.sequence < second.sequence
                    and positions[first] > positions[second.operation_id]
                    for first, completion in actual.items()
                    for second in order
                ):
                    continue
                values = {}
                for invocation in order:
                    operation = invocation.operation
                    if isinstance(operation, Put):
                        values[operation.key] = operation.value
                    elif isinstance(operation, Delete):
                        values.pop(operation.key, None)
                    elif actual[invocation.operation_id].result != values.get(operation.key):
                        break
                else:
                    return True
    return False


def test_200_fixed_seed_histories_agree_with_global_exhaustive_oracle():
    rng = random.Random(20261104)
    checker = MultiKeyKVLinearizabilityChecker()
    keys = ("a", "b", "c")
    for trial in range(200):
        h = OperationHistory()
        pending = []
        for index in range(rng.randint(1, 6)):
            identity = f"id-{index}"
            key = rng.choice(keys)
            name = rng.choice(("put", "del", "get"))
            operation = (
                Put(key, rng.choice(("a", "b", "c")))
                if name == "put" else Delete(key) if name == "del" else Get(key)
            )
            h.invoke(identity, f"client-{index}", operation)
            pending.append((identity, operation))
            if pending and rng.random() < 0.65:
                i = rng.randrange(len(pending))
                opid, command = pending.pop(i)
                response = rng.choice((None, "a", "b", "c")) if isinstance(command, Get) else None
                h.respond(opid, response)
        if pending and rng.random() < 0.35:
            i = rng.randrange(len(pending))
            opid, command = pending.pop(i)
            response = rng.choice((None, "a", "b", "c")) if isinstance(command, Get) else None
            h.respond(opid, response)
        before = copy.deepcopy(vars(h))
        result = checker.check_with_pending_extensions(h, max_states_per_key=100000)
        expected = reference_bruteforce_global(h)
        assert result.linearizable == expected, (
            f"trial={trial}, expected={expected}, verdict={result}, "
            f"events={h.invocations()}, completed={h.completed()}"
        )
        assert result == checker.check_with_pending_extensions(h)
        assert vars(h) == before


def test_public_package_export_resolves_to_same_oracle():
    from distlab import (
        MultiKeyCompletionExtensionResult as RootResult,
    )
    from distlab import (
        MultiKeyKVLinearizabilityChecker as RootChecker,
    )
    from distlab.multikey_linearizability import MultiKeyCompletionExtensionResult
    assert RootChecker is MultiKeyKVLinearizabilityChecker
    assert RootResult is MultiKeyCompletionExtensionResult
