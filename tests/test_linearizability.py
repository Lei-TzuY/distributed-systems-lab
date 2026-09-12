import pytest

from distlab import (
    Delete,
    Get,
    InvalidHistory,
    OperationHistory,
    Put,
    SingleKeyKVLinearizabilityChecker,
)


def test_sequential_put_get_history_is_linearizable() -> None:
    history = OperationHistory()
    history.invoke("put", "client-a", Put("x", "one"))
    history.respond("put")
    history.invoke("get", "client-b", Get("x"))
    history.respond("get", "one")

    result = SingleKeyKVLinearizabilityChecker().check(history)

    assert result.linearizable
    assert result.order == ("put", "get")


def test_real_time_order_rejects_stale_read() -> None:
    history = OperationHistory()
    history.invoke("put", "client-a", Put("x", "one"))
    history.respond("put")
    history.invoke("get", "client-b", Get("x"))
    history.respond("get", None)

    result = SingleKeyKVLinearizabilityChecker().check(history)

    assert not result.linearizable
    assert result.order == ()


def test_overlapping_operations_may_linearize_in_response_consistent_order() -> None:
    history = OperationHistory()
    history.invoke("put", "writer", Put("x", "one"))
    history.invoke("get", "reader", Get("x"))
    history.respond("get", None)
    history.respond("put")

    result = SingleKeyKVLinearizabilityChecker().check(history)

    assert result.linearizable
    assert result.order == ("get", "put")


def test_delete_updates_sequential_spec() -> None:
    history = OperationHistory()
    history.invoke("put", "writer", Put("x", "one"))
    history.respond("put")
    history.invoke("delete", "writer", Delete("x"))
    history.respond("delete")
    history.invoke("get", "reader", Get("x"))
    history.respond("get", None)

    result = SingleKeyKVLinearizabilityChecker().check(history)

    assert result.linearizable
    assert result.order == ("put", "delete", "get")


def test_pending_invocation_may_be_omitted_from_completion() -> None:
    history = OperationHistory()
    history.invoke("pending-put", "writer", Put("x", "one"))
    history.invoke("get", "reader", Get("x"))
    history.respond("get", None)

    result = SingleKeyKVLinearizabilityChecker().check(history)

    assert result.linearizable
    assert result.order == ("get",)
    assert tuple(item.operation_id for item in history.pending()) == ("pending-put",)


def test_active_client_request_identity_is_unique_until_retired() -> None:
    history = OperationHistory()
    history.invoke("first", "client-a", Put("x", "one"))
    history.attach_client_request_id("first", 7)
    history.invoke("duplicate", "client-a", Put("x", "two"))

    with pytest.raises(InvalidHistory, match="already attached to active operation 'first'"):
        history.attach_client_request_id("duplicate", 7)

    assert history.client_request_id("first") == 7
    assert history.client_request_id("duplicate") is None

    history.respond("first")
    assert history.retire_client_request_id("first") == 7
    history.attach_client_request_id("duplicate", 7)
    assert history.client_request_id("duplicate") == 7


def test_same_request_id_is_independent_across_clients() -> None:
    history = OperationHistory()
    history.invoke("client-a-write", "client-a", Put("x", "one"))
    history.attach_client_request_id("client-a-write", 3)
    history.invoke("client-b-write", "client-b", Put("x", "two"))
    history.attach_client_request_id("client-b-write", 3)

    assert history.client_request_id("client-a-write") == 3
    assert history.client_request_id("client-b-write") == 3


def test_checker_rejects_multi_key_history() -> None:
    history = OperationHistory()
    history.invoke("put-x", "writer", Put("x", "one"))
    history.respond("put-x")
    history.invoke("put-y", "writer", Put("y", "two"))
    history.respond("put-y")

    with pytest.raises(InvalidHistory, match="single-key"):
        SingleKeyKVLinearizabilityChecker().check(history)


def test_history_rejects_duplicate_invocation_and_response() -> None:
    history = OperationHistory()
    history.invoke("op", "client", Get("x"))

    with pytest.raises(InvalidHistory, match="duplicate invocation"):
        history.invoke("op", "client", Get("x"))

    history.respond("op", None)
    with pytest.raises(InvalidHistory, match="duplicate response"):
        history.respond("op", None)


def test_history_rejects_response_without_invocation() -> None:
    history = OperationHistory()

    with pytest.raises(InvalidHistory, match="without invocation"):
        history.respond("missing")


def test_deterministic_search_returns_stable_order_for_equivalent_writes() -> None:
    history = OperationHistory()
    history.invoke("first", "a", Put("x", "one"))
    history.invoke("second", "b", Put("x", "two"))
    history.respond("first")
    history.respond("second")

    checker = SingleKeyKVLinearizabilityChecker()

    first = checker.check(history)
    second = checker.check(history)

    assert first == second
    assert first.linearizable
    assert first.order == ("first", "second")
