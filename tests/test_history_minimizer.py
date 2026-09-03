import pytest

from distlab import (
    Get,
    NonLinearizableHistoryMinimizer,
    OperationHistory,
    Put,
    SingleKeyKVLinearizabilityChecker,
)


def _failing_history() -> OperationHistory:
    history = OperationHistory()
    history.invoke("initial-read", "reader", Get("x"))
    history.respond("initial-read", None)
    history.invoke("write", "writer", Put("x", "one"))
    history.respond("write")
    history.invoke("stale-read", "reader", Get("x"))
    history.respond("stale-read", None)
    history.invoke("pending-noise", "writer", Put("x", "later"))
    return history


def test_minimizer_reduces_to_one_minimal_non_linearizable_witness() -> None:
    checker = SingleKeyKVLinearizabilityChecker()
    result = NonLinearizableHistoryMinimizer(checker).minimize(_failing_history())

    assert result.operation_ids == ("write", "stale-read")
    assert result.removed_operation_ids == ("initial-read", "pending-noise")
    assert not checker.check(result.history).linearizable
    assert tuple(item.operation_id for item in result.history.completed()) == (
        "write",
        "stale-read",
    )
    assert result.history.pending() == ()


def test_minimizer_is_deterministic_for_same_history() -> None:
    minimizer = NonLinearizableHistoryMinimizer()

    first = minimizer.minimize(_failing_history())
    second = minimizer.minimize(_failing_history())

    assert first.operation_ids == second.operation_ids
    assert first.removed_operation_ids == second.removed_operation_ids
    assert first.history.invocations() == second.history.invocations()
    assert first.history.completed() == second.history.completed()


def test_minimized_history_preserves_relative_event_order() -> None:
    result = NonLinearizableHistoryMinimizer().minimize(_failing_history())
    completed = result.history.completed()

    assert completed[0].operation_id == "write"
    assert completed[0].completion.sequence < completed[1].invocation.sequence
    assert completed[1].operation_id == "stale-read"


def test_minimizer_rejects_linearizable_history() -> None:
    history = OperationHistory()
    history.invoke("write", "writer", Put("x", "one"))
    history.respond("write")
    history.invoke("read", "reader", Get("x"))
    history.respond("read", "one")

    with pytest.raises(ValueError, match="must be non-linearizable"):
        NonLinearizableHistoryMinimizer().minimize(history)
