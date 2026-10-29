import copy
import itertools

import pytest

from distlab.history_minimizer import NonLinearizableHistoryMinimizer
from distlab.kv import Delete, Put
from distlab.linearizability import (
    Get,
    InvalidHistory,
    LinearizabilitySearchExhausted,
    OperationHistory,
    SingleKeyKVLinearizabilityChecker,
)
from distlab.multipaxos_state_machine import Append


def lost_ack_history():
    history = OperationHistory()
    history.invoke("write", "A", Put("k", "v"))
    history.attach_client_request_id("write", 7)
    history.invoke("read", "B", Get("k"))
    history.respond("read", "v")
    return history


def replay_witness(history, result):
    """Independent sequential specification and real-time replay of a positive witness."""
    assert result.linearizable
    invocations = {i.operation_id: i for i in history.invocations()}
    completed = {c.operation_id: c for c in history.completed()}
    assert len(result.order) == len(set(result.order))
    assert set(completed) <= set(result.order) <= set(invocations)
    assert result.included_pending == tuple(i for i in result.order if i not in completed)
    assert result.omitted_pending == tuple(
        i.operation_id for i in history.pending() if i.operation_id not in result.order
    )
    positions = {operation_id: index for index, operation_id in enumerate(result.order)}
    for earlier in completed.values():
        for later in result.order:
            if earlier.completion.sequence < invocations[later].sequence:
                assert positions[earlier.operation_id] < positions[later]
    value = None
    for operation_id in result.order:
        operation = invocations[operation_id].operation
        if isinstance(operation, Put):
            value = operation.value
        elif isinstance(operation, Delete):
            value = None
        else:
            assert isinstance(operation, Get) and operation_id in completed
            assert completed[operation_id].completion.result == value


def test_lost_ack_pending_put_explains_read_without_fabricating_completion():
    history = lost_ack_history()
    original = copy.deepcopy(vars(history))
    checker = SingleKeyKVLinearizabilityChecker()
    assert not checker.check(history).linearizable
    result = checker.check_with_pending_extensions(history)
    assert result.linearizable
    assert result.order == ("write", "read")
    assert result.included_pending == ("write",)
    assert result.omitted_pending == ()
    replay_witness(history, result)
    assert vars(history) == original
    assert history.client_request_id("write") == 7
    assert [r.operation_id for r in history.completed()] == ["read"]
    assert not checker.check(history).linearizable  # Opt-in does not alter the default.


def test_impossible_value_has_no_legal_completion_extension():
    history = OperationHistory()
    history.invoke("write", "A", Put("k", "v"))
    history.invoke("read", "B", Get("k"))
    history.respond("read", "forged")
    result = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert not result.linearizable
    assert result.order == result.included_pending == result.omitted_pending == ()


def test_completed_stale_read_cannot_move_before_a_completed_write():
    history = OperationHistory()
    history.invoke("write", "A", Put("k", "new"))
    history.respond("write")
    history.invoke("read", "B", Get("k"))
    history.respond("read", None)
    assert (
        not SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history).linearizable
    )


def test_future_pending_delete_cannot_explain_an_earlier_stale_read():
    history = OperationHistory()
    history.invoke("write", "A", Put("k", "new"))
    history.respond("write")
    history.invoke("read", "B", Get("k"))
    history.respond("read", None)
    history.invoke("too-late-delete", "C", Delete("k"))
    assert (
        not SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history).linearizable
    )


def test_pending_delete_may_explain_observed_absence():
    history = OperationHistory()
    history.invoke("write", "A", Put("k", "v"))
    history.respond("write")
    history.invoke("delete", "B", Delete("k"))
    history.invoke("read", "C", Get("k"))
    history.respond("read", None)
    result = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert result.order == ("write", "delete", "read")
    assert result.included_pending == ("delete",)
    replay_witness(history, result)


def test_pending_write_has_no_imaginary_early_response_edge():
    history = OperationHistory()
    history.invoke("pending", "A", Put("k", "old"))
    history.invoke("completed", "B", Put("k", "new"))
    history.respond("completed")
    history.invoke("read", "C", Get("k"))
    history.respond("read", "old")
    result = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    # The pending write overlaps both completed operations and can linearize
    # between them. Invocation order is not an imaginary response time.
    assert result.order == ("completed", "pending", "read")
    replay_witness(history, result)


def test_irrelevant_pending_writes_and_unobserved_reads_are_omitted():
    history = OperationHistory()
    history.invoke("irrelevant", "A", Put("k", "noise"))
    history.invoke("pending-read", "B", Get("k"))
    history.invoke("abandoned-read", "C", Get("k"))
    history.abandon("abandoned-read")
    history.invoke("read", "D", Get("k"))
    history.respond("read", None)
    before = copy.deepcopy(vars(history))
    result = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert result.order == ("read",)
    assert result.included_pending == ()
    assert result.omitted_pending == ("irrelevant", "pending-read", "abandoned-read")
    assert vars(history) == before
    replay_witness(history, result)


def test_two_pending_candidates_can_both_be_needed():
    history = OperationHistory()
    history.invoke("a", "A", Put("k", "a"))
    history.invoke("b", "B", Put("k", "b"))
    history.invoke("read-b", "C", Get("k"))
    history.respond("read-b", "b")
    history.invoke("read-a", "C", Get("k"))
    history.respond("read-a", "a")
    result = SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    assert result.order == ("b", "read-b", "a", "read-a")
    assert result.included_pending == ("b", "a")
    replay_witness(history, result)


def test_empty_and_pending_only_histories_need_no_hypothetical_effects():
    history = OperationHistory()
    checker = SingleKeyKVLinearizabilityChecker()
    assert checker.check_with_pending_extensions(history).order == ()
    history.invoke("write", "A", Put("k", "v"))
    history.invoke("read", "B", Get("k"))
    result = checker.check_with_pending_extensions(history)
    assert result.linearizable and result.order == result.included_pending == ()
    assert result.omitted_pending == ("write", "read")
    replay_witness(history, result)


def test_repeated_search_is_deterministic_and_never_retires_client_metadata():
    history = lost_ack_history()
    before = copy.deepcopy(vars(history))
    checker = SingleKeyKVLinearizabilityChecker()
    results = [checker.check_with_pending_extensions(history) for _ in range(10)]
    assert all(r == results[0] for r in results)
    assert vars(history) == before


@pytest.mark.parametrize("pending_operation", [Put("other", "v"), Delete("other"), Get("other")])
def test_single_key_contract_includes_pending_operations(pending_operation):
    history = OperationHistory()
    history.invoke("read", "A", Get("k"))
    history.respond("read", None)
    history.invoke("pending", "B", pending_operation)
    with pytest.raises(InvalidHistory, match="single-key"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)


def test_pending_only_multi_key_history_is_rejected_even_if_everything_could_be_omitted():
    history = OperationHistory()
    history.invoke("a", "A", Get("a"))
    history.invoke("b", "B", Get("b"))
    with pytest.raises(InvalidHistory, match="single-key"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_invalid_state_budgets_fail_clearly(budget):
    with pytest.raises(ValueError, match="max_states"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(
            OperationHistory(),
            max_states=budget,
        )


@pytest.mark.parametrize("budget", [0, 65, True, 1.5, None])
def test_invalid_operation_budgets_fail_clearly(budget):
    with pytest.raises(ValueError, match="max_operations"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(
            OperationHistory(),
            max_operations=budget,
        )


def test_state_budget_exhaustion_is_distinct_from_an_illegal_history():
    history = lost_ack_history()
    before = copy.deepcopy(vars(history))
    checker = SingleKeyKVLinearizabilityChecker()
    with pytest.raises(LinearizabilitySearchExhausted, match="max_states=1"):
        checker.check_with_pending_extensions(history, max_states=1)
    assert vars(history) == before
    assert checker.check_with_pending_extensions(history, max_states=2).linearizable


def test_operation_budget_counts_pending_reads_and_has_a_documented_upper_bound():
    history = OperationHistory()
    for index in range(33):
        history.invoke(str(index), "A", Get("k"))
    checker = SingleKeyKVLinearizabilityChecker()
    with pytest.raises(LinearizabilitySearchExhausted, match="max_operations=32"):
        checker.check_with_pending_extensions(history)
    result = checker.check_with_pending_extensions(history, max_operations=64)
    assert result.linearizable and len(result.omitted_pending) == 33
    assert history.completed() == ()


def test_unsupported_values_and_append_are_rejected():
    history = OperationHistory()
    history.invoke("bad-put", "A", Put("k", None))
    with pytest.raises(TypeError, match="string Put values"):
        SingleKeyKVLinearizabilityChecker().check_with_pending_extensions(history)
    with pytest.raises(TypeError, match="history operation"):
        OperationHistory().invoke("append", "A", Append("k", "x"))


def test_legacy_minimization_still_uses_the_completed_only_oracle():
    history = lost_ack_history()
    checker = SingleKeyKVLinearizabilityChecker()
    assert checker.check_with_pending_extensions(history).linearizable
    assert not checker.check(history).linearizable
    minimized = NonLinearizableHistoryMinimizer().minimize(history)
    assert minimized.operation_ids == ("read",)
    assert minimized.history.pending() == ()


def brute_force_extension_exists(history):
    """Small permutation reference, independent of the memoized mask search."""
    mandatory = tuple(c.invocation for c in history.completed())
    optional = tuple(i for i in history.pending() if isinstance(i.operation, (Put, Delete)))
    responses = {c.operation_id: c.completion for c in history.completed()}
    for count in range(len(optional) + 1):
        for subset in itertools.combinations(optional, count):
            for order in itertools.permutations((*mandatory, *subset)):
                positions = {i.operation_id: n for n, i in enumerate(order)}
                if any(
                    response.sequence < later.sequence
                    and positions[earlier] >= positions[later.operation_id]
                    for earlier, response in responses.items()
                    for later in order
                ):
                    continue
                value = None
                for invocation in order:
                    operation = invocation.operation
                    if isinstance(operation, Put):
                        value = operation.value
                    elif isinstance(operation, Delete):
                        value = None
                    elif responses[invocation.operation_id].result != value:
                        break
                else:
                    return True
    return False


@pytest.mark.parametrize("values", tuple(itertools.product((None, "a", "b", "forged"), repeat=2)))
@pytest.mark.parametrize("late_b", [False, True])
def test_bounded_histories_match_independent_permutation_spec(values, late_b):
    history = OperationHistory()
    history.invoke("base", "A", Put("k", "base"))
    history.respond("base")
    history.invoke("a", "B", Put("k", "a"))
    if not late_b:
        history.invoke("b", "C", Put("k", "b"))
    history.invoke("read-1", "D", Get("k"))
    history.respond("read-1", values[0])
    if late_b:
        history.invoke("b", "C", Put("k", "b"))
    history.invoke("read-2", "D", Get("k"))
    history.respond("read-2", values[1])
    checker = SingleKeyKVLinearizabilityChecker()
    result = checker.check_with_pending_extensions(history)
    assert result.linearizable == brute_force_extension_exists(history)
    if result.linearizable:
        replay_witness(history, result)


def test_completed_only_witness_keeps_the_legacy_order():
    history = OperationHistory()
    history.invoke("a", "A", Put("k", "a"))
    history.invoke("b", "B", Put("k", "b"))
    history.respond("a")
    history.respond("b")
    history.invoke("read", "C", Get("k"))
    history.respond("read", "b")
    checker = SingleKeyKVLinearizabilityChecker()
    legacy = checker.check(history)
    result = checker.check_with_pending_extensions(history)
    assert result.linearizable == legacy.linearizable
    assert result.order == legacy.order == ("a", "b", "read")
    assert result.included_pending == result.omitted_pending == ()
