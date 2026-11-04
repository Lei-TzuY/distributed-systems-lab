"""Compositional pending-aware linearizability for independent KV keys.

Each Put/Delete/Get touches one object (one key). By locality of
linearizability, the whole history is linearizable iff every key's projection
is linearizable. Expose per-key witnesses, *not* a fabricated global order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kv import Delete, Put
from .linearizability import (
    CompletionExtensionResult,
    Get,
    InvalidHistory,
    LinearizabilitySearchExhausted,
    OperationHistory,
    SingleKeyKVLinearizabilityChecker,
)


@dataclass(frozen=True, slots=True)
class MultiKeyCompletionExtensionResult:
    """An existential verdict with deterministic, independent per-key witnesses."""

    linearizable: bool
    by_key: tuple[tuple[str, CompletionExtensionResult], ...]


class MultiKeyKVLinearizabilityChecker:
    """Composition of exact single-key completion-extension oracles.

    Only individual-key Put/Delete/Get are supported. max_operations and
    max_keys bound the complete input; max_states_per_key bounds each oracle,
    so total state expansions are bounded by max_keys * max_states_per_key.
    Resource exhaustion is an exception, never a False safety verdict.
    """

    def check_with_pending_extensions(
        self,
        history: OperationHistory,
        *,
        max_operations: int = 32,
        max_keys: int = 16,
        max_states_per_key: int = 100_000,
    ) -> MultiKeyCompletionExtensionResult:
        if not isinstance(history, OperationHistory):
            raise TypeError("expected OperationHistory")
        if type(max_operations) is not int or not 1 <= max_operations <= 64:
            raise ValueError("max_operations must be an integer in 1..64")
        if type(max_keys) is not int or not 1 <= max_keys <= 64:
            raise ValueError("max_keys must be an integer in 1..64")
        if type(max_states_per_key) is not int or max_states_per_key <= 0:
            raise ValueError("max_states_per_key must be a positive integer")

        invocations = history.invocations()
        if len(invocations) > max_operations:
            raise LinearizabilitySearchExhausted(
                f"history has {len(invocations)} invocations; max_operations={max_operations}"
            )
        by_identity = {inv.operation_id: inv for inv in invocations}
        projected: dict[str, OperationHistory] = {}
        for invocation in invocations:
            operation = invocation.operation
            if not isinstance(operation, (Put, Delete, Get)):
                raise TypeError("multi-key extensions support only Put/Delete/Get")
            if not isinstance(operation.key, str) or not operation.key:
                raise InvalidHistory("multi-key operations require non-empty string keys")
            if isinstance(operation, Put) and not isinstance(operation.value, str):
                raise TypeError("multi-key extensions require string Put values")
            projected.setdefault(operation.key, OperationHistory())

        if len(projected) > max_keys:
            raise LinearizabilitySearchExhausted(
                f"history touches {len(projected)} keys; max_keys={max_keys}"
            )

        # Only filter out unrelated keys; replay all remaining *observed*
        # invocation/response events in original sequence order. This preserves
        # each key's real-time precedence without constructing fake responses.
        events: list[tuple[int, bool, object]] = [
            (inv.sequence, True, inv) for inv in invocations
        ]
        events.extend(
            (item.completion.sequence, False, item)
            for item in history.completed()
        )
        for _, is_invocation, item in sorted(events, key=lambda event: event[0]):
            if is_invocation:
                projected[item.operation.key].invoke(
                    item.operation_id, item.client_id, item.operation
                )
            else:
                key = by_identity[item.operation_id].operation.key
                projected[key].respond(
                    item.operation_id, item.completion.result
                )

        checker = SingleKeyKVLinearizabilityChecker()
        results = tuple(
            (
                key,
                checker.check_with_pending_extensions(
                    projected[key],
                    max_operations=max_operations,
                    max_states=max_states_per_key,
                ),
            )
            for key in sorted(projected)
        )
        return MultiKeyCompletionExtensionResult(
            linearizable=all(result.linearizable for _, result in results),
            by_key=results,
        )
