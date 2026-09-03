from __future__ import annotations

from dataclasses import dataclass

from .linearizability import (
    Completion,
    Invocation,
    OperationHistory,
    SingleKeyKVLinearizabilityChecker,
)


@dataclass(frozen=True, slots=True)
class HistoryMinimizationResult:
    """A deterministic 1-minimal non-linearizable history."""

    history: OperationHistory
    operation_ids: tuple[str, ...]
    removed_operation_ids: tuple[str, ...]


class NonLinearizableHistoryMinimizer:
    """Greedily reduce a failing history while preserving non-linearizability.

    The minimizer only keeps completed operations because pending invocations may
    be omitted by the linearizability completion rule and therefore cannot be
    required to witness a failure for the current checker. Completed operations
    are considered for deletion in deterministic invocation order. The pass is
    repeated after every successful deletion, yielding a 1-minimal witness: no
    remaining single operation can be removed while preserving the failure.
    """

    def __init__(self, checker: SingleKeyKVLinearizabilityChecker | None = None) -> None:
        self.checker = checker if checker is not None else SingleKeyKVLinearizabilityChecker()

    def minimize(self, history: OperationHistory) -> HistoryMinimizationResult:
        if self.checker.check(history).linearizable:
            raise ValueError("history must be non-linearizable before minimization")

        original_ids = tuple(item.operation_id for item in history.invocations())
        current = [item.operation_id for item in history.completed()]
        if not current:
            raise AssertionError("non-linearizable history must contain completed operations")

        changed = True
        while changed:
            changed = False
            for operation_id in tuple(current):
                candidate_ids = [item for item in current if item != operation_id]
                candidate = self._project(history, frozenset(candidate_ids))
                if not self.checker.check(candidate).linearizable:
                    current = candidate_ids
                    changed = True
                    break

        kept = tuple(current)
        minimized = self._project(history, frozenset(kept))
        removed = tuple(operation_id for operation_id in original_ids if operation_id not in kept)
        return HistoryMinimizationResult(minimized, kept, removed)

    @staticmethod
    def _project(history: OperationHistory, keep: frozenset[str]) -> OperationHistory:
        projected = OperationHistory()
        invocations = {item.operation_id: item for item in history.invocations()}
        completions = {item.operation_id: item.completion for item in history.completed()}
        events: list[Invocation | Completion] = [
            item for item in invocations.values() if item.operation_id in keep
        ]
        events.extend(item for operation_id, item in completions.items() if operation_id in keep)
        events.sort(key=lambda item: item.sequence)

        for event in events:
            if isinstance(event, Invocation):
                projected.invoke(event.operation_id, event.client_id, event.operation)
            else:
                projected.respond(event.operation_id, event.result)
        return projected
