from __future__ import annotations

from dataclasses import dataclass

from .kv import Delete, Put


@dataclass(frozen=True, slots=True)
class Get:
    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("KV key must be non-empty")


HistoryOperation = Put | Delete | Get
HistoryResult = str | None


@dataclass(frozen=True, slots=True)
class Invocation:
    operation_id: str
    client_id: str
    operation: HistoryOperation
    sequence: int


@dataclass(frozen=True, slots=True)
class Completion:
    operation_id: str
    result: HistoryResult
    sequence: int


@dataclass(frozen=True, slots=True)
class CompletedOperation:
    invocation: Invocation
    completion: Completion

    @property
    def operation_id(self) -> str:
        return self.invocation.operation_id


@dataclass(frozen=True, slots=True)
class LinearizabilityResult:
    linearizable: bool
    order: tuple[str, ...]


class InvalidHistory(ValueError):
    """Raised when an operation history is structurally invalid."""


class OperationHistory:
    """Deterministic invocation/response history for bounded correctness checks.

    Sequence numbers are assigned locally in insertion order. Pending invocations
    are retained in the history but may be omitted from a legal linearization,
    matching the standard completion rule for incomplete histories. An invocation
    may also be marked abandoned when the client deliberately stops awaiting it;
    abandonment is terminal client metadata, not a fabricated response, so the
    invocation remains pending and omittable for linearizability checking.

    Pending client write request IDs are first-class recovery metadata. They may
    only be attached to unresolved Put/Delete invocations, each active
    ``(client_id, request_id)`` identity belongs to exactly one logical operation,
    and the identity is retired only after that invocation has a recorded response.
    This keeps exact-retry reconstruction unambiguous and structurally coupled to
    the history lifecycle instead of relying on wrapper-local attributes.
    """

    def __init__(self) -> None:
        self._sequence = 0
        self._invocations: dict[str, Invocation] = {}
        self._completions: dict[str, Completion] = {}
        self._abandoned: set[str] = set()
        self._client_request_ids: dict[str, int] = {}
        self._active_client_requests: dict[tuple[str, int], str] = {}

    def invoke(
        self,
        operation_id: str,
        client_id: str,
        operation: HistoryOperation,
    ) -> Invocation:
        if not operation_id:
            raise ValueError("operation_id must be non-empty")
        if not client_id:
            raise ValueError("client_id must be non-empty")
        if operation_id in self._invocations:
            raise InvalidHistory(f"duplicate invocation for operation {operation_id!r}")
        if not isinstance(operation, (Put, Delete, Get)):
            raise TypeError("history operation must be Put, Delete, or Get")

        invocation = Invocation(operation_id, client_id, operation, self._next_sequence())
        self._invocations[operation_id] = invocation
        return invocation

    def respond(self, operation_id: str, result: HistoryResult = None) -> Completion:
        invocation = self._invocations.get(operation_id)
        if invocation is None:
            raise InvalidHistory(f"response without invocation for operation {operation_id!r}")
        if operation_id in self._completions:
            raise InvalidHistory(f"duplicate response for operation {operation_id!r}")
        if operation_id in self._abandoned:
            raise InvalidHistory(f"response for abandoned operation {operation_id!r}")
        if isinstance(invocation.operation, (Put, Delete)) and result is not None:
            raise InvalidHistory("Put/Delete responses must be None")
        if (
            isinstance(invocation.operation, Get)
            and result is not None
            and not isinstance(result, str)
        ):
            raise InvalidHistory("Get response must be str or None")

        completion = Completion(operation_id, result, self._next_sequence())
        self._completions[operation_id] = completion
        return completion

    def abandon(self, operation_id: str) -> Invocation:
        """Mark an incomplete invocation as terminal from the client's perspective."""

        invocation = self._invocations.get(operation_id)
        if invocation is None:
            raise InvalidHistory(f"abandon without invocation for operation {operation_id!r}")
        if operation_id in self._completions:
            raise InvalidHistory(f"cannot abandon completed operation {operation_id!r}")
        if operation_id in self._abandoned:
            raise InvalidHistory(f"duplicate abandonment for operation {operation_id!r}")
        self._abandoned.add(operation_id)
        return invocation

    def attach_client_request_id(self, operation_id: str, request_id: int) -> None:
        """Attach unique exact-retry identity to an unresolved client write invocation."""

        invocation = self._invocations.get(operation_id)
        if invocation is None:
            raise InvalidHistory(
                f"client request identity without invocation for operation {operation_id!r}"
            )
        if not isinstance(invocation.operation, (Put, Delete)):
            raise InvalidHistory("client request identity may only be attached to Put/Delete")
        if operation_id in self._completions:
            raise InvalidHistory(
                f"client request identity cannot be attached after response for {operation_id!r}"
            )
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        if operation_id in self._client_request_ids:
            raise InvalidHistory(
                f"duplicate client request identity for operation {operation_id!r}"
            )
        identity = (invocation.client_id, request_id)
        existing_operation_id = self._active_client_requests.get(identity)
        if existing_operation_id is not None:
            raise InvalidHistory(
                "client request identity is already attached to active operation "
                f"{existing_operation_id!r}: client={invocation.client_id!r}, "
                f"request_id={request_id}"
            )
        self._client_request_ids[operation_id] = request_id
        self._active_client_requests[identity] = operation_id

    def client_request_id(self, operation_id: str) -> int | None:
        """Return the recovery request ID attached to an invocation, if any."""

        return self._client_request_ids.get(operation_id)

    def retire_client_request_id(self, operation_id: str) -> int:
        """Retire exact-retry metadata after the write receives a response."""

        if operation_id not in self._completions:
            raise InvalidHistory(
                f"client request identity cannot retire before response for {operation_id!r}"
            )
        invocation = self._invocations[operation_id]
        try:
            request_id = self._client_request_ids.pop(operation_id)
        except KeyError as exc:
            raise InvalidHistory(
                f"missing client request identity for completed operation {operation_id!r}"
            ) from exc
        identity = (invocation.client_id, request_id)
        owner = self._active_client_requests.get(identity)
        if owner != operation_id:
            self._client_request_ids[operation_id] = request_id
            raise InvalidHistory(
                f"active client request identity owner mismatch for operation {operation_id!r}"
            )
        del self._active_client_requests[identity]
        return request_id

    def is_abandoned(self, operation_id: str) -> bool:
        return operation_id in self._abandoned

    def abandoned(self) -> tuple[Invocation, ...]:
        return tuple(
            invocation
            for invocation in self.invocations()
            if invocation.operation_id in self._abandoned
        )

    def invocations(self) -> tuple[Invocation, ...]:
        return tuple(sorted(self._invocations.values(), key=lambda item: item.sequence))

    def completed(self) -> tuple[CompletedOperation, ...]:
        operations = [
            CompletedOperation(invocation, self._completions[operation_id])
            for operation_id, invocation in self._invocations.items()
            if operation_id in self._completions
        ]
        return tuple(sorted(operations, key=lambda item: item.invocation.sequence))

    def pending(self) -> tuple[Invocation, ...]:
        return tuple(
            invocation
            for invocation in self.invocations()
            if invocation.operation_id not in self._completions
        )

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence


class SingleKeyKVLinearizabilityChecker:
    """Exhaustive deterministic linearizability checker for one KV key.

    The checker respects real-time precedence: if operation A completes before
    operation B is invoked, A must precede B. Overlapping operations may be
    explored in either order. The bounded DFS memoizes equivalent search states
    so failing histories remain deterministic and reproducible.
    """

    def check(self, history: OperationHistory) -> LinearizabilityResult:
        operations = history.completed()
        if not operations:
            return LinearizabilityResult(True, ())

        keys = {self._key(item.invocation.operation) for item in operations}
        if len(keys) != 1:
            raise InvalidHistory(
                "single-key checker requires all completed operations to use one key"
            )

        predecessors = self._predecessors(operations)
        full_mask = (1 << len(operations)) - 1
        failed: set[tuple[int, str | None]] = set()

        def search(mask: int, value: str | None) -> tuple[str, ...] | None:
            if mask == full_mask:
                return ()
            memo_key = (mask, value)
            if memo_key in failed:
                return None

            for index, item in enumerate(operations):
                bit = 1 << index
                if mask & bit:
                    continue
                if predecessors[index] & ~mask:
                    continue
                next_value = self._apply(item, value)
                if next_value is _INVALID:
                    continue
                suffix = search(mask | bit, next_value)
                if suffix is not None:
                    return (item.operation_id, *suffix)

            failed.add(memo_key)
            return None

        order = search(0, None)
        if order is None:
            return LinearizabilityResult(False, ())
        return LinearizabilityResult(True, order)

    @staticmethod
    def _predecessors(operations: tuple[CompletedOperation, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for current in operations:
            mask = 0
            for index, other in enumerate(operations):
                if other.completion.sequence < current.invocation.sequence:
                    mask |= 1 << index
            result.append(mask)
        return tuple(result)

    @staticmethod
    def _key(operation: HistoryOperation) -> str:
        return operation.key

    @staticmethod
    def _apply(item: CompletedOperation, value: str | None) -> str | object | None:
        operation = item.invocation.operation
        result = item.completion.result
        if isinstance(operation, Put):
            return operation.value if result is None else _INVALID
        if isinstance(operation, Delete):
            return None if result is None else _INVALID
        if isinstance(operation, Get):
            return value if result == value else _INVALID
        raise TypeError(f"unsupported history operation {operation!r}")


_INVALID = object()
