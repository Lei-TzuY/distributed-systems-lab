from __future__ import annotations

from typing import TYPE_CHECKING

from .kv import ClientRequest, ClientRequestConflict, Delete, Put, ReplicatedKV
from .linearizability import Get, InvalidHistory, OperationHistory

if TYPE_CHECKING:
    from .linearizable_read import LinearizableKVReader


class KVClientHistory:
    """Capture client-visible KV executions into a linearizability history.

    A write invocation remains pending until a replica has durably applied the
    corresponding ``ClientRequest``. Retries re-submit the exact same pending
    request without creating a second logical history operation. The request-id
    identity is retained by the shared ``OperationHistory`` while unresolved so
    rebuilding this wrapper preserves pending writes and exact-retry semantics.
    Reads may be sampled directly from one replica or routed through
    ``LinearizableKVReader`` so the recorded client response is backed by the real
    Raft read barrier. Explicitly abandoned linearizable reads remain incomplete
    evidence in the shared history, but are terminal from the client API's
    perspective and are not resurrected when either the client session or history
    wrapper is rebuilt.
    """

    def __init__(self, kv: ReplicatedKV, history: OperationHistory | None = None) -> None:
        self.kv = kv
        self.sim = kv.sim
        self.history = history if history is not None else OperationHistory()
        self._pending_writes: dict[str, ClientRequest] = {}
        for invocation in self.history.pending():
            request_id = self.history.client_request_id(invocation.operation_id)
            if request_id is None:
                continue
            if not isinstance(invocation.operation, (Put, Delete)):
                raise ValueError(
                    "client request identity is attached to a non-write history invocation"
                )
            self._pending_writes[invocation.operation_id] = ClientRequest(
                invocation.client_id,
                request_id,
                invocation.operation,
            )

    def invoke_write(
        self,
        operation_id: str,
        client_id: str,
        request_id: int,
        operation: Put | Delete,
    ) -> ClientRequest:
        if not isinstance(operation, (Put, Delete)):
            raise TypeError("write operation must be Put or Delete")
        request = ClientRequest(client_id, request_id, operation)
        self._ensure_client_request_identity_available(client_id, request_id)
        self.history.invoke(operation_id, client_id, operation)
        self.history.attach_client_request_id(operation_id, request_id)
        self._pending_writes[operation_id] = request
        self.sim._record(
            "client-invoke",
            operation_id=operation_id,
            client_id=client_id,
            request_id=request_id,
            operation=self._operation_name(operation),
            key=operation.key,
        )
        return request

    def retry_write(self, operation_id: str) -> ClientRequest:
        """Return the exact pending request for a transport/protocol retry.

        A retry is not a new client-visible operation, so it deliberately does
        not add another invocation to ``OperationHistory``. Requiring the
        original operation to remain pending also prevents a completed logical
        write from being accidentally re-opened with the same operation id.
        """

        request = self._pending_writes.get(operation_id)
        if request is None:
            raise ValueError(f"unknown pending write {operation_id!r}")

        operation = request.operation
        self.sim._record(
            "client-retry",
            operation_id=operation_id,
            client_id=request.client_id,
            request_id=request.request_id,
            operation=self._operation_name(operation),
            key=operation.key,
        )
        return request

    def complete_write(self, operation_id: str, node_id: str) -> None:
        request = self._pending_writes.get(operation_id)
        if request is None:
            raise ValueError(f"unknown pending write {operation_id!r}")
        applied = self.kv.client_requests(node_id).get((request.client_id, request.request_id))
        if applied is None:
            raise RuntimeError(
                "cannot complete a client write before the target replica applied its request"
            )
        if applied != request.operation:
            raise ClientRequestConflict(
                "cannot complete a client write from conflicting applied request identity: "
                f"client={request.client_id!r}, request_id={request.request_id}"
            )

        self.history.respond(operation_id)
        self.history.retire_client_request_id(operation_id)
        del self._pending_writes[operation_id]
        self.sim._record(
            "client-response",
            operation_id=operation_id,
            client_id=request.client_id,
            request_id=request.request_id,
            node=node_id,
            result=None,
        )

    def read(self, operation_id: str, client_id: str, node_id: str, key: str) -> str | None:
        operation = Get(key)
        self.history.invoke(operation_id, client_id, operation)
        self.sim._record(
            "client-invoke",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            node=node_id,
        )
        result = self.kv.get(node_id, key)
        self.history.respond(operation_id, result)
        self.sim._record(
            "client-response",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            node=node_id,
            result=result,
        )
        return result

    def linearizable_read(
        self,
        operation_id: str,
        client_id: str,
        reader: LinearizableKVReader,
        key: str,
        *,
        max_attempts_per_peer: int = 1,
    ) -> str | None:
        """Record a client-visible read backed by the Raft linearizable-read path.

        The invocation is recorded before the quorum barrier so failed reads remain
        incomplete operations in the history rather than fabricated responses.
        A reader tied to another replicated state machine is rejected before history
        mutation to keep the evidence layer and protocol execution on one cluster.
        """

        if reader.kv is not self.kv:
            raise ValueError("linearizable reader must use the same replicated KV state")

        operation = Get(key)
        self.history.invoke(operation_id, client_id, operation)
        self.sim._record(
            "client-invoke",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            node=reader.leader.node_id,
            consistency="linearizable",
        )
        return self._complete_linearizable_read(
            operation_id,
            client_id,
            reader,
            key,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def retry_linearizable_read(
        self,
        operation_id: str,
        client_id: str,
        reader: LinearizableKVReader,
        key: str,
        *,
        max_attempts_per_peer: int = 1,
    ) -> str | None:
        """Retry an incomplete linearizable read without creating a new invocation."""

        if reader.kv is not self.kv:
            raise ValueError("linearizable reader must use the same replicated KV state")
        if self.history.is_abandoned(operation_id):
            raise ValueError(f"abandoned linearizable read {operation_id!r} cannot be retried")
        pending = {item.operation_id: item for item in self.history.pending()}.get(operation_id)
        if pending is None:
            raise ValueError(f"unknown pending linearizable read {operation_id!r}")
        if pending.client_id != client_id or pending.operation != Get(key):
            raise ValueError("linearizable read retry must match the pending invocation")

        self.sim._record(
            "client-retry",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            node=reader.leader.node_id,
            consistency="linearizable",
        )
        return self._complete_linearizable_read(
            operation_id,
            client_id,
            reader,
            key,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def abandon_linearizable_read(self, operation_id: str, client_id: str, key: str) -> None:
        """Make a failed read terminal while retaining its incomplete history evidence."""

        if self.history.is_abandoned(operation_id):
            raise ValueError(f"linearizable read {operation_id!r} was already abandoned")
        pending = {item.operation_id: item for item in self.history.pending()}.get(operation_id)
        if pending is None:
            raise ValueError(f"unknown pending linearizable read {operation_id!r}")
        if pending.client_id != client_id or pending.operation != Get(key):
            raise ValueError("linearizable read abandonment must match the pending invocation")
        self.history.abandon(operation_id)
        self.sim._record(
            "client-abandon",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            consistency="linearizable",
        )

    def is_abandoned_read(self, operation_id: str) -> bool:
        return self.history.is_abandoned(operation_id)

    def _complete_linearizable_read(
        self,
        operation_id: str,
        client_id: str,
        reader: LinearizableKVReader,
        key: str,
        *,
        max_attempts_per_peer: int,
    ) -> str | None:
        result = reader.get(key, max_attempts_per_peer=max_attempts_per_peer)
        self.history.respond(operation_id, result)
        self.sim._record(
            "client-response",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            node=reader.leader.node_id,
            result=result,
            consistency="linearizable",
        )
        return result

    def pending_write(self, operation_id: str) -> ClientRequest | None:
        return self._pending_writes.get(operation_id)

    def _ensure_client_request_identity_available(self, client_id: str, request_id: int) -> None:
        for invocation in self.history.invocations():
            if invocation.client_id != client_id:
                continue
            if self.history.client_request_id(invocation.operation_id) != request_id:
                continue
            raise InvalidHistory(
                "client request identity is already attached to active operation "
                f"{invocation.operation_id!r}: client={client_id!r}, request_id={request_id}"
            )

    @staticmethod
    def _operation_name(operation: Put | Delete) -> str:
        return "put" if isinstance(operation, Put) else "delete"
