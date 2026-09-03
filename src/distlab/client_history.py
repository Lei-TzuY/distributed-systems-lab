from __future__ import annotations

from .kv import ClientRequest, Delete, Put, ReplicatedKV
from .linearizability import Get, OperationHistory


class KVClientHistory:
    """Capture client-visible KV executions into a linearizability history.

    A write invocation remains pending until a replica has durably applied the
    corresponding ``ClientRequest``. Retries may re-submit the same request to
    Raft without creating a second logical history operation. Reads are sampled
    from one replica and completed immediately with the value observed there.
    """

    def __init__(self, kv: ReplicatedKV, history: OperationHistory | None = None) -> None:
        self.kv = kv
        self.sim = kv.sim
        self.history = history if history is not None else OperationHistory()
        self._pending_writes: dict[str, ClientRequest] = {}

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
        self.history.invoke(operation_id, client_id, operation)
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

    def complete_write(self, operation_id: str, node_id: str) -> None:
        request = self._pending_writes.get(operation_id)
        if request is None:
            raise ValueError(f"unknown pending write {operation_id!r}")
        if not self.kv.has_applied_request(node_id, request.client_id, request.request_id):
            raise RuntimeError(
                "cannot complete a client write before the target replica applied its request"
            )

        self.history.respond(operation_id)
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

    def pending_write(self, operation_id: str) -> ClientRequest | None:
        return self._pending_writes.get(operation_id)

    @staticmethod
    def _operation_name(operation: Put | Delete) -> str:
        return "put" if isinstance(operation, Put) else "delete"
