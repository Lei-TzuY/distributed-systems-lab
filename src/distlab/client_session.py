from __future__ import annotations

from dataclasses import dataclass

from .client_history import KVClientHistory
from .kv import ClientRequest, Delete, Put


class ClientSessionError(RuntimeError):
    """Base error for deterministic client-session ordering violations."""


class SessionWritePending(ClientSessionError):
    """Raised when a session tries to pipeline a second write."""


class StaleClientRequest(ClientSessionError):
    """Raised when a completed request id is reused as a new logical write."""


@dataclass(frozen=True, slots=True)
class PendingSessionWrite:
    operation_id: str
    request: ClientRequest


class KVClientSession:
    """Single-flight monotonic request sequencing for one logical KV client.

    Exact retries reuse the active ``ClientRequest`` through ``retry_write``.
    A new logical write must use a request id strictly greater than the most
    recently completed request. Sessions may be reconstructed from a replica's
    applied deduplication state, including state restored from a durable KV
    snapshot.
    """

    def __init__(
        self,
        clients: KVClientHistory,
        client_id: str,
        *,
        last_completed_request_id: int = -1,
    ) -> None:
        if not client_id:
            raise ValueError("client_id must be non-empty")
        if last_completed_request_id < -1:
            raise ValueError("last_completed_request_id must be at least -1")
        self.clients = clients
        self.client_id = client_id
        self.last_completed_request_id = last_completed_request_id
        self._pending: PendingSessionWrite | None = None

    @classmethod
    def recover(
        cls,
        clients: KVClientHistory,
        client_id: str,
        node_id: str,
    ) -> KVClientSession:
        requests = clients.kv.client_requests(node_id)
        completed = [request_id for (owner, request_id) in requests if owner == client_id]
        last_completed = max(completed, default=-1)
        session = cls(
            clients,
            client_id,
            last_completed_request_id=last_completed,
        )
        clients.sim._record(
            "client-session-recover",
            client_id=client_id,
            node=node_id,
            last_completed_request_id=last_completed,
        )
        return session

    def invoke_write(
        self,
        operation_id: str,
        request_id: int,
        operation: Put | Delete,
    ) -> ClientRequest:
        if self._pending is not None:
            raise SessionWritePending(
                f"client {self.client_id!r} already has pending write "
                f"{self._pending.operation_id!r}"
            )
        if request_id <= self.last_completed_request_id:
            raise StaleClientRequest(
                f"client {self.client_id!r} request_id {request_id} is not newer than "
                f"completed request_id {self.last_completed_request_id}"
            )
        request = self.clients.invoke_write(
            operation_id,
            self.client_id,
            request_id,
            operation,
        )
        self._pending = PendingSessionWrite(operation_id, request)
        return request

    def retry_write(self, operation_id: str) -> ClientRequest:
        pending = self._require_pending(operation_id)
        retried = self.clients.retry_write(operation_id)
        if retried != pending.request:
            raise AssertionError("client history retry changed the active request identity")
        return retried

    def complete_write(self, operation_id: str, node_id: str) -> None:
        pending = self._require_pending(operation_id)
        self.clients.complete_write(operation_id, node_id)
        self.last_completed_request_id = pending.request.request_id
        self._pending = None
        self.clients.sim._record(
            "client-session-advance",
            client_id=self.client_id,
            request_id=self.last_completed_request_id,
            node=node_id,
        )

    def pending_write(self) -> PendingSessionWrite | None:
        return self._pending

    def _require_pending(self, operation_id: str) -> PendingSessionWrite:
        pending = self._pending
        if pending is None or pending.operation_id != operation_id:
            raise ValueError(f"unknown pending session write {operation_id!r}")
        return pending
