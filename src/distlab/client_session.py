from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .client_history import KVClientHistory
from .kv import ClientRequest, Delete, Put
from .linearizability import Get

if TYPE_CHECKING:
    from .linearizable_read import LinearizableKVReader


class ClientSessionError(RuntimeError):
    """Base error for deterministic client-session ordering violations."""


class SessionWritePending(ClientSessionError):
    """Raised when another session operation would overtake a pending write."""


class SessionReadPending(ClientSessionError):
    """Raised when another session operation would overtake a pending read."""


class StaleClientRequest(ClientSessionError):
    """Raised when a completed request id is reused as a new logical write."""


@dataclass(frozen=True, slots=True)
class PendingSessionWrite:
    operation_id: str
    request: ClientRequest


@dataclass(frozen=True, slots=True)
class PendingSessionRead:
    operation_id: str
    key: str


class KVClientSession:
    """Single-flight monotonic request sequencing for one logical KV client.

    Exact retries reuse the active ``ClientRequest`` through ``retry_write``.
    A new logical write must use a request id strictly greater than the most
    recently completed request. Linearizable reads are routed through the real
    Raft read barrier and cannot overtake a pending write. Failed reads remain
    the active session operation and may be retried exactly without creating a
    second history invocation, preserving client program order across failures.
    Sessions may be reconstructed from a replica's applied deduplication state,
    including state restored from a durable KV snapshot; ``recover_linearizable``
    should be used when recovery must not trust a potentially stale replica.
    Recovery also restores an unresolved client invocation from the shared
    history so reconstructing the session cannot bypass its single-flight fence.
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
        self._pending_read: PendingSessionRead | None = None

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
        session._restore_pending_operation()
        clients.sim._record(
            "client-session-recover",
            client_id=client_id,
            node=node_id,
            last_completed_request_id=last_completed,
            pending_operation=session._pending_operation_id(),
        )
        return session

    @classmethod
    def recover_linearizable(
        cls,
        clients: KVClientHistory,
        client_id: str,
        reader: LinearizableKVReader,
        *,
        max_attempts_per_peer: int = 1,
    ) -> KVClientSession:
        """Recover the sequence floor from quorum-confirmed leader state.

        This prevents a lagging follower from moving a reconstructed client's
        monotonic request-id floor backwards. The existing Raft linearizable
        read barrier first confirms current-leader authority and applies all
        committed entries to the leader state machine; only then is the durable
        client deduplication state inspected.
        """

        if reader.kv is not clients.kv:
            raise ValueError("client history and reader must belong to the same KV state")
        evidence = reader.barrier(max_attempts_per_peer=max_attempts_per_peer)
        session = cls.recover(clients, client_id, evidence.leader)
        clients.sim._record(
            "client-session-recover-linearizable",
            client_id=client_id,
            node=evidence.leader,
            term=evidence.term,
            commit_index=evidence.commit_index,
            last_completed_request_id=session.last_completed_request_id,
            pending_operation=session._pending_operation_id(),
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
        if self._pending_read is not None:
            raise SessionReadPending(
                f"client {self.client_id!r} cannot write while read "
                f"{self._pending_read.operation_id!r} is pending"
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

    def linearizable_read(
        self,
        operation_id: str,
        reader: LinearizableKVReader,
        key: str,
        *,
        max_attempts_per_peer: int = 1,
    ) -> str | None:
        """Execute a client-program-ordered linearizable read.

        A pending write must first complete (or be retried to completion) so a
        later read cannot appear before it in the client-visible history. If the
        read barrier fails after recording the invocation, the same read remains
        pending and blocks later session operations until
        ``retry_linearizable_read`` completes it. Validation failures before
        history mutation do not leave phantom session-local pending state.
        """

        if self._pending is not None:
            raise SessionWritePending(
                f"client {self.client_id!r} cannot read while write "
                f"{self._pending.operation_id!r} is pending"
            )
        if self._pending_read is not None:
            raise SessionReadPending(
                f"client {self.client_id!r} already has pending read "
                f"{self._pending_read.operation_id!r}"
            )
        existing_operation_ids = {
            item.operation_id for item in self.clients.history.invocations()
        }
        self._pending_read = PendingSessionRead(operation_id, key)
        try:
            result = self.clients.linearizable_read(
                operation_id,
                self.client_id,
                reader,
                key,
                max_attempts_per_peer=max_attempts_per_peer,
            )
        except Exception:
            pending = {
                item.operation_id: item for item in self.clients.history.pending()
            }.get(operation_id)
            if (
                operation_id in existing_operation_ids
                or pending is None
                or pending.client_id != self.client_id
                or pending.operation != Get(key)
            ):
                self._pending_read = None
            raise
        else:
            self._pending_read = None
            return result

    def retry_linearizable_read(
        self,
        operation_id: str,
        reader: LinearizableKVReader,
        *,
        max_attempts_per_peer: int = 1,
    ) -> str | None:
        """Retry the exact pending read without creating a new history invocation."""

        pending = self._require_pending_read(operation_id)
        result = self.clients.retry_linearizable_read(
            operation_id,
            self.client_id,
            reader,
            pending.key,
            max_attempts_per_peer=max_attempts_per_peer,
        )
        self._pending_read = None
        return result

    def pending_write(self) -> PendingSessionWrite | None:
        return self._pending

    def pending_read(self) -> PendingSessionRead | None:
        return self._pending_read

    def _restore_pending_operation(self) -> None:
        pending = [
            item for item in self.clients.history.pending() if item.client_id == self.client_id
        ]
        if not pending:
            return
        if len(pending) != 1:
            raise ClientSessionError(
                f"client {self.client_id!r} has {len(pending)} unresolved history operations"
            )

        invocation = pending[0]
        operation = invocation.operation
        if isinstance(operation, (Put, Delete)):
            request = self.clients.pending_write(invocation.operation_id)
            if (
                request is None
                or request.client_id != self.client_id
                or request.operation != operation
            ):
                raise ClientSessionError(
                    "pending write history is inconsistent with client request state"
                )
            if request.request_id < self.last_completed_request_id:
                raise ClientSessionError(
                    "pending write request id precedes recovered durable sequence floor: "
                    f"request_id={request.request_id}, "
                    f"last_completed_request_id={self.last_completed_request_id}"
                )
            self._pending = PendingSessionWrite(invocation.operation_id, request)
            kind = "write"
        elif isinstance(operation, Get):
            self._pending_read = PendingSessionRead(invocation.operation_id, operation.key)
            kind = "read"
        else:
            raise ClientSessionError("unsupported pending client history operation")

        self.clients.sim._record(
            "client-session-recover-pending",
            client_id=self.client_id,
            operation_id=invocation.operation_id,
            operation=kind,
        )

    def _pending_operation_id(self) -> str | None:
        if self._pending is not None:
            return self._pending.operation_id
        if self._pending_read is not None:
            return self._pending_read.operation_id
        return None

    def _require_pending(self, operation_id: str) -> PendingSessionWrite:
        pending = self._pending
        if pending is None or pending.operation_id != operation_id:
            raise ValueError(f"unknown pending session write {operation_id!r}")
        return pending

    def _require_pending_read(self, operation_id: str) -> PendingSessionRead:
        pending = self._pending_read
        if pending is None or pending.operation_id != operation_id:
            raise ValueError(f"unknown pending session read {operation_id!r}")
        return pending
