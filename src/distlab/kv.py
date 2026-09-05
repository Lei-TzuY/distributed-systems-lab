from __future__ import annotations

from dataclasses import dataclass

from .commit_recovery import CommitRecoveryBarrier
from .raft import LogEntry, RaftCluster
from .state_machine import AppliedEntry, StateMachineApplier


@dataclass(frozen=True, slots=True)
class Put:
    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("KV key must be non-empty")


@dataclass(frozen=True, slots=True)
class Delete:
    key: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("KV key must be non-empty")


KVOperation = Put | Delete


@dataclass(frozen=True, slots=True)
class ClientRequest:
    """Stable client operation identity used for deterministic retry deduplication."""

    client_id: str
    request_id: int
    operation: KVOperation

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if not isinstance(self.operation, (Put, Delete)):
            raise TypeError("client request operation must be Put or Delete")


KVCommand = Put | Delete | ClientRequest | CommitRecoveryBarrier


class InvalidKVCommand(TypeError):
    """Raised when a committed log entry is not a supported KV command."""


class ClientRequestConflict(AssertionError):
    """Raised when one client request identity is reused for a different operation."""


class ReplicatedKV:
    """Deterministic key/value state derived from durable Raft application state.

    Before snapshot compaction the durable applied history is replayed from index
    one. After compaction, the durable KV snapshot seeds user-visible state and
    client deduplication identities, and only the retained applied suffix after
    that snapshot boundary is replayed.
    """

    def __init__(self, cluster: RaftCluster, applier: StateMachineApplier | None = None) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.applier = applier if applier is not None else StateMachineApplier(cluster)
        if self.applier.cluster is not cluster:
            raise ValueError("state-machine applier must belong to the same Raft cluster")

        self._state: dict[str, dict[str, str]] = {node_id: {} for node_id in cluster.node_ids}
        self._requests: dict[str, dict[tuple[str, int], KVOperation]] = {
            node_id: {} for node_id in cluster.node_ids
        }
        for node_id in cluster.node_ids:
            self._restore_snapshot_boundary(node_id)
            history = self.applier.applied_entries(node_id)
            self._validate_entries(history)
            self._validate_request_conflicts(node_id, history)
            for entry in history:
                self._apply_command(node_id, entry.command, emit_trace=False)

    def get(self, node_id: str, key: str) -> str | None:
        self._require_node(node_id)
        return self._state[node_id].get(key)

    def snapshot(self, node_id: str) -> dict[str, str]:
        self._require_node(node_id)
        return dict(self._state[node_id])

    def client_requests(self, node_id: str) -> dict[tuple[str, int], KVOperation]:
        """Return the full deduplication state represented by the current KV image."""
        self._require_node(node_id)
        return dict(self._requests[node_id])

    def has_applied_request(self, node_id: str, client_id: str, request_id: int) -> bool:
        self._require_node(node_id)
        return (client_id, request_id) in self._requests[node_id]

    def apply_committed(self, node_id: str) -> tuple[AppliedEntry, ...]:
        """Apply newly committed KV commands in Raft log order exactly once."""
        self._require_node(node_id)
        node = self.cluster.node(node_id)
        start = self.applier.last_applied(node_id) + 1
        if node.commit_index >= start:
            count = node.commit_index - start + 1
            pending = node.log_view.suffix_from(start)[:count]
        else:
            pending = ()
        self._validate_entries(pending)
        self._validate_request_conflicts(node_id, pending)

        applied = self.applier.apply_committed(node_id)
        for record in applied:
            self._apply_command(node_id, record.entry.command, emit_trace=True, index=record.index)
        return applied

    def assert_replica_consistency(self) -> None:
        """Equal durable snapshot/suffix prefixes must yield identical KV/dedup state."""
        by_prefix: dict[
            tuple[int, int, tuple[LogEntry, ...]],
            tuple[str, dict[str, str], dict[tuple[str, int], KVOperation]],
        ] = {}
        for node_id in self.cluster.node_ids:
            prefix = (
                self.applier.applied_base_index(node_id),
                self.applier.applied_base_term(node_id),
                self.applier.applied_entries(node_id),
            )
            current = self.snapshot(node_id)
            requests = dict(self._requests[node_id])
            existing = by_prefix.get(prefix)
            if existing is not None and (existing[1] != current or existing[2] != requests):
                other_id, other_state, other_requests = existing
                raise AssertionError(
                    "deterministic KV state diverged for identical applied prefix: "
                    f"{other_id!r}=({other_state!r}, {other_requests!r}), "
                    f"{node_id!r}=({current!r}, {requests!r})"
                )
            by_prefix.setdefault(prefix, (node_id, current, requests))

    def _restore_snapshot_boundary(self, node_id: str) -> None:
        base_index = self.applier.applied_base_index(node_id)
        if base_index == 0:
            return

        from .snapshot import KVSnapshot

        value = self.sim.persistent_state[node_id].get("kv_snapshot")
        if not isinstance(value, KVSnapshot):
            raise AssertionError(
                f"compacted state-machine history for {node_id!r} requires a durable KV snapshot"
            )
        if value.last_included_index != base_index:
            raise AssertionError(
                f"KV snapshot index for {node_id!r} diverges from state-machine boundary"
            )
        if value.last_included_term != self.applier.applied_base_term(node_id):
            raise AssertionError(
                f"KV snapshot term for {node_id!r} diverges from state-machine boundary"
            )

        self._state[node_id] = dict(value.state)
        for item in value.client_requests:
            identity = (item.client_id, item.request_id)
            previous = self._requests[node_id].get(identity)
            if previous is not None and previous != item.operation:
                raise ClientRequestConflict(
                    "snapshot contains conflicting client request identity: "
                    f"client={item.client_id!r}, request_id={item.request_id}"
                )
            self._requests[node_id][identity] = item.operation

    def _apply_command(
        self,
        node_id: str,
        command: object,
        *,
        emit_trace: bool,
        index: int | None = None,
    ) -> None:
        client_id: str | None = None
        request_id: int | None = None
        duplicate = False
        operation: object = command
        if isinstance(command, ClientRequest):
            client_id = command.client_id
            request_id = command.request_id
            identity = (client_id, request_id)
            previous = self._requests[node_id].get(identity)
            if previous is not None:
                if previous != command.operation:
                    raise ClientRequestConflict(
                        "client request identity reused with different operation: "
                        f"client={client_id!r}, request_id={request_id}"
                    )
                duplicate = True
            else:
                self._requests[node_id][identity] = command.operation
            operation = command.operation

        if not duplicate:
            self._execute_operation(node_id, operation)

        if emit_trace:
            if isinstance(operation, Put):
                operation_name = "put"
                key: str | None = operation.key
                value: str | None = operation.value
            elif isinstance(operation, Delete):
                operation_name = "delete"
                key = operation.key
                value = None
            elif isinstance(operation, CommitRecoveryBarrier):
                operation_name = "commit-recovery-barrier"
                key = None
                value = None
            else:
                raise InvalidKVCommand(f"unsupported KV command {operation!r}")
            self.sim._record(
                "kv-apply",
                node=node_id,
                index=index,
                operation=operation_name,
                key=key,
                value=value,
                client_id=client_id,
                request_id=request_id,
                duplicate=duplicate,
            )

    def _execute_operation(self, node_id: str, operation: object) -> None:
        state = self._state[node_id]
        if isinstance(operation, Put):
            state[operation.key] = operation.value
        elif isinstance(operation, Delete):
            state.pop(operation.key, None)
        elif isinstance(operation, CommitRecoveryBarrier):
            return
        else:
            raise InvalidKVCommand(f"unsupported KV command {operation!r}")

    @staticmethod
    def _validate_entries(entries: tuple[LogEntry, ...]) -> None:
        for entry in entries:
            command = entry.command
            if isinstance(command, ClientRequest):
                if not isinstance(command.operation, (Put, Delete)):
                    raise InvalidKVCommand(f"unsupported KV command {command.operation!r}")
            elif not isinstance(command, (Put, Delete, CommitRecoveryBarrier)):
                raise InvalidKVCommand(f"unsupported KV command {command!r}")

    def _validate_request_conflicts(self, node_id: str, entries: tuple[LogEntry, ...]) -> None:
        seen = dict(self._requests[node_id])
        for entry in entries:
            command = entry.command
            if not isinstance(command, ClientRequest):
                continue
            identity = (command.client_id, command.request_id)
            previous = seen.get(identity)
            if previous is not None and previous != command.operation:
                raise ClientRequestConflict(
                    "client request identity reused with different operation: "
                    f"client={command.client_id!r}, request_id={command.request_id}"
                )
            seen.setdefault(identity, command.operation)

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._state:
            raise ValueError(f"unknown Raft node {node_id!r}")
