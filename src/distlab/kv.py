from __future__ import annotations

from dataclasses import dataclass

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


KVCommand = Put | Delete


class InvalidKVCommand(TypeError):
    """Raised when a committed log entry is not a supported KV command."""


class ReplicatedKV:
    """Deterministic key/value state derived from each node's applied Raft prefix.

    KV state is intentionally rebuildable rather than independently persisted.
    The durable source of truth is the applied Raft history owned by
    ``StateMachineApplier``. Reconstructing this object after crash/restart
    replays that durable prefix and therefore cannot expose uncommitted data.
    """

    def __init__(self, cluster: RaftCluster, applier: StateMachineApplier | None = None) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.applier = applier if applier is not None else StateMachineApplier(cluster)
        if self.applier.cluster is not cluster:
            raise ValueError("state-machine applier must belong to the same Raft cluster")

        self._state: dict[str, dict[str, str]] = {node_id: {} for node_id in cluster.node_ids}
        for node_id in cluster.node_ids:
            history = self.applier.applied_entries(node_id)
            self._validate_entries(history)
            for entry in history:
                self._apply_command(node_id, entry.command, emit_trace=False)

    def get(self, node_id: str, key: str) -> str | None:
        self._require_node(node_id)
        return self._state[node_id].get(key)

    def snapshot(self, node_id: str) -> dict[str, str]:
        self._require_node(node_id)
        return dict(self._state[node_id])

    def apply_committed(self, node_id: str) -> tuple[AppliedEntry, ...]:
        """Apply newly committed KV commands in Raft log order exactly once."""

        self._require_node(node_id)
        node = self.cluster.node(node_id)
        start = self.applier.last_applied(node_id) + 1
        if node.commit_index >= start:
            self._validate_entries(node.log[start - 1 : node.commit_index])

        applied = self.applier.apply_committed(node_id)
        for record in applied:
            self._apply_command(node_id, record.entry.command, emit_trace=True, index=record.index)
        return applied

    def assert_replica_consistency(self) -> None:
        """Equal applied prefixes must produce identical deterministic KV state."""

        by_prefix: dict[tuple[LogEntry, ...], tuple[str, dict[str, str]]] = {}
        for node_id in self.cluster.node_ids:
            prefix = self.applier.applied_entries(node_id)
            current = self.snapshot(node_id)
            existing = by_prefix.get(prefix)
            if existing is not None and existing[1] != current:
                other_id, other_state = existing
                raise AssertionError(
                    "deterministic KV state diverged for identical applied prefix: "
                    f"{other_id!r}={other_state!r}, {node_id!r}={current!r}"
                )
            by_prefix.setdefault(prefix, (node_id, current))

    def _apply_command(
        self,
        node_id: str,
        command: object,
        *,
        emit_trace: bool,
        index: int | None = None,
    ) -> None:
        state = self._state[node_id]
        if isinstance(command, Put):
            state[command.key] = command.value
            operation = "put"
            value: str | None = command.value
        elif isinstance(command, Delete):
            state.pop(command.key, None)
            operation = "delete"
            value = None
        else:  # validation should make this unreachable
            raise InvalidKVCommand(f"unsupported KV command {command!r}")

        if emit_trace:
            self.sim._record(
                "kv-apply",
                node=node_id,
                index=index,
                operation=operation,
                key=command.key,
                value=value,
            )

    @staticmethod
    def _validate_entries(entries: tuple[LogEntry, ...]) -> None:
        for entry in entries:
            if not isinstance(entry.command, (Put, Delete)):
                raise InvalidKVCommand(f"unsupported KV command {entry.command!r}")

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._state:
            raise ValueError(f"unknown Raft node {node_id!r}")
