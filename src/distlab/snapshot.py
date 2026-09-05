from __future__ import annotations

from dataclasses import dataclass

from .commit_recovery import CommitRecoveryBarrier
from .kv import ClientRequest, Delete, KVOperation, Put, ReplicatedKV
from .raft import RaftCluster


@dataclass(frozen=True, slots=True)
class SnapshotClientRequest:
    client_id: str
    request_id: int
    operation: KVOperation


@dataclass(frozen=True, slots=True)
class KVSnapshot:
    """Durable replicated-state-machine checkpoint at an applied Raft index."""

    last_included_index: int
    last_included_term: int
    state: tuple[tuple[str, str], ...]
    client_requests: tuple[SnapshotClientRequest, ...]

    def __post_init__(self) -> None:
        if self.last_included_index <= 0:
            raise ValueError("snapshot index must be positive")
        if self.last_included_term < 0:
            raise ValueError("snapshot term must be non-negative")
        if tuple(sorted(self.state)) != self.state:
            raise ValueError("snapshot state must be canonically sorted")
        identities = [(item.client_id, item.request_id) for item in self.client_requests]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise ValueError("snapshot client requests must be unique and canonically sorted")


class KVSnapshotStore:
    """Create, validate, and compact durable KV snapshots.

    The checkpoint includes client deduplication identities because restoring only
    the user-visible map would allow a retried request to execute twice after a
    future InstallSnapshot path. Snapshot-driven compaction advances the durable
    Raft log boundary while the full applied history remains authoritative until
    state-machine history compaction is implemented separately.
    """

    _PERSISTENT_KEY = "kv_snapshot"

    def __init__(self, cluster: RaftCluster, kv: ReplicatedKV) -> None:
        if kv.cluster is not cluster:
            raise ValueError("KV state must belong to the same Raft cluster")
        self.cluster = cluster
        self.kv = kv
        self.sim = cluster.sim
        for node_id in cluster.node_ids:
            snapshot = self.latest(node_id)
            if snapshot is not None:
                self._validate_against_applied_history(node_id, snapshot)

    def latest(self, node_id: str) -> KVSnapshot | None:
        self._require_node(node_id)
        value = self.sim.persistent_state[node_id].get(self._PERSISTENT_KEY)
        if value is None:
            return None
        if not isinstance(value, KVSnapshot):
            raise TypeError("persistent KV snapshot must be a KVSnapshot")
        return value

    def create(self, node_id: str) -> KVSnapshot:
        self._require_node(node_id)
        last_applied = self.kv.applier.last_applied(node_id)
        if last_applied == 0:
            raise ValueError("cannot snapshot an empty applied state machine")

        node = self.cluster.node(node_id)
        if last_applied > node.last_log_index:
            raise AssertionError("applied index cannot exceed the local Raft log")
        snapshot = KVSnapshot(
            last_included_index=last_applied,
            last_included_term=node.log_view.term_at(last_applied),
            state=tuple(sorted(self.kv.snapshot(node_id).items())),
            client_requests=self._client_requests(node_id, last_applied),
        )
        previous = self.latest(node_id)
        if previous is not None and previous.last_included_index > last_applied:
            raise AssertionError("snapshot index cannot move backwards")
        self.sim.persistent_state[node_id][self._PERSISTENT_KEY] = snapshot
        self.sim._record(
            "raft-kv-snapshot-persist",
            node=node_id,
            last_included_index=snapshot.last_included_index,
            last_included_term=snapshot.last_included_term,
            key_count=len(snapshot.state),
            client_request_count=len(snapshot.client_requests),
        )
        return snapshot

    def compact(self, node_id: str) -> KVSnapshot:
        """Persist a checkpoint and discard its covered retained Raft prefix."""
        snapshot = self.create(node_id)
        node = self.cluster.node(node_id)
        previous_base_index = node.log_base_index
        previous_retained_count = len(node.log)
        compacted = node.log_view.compact_through(snapshot.last_included_index)
        if compacted.base_term != snapshot.last_included_term:
            raise AssertionError("snapshot term diverges from compacted Raft boundary")

        persistent = self.sim.persistent_state[node_id]
        persistent["log"] = compacted.entries
        persistent["log_base_index"] = compacted.base_index
        persistent["log_base_term"] = compacted.base_term
        self.sim._record(
            "raft-log-compact",
            node=node_id,
            previous_base_index=previous_base_index,
            log_base_index=compacted.base_index,
            log_base_term=compacted.base_term,
            previous_retained_count=previous_retained_count,
            retained_count=len(compacted.entries),
        )
        return snapshot

    def _client_requests(
        self, node_id: str, last_included_index: int
    ) -> tuple[SnapshotClientRequest, ...]:
        requests: dict[tuple[str, int], SnapshotClientRequest] = {}
        entries = self.kv.applier.applied_entries(node_id)[:last_included_index]
        for entry in entries:
            command = entry.command
            if not isinstance(command, ClientRequest):
                continue
            identity = (command.client_id, command.request_id)
            item = SnapshotClientRequest(
                client_id=command.client_id,
                request_id=command.request_id,
                operation=command.operation,
            )
            existing = requests.get(identity)
            if existing is not None and existing != item:
                raise AssertionError("conflicting client identity in applied snapshot prefix")
            requests[identity] = item
        return tuple(requests[key] for key in sorted(requests))

    def _validate_against_applied_history(self, node_id: str, snapshot: KVSnapshot) -> None:
        history = self.kv.applier.applied_entries(node_id)
        if snapshot.last_included_index > len(history):
            raise AssertionError("snapshot exceeds durable applied history")
        included = history[: snapshot.last_included_index]
        if included[-1].term != snapshot.last_included_term:
            raise AssertionError("snapshot term diverges from durable applied history")

        state: dict[str, str] = {}
        requests: dict[tuple[str, int], SnapshotClientRequest] = {}
        for entry in included:
            command = entry.command
            operation: object = command
            if isinstance(command, ClientRequest):
                identity = (command.client_id, command.request_id)
                item = SnapshotClientRequest(
                    client_id=command.client_id,
                    request_id=command.request_id,
                    operation=command.operation,
                )
                previous = requests.get(identity)
                if previous is not None:
                    if previous != item:
                        raise AssertionError("snapshot prefix contains conflicting client request")
                    continue
                requests[identity] = item
                operation = command.operation
            if isinstance(operation, Put):
                state[operation.key] = operation.value
            elif isinstance(operation, Delete):
                state.pop(operation.key, None)
            elif isinstance(operation, CommitRecoveryBarrier):
                continue
            else:
                raise TypeError(f"unsupported snapshot command {type(operation).__name__}")

        if tuple(sorted(state.items())) != snapshot.state:
            raise AssertionError("snapshot KV state diverges from durable applied history")
        expected_requests = tuple(requests[key] for key in sorted(requests))
        if expected_requests != snapshot.client_requests:
            raise AssertionError("snapshot dedup state diverges from durable applied history")

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.cluster.nodes:
            raise ValueError(f"unknown Raft node {node_id!r}")
