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
    """Create, validate, compact, and install durable KV snapshots.

    Snapshot compaction advances both the durable Raft log boundary and the
    durable state-machine boundary. KV state plus client deduplication identities
    therefore become the source of truth for the discarded applied prefix, while
    later applied entries remain explicit and replayable.
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
            client_requests=self._client_requests(node_id),
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
        """Persist a checkpoint and discard its covered Raft/applied prefixes."""
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
        self.kv.applier.compact_through(
            node_id,
            snapshot.last_included_index,
            snapshot.last_included_term,
        )
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

    def install(self, node_id: str, snapshot: KVSnapshot) -> None:
        """Install a newer durable snapshot into a follower behind its boundary.

        This receiver-side primitive intentionally handles the lagging-follower
        case where the local log does not extend past the incoming snapshot. A
        future transport slice can invoke it from InstallSnapshot RPC delivery;
        preserving a matching suffix beyond the snapshot boundary remains a
        separate optimization and is rejected here rather than silently losing it.
        """
        self._require_node(node_id)
        if not isinstance(snapshot, KVSnapshot):
            raise TypeError("installed snapshot must be a KVSnapshot")

        node = self.cluster.node(node_id)
        previous = self.latest(node_id)
        if previous is not None:
            if previous.last_included_index > snapshot.last_included_index:
                raise ValueError("installed snapshot cannot move backwards")
            if previous.last_included_index == snapshot.last_included_index:
                if previous != snapshot:
                    raise AssertionError("snapshot contents diverge at existing boundary")
                return
        if snapshot.last_included_index < node.log_base_index:
            raise ValueError("installed snapshot cannot precede the local Raft boundary")
        if node.last_log_index > snapshot.last_included_index:
            raise ValueError("snapshot install cannot discard a retained suffix beyond its boundary")

        persistent = self.sim.persistent_state[node_id]
        previous_log_base_index = node.log_base_index
        previous_last_log_index = node.last_log_index
        previous_applied_index = self.kv.applier.last_applied(node_id)

        persistent[self._PERSISTENT_KEY] = snapshot
        persistent["log"] = ()
        persistent["log_base_index"] = snapshot.last_included_index
        persistent["log_base_term"] = snapshot.last_included_term
        persistent["state_machine_applied"] = ()
        persistent["state_machine_base_index"] = snapshot.last_included_index
        persistent["state_machine_base_term"] = snapshot.last_included_term

        applier = self.kv.applier
        applier._base_index[node_id] = snapshot.last_included_index
        applier._base_term[node_id] = snapshot.last_included_term
        applier._applied[node_id] = []
        applier._last_applied[node_id] = snapshot.last_included_index

        self.kv._state[node_id] = dict(snapshot.state)
        self.kv._requests[node_id] = {
            (item.client_id, item.request_id): item.operation for item in snapshot.client_requests
        }
        if node.commit_index < snapshot.last_included_index:
            node.advance_commit_index(snapshot.last_included_index, source="install-snapshot")

        self.sim._record(
            "raft-kv-snapshot-install",
            node=node_id,
            previous_log_base_index=previous_log_base_index,
            previous_last_log_index=previous_last_log_index,
            previous_applied_index=previous_applied_index,
            last_included_index=snapshot.last_included_index,
            last_included_term=snapshot.last_included_term,
            key_count=len(snapshot.state),
            client_request_count=len(snapshot.client_requests),
        )
        self.cluster.assert_log_matching()
        applier.assert_state_machine_safety()

    def _client_requests(self, node_id: str) -> tuple[SnapshotClientRequest, ...]:
        requests = self.kv.client_requests(node_id)
        return tuple(
            SnapshotClientRequest(
                client_id=client_id,
                request_id=request_id,
                operation=requests[(client_id, request_id)],
            )
            for client_id, request_id in sorted(requests)
        )

    def _validate_against_applied_history(self, node_id: str, snapshot: KVSnapshot) -> None:
        base_index = self.kv.applier.applied_base_index(node_id)
        base_term = self.kv.applier.applied_base_term(node_id)
        if base_index > 0:
            if snapshot.last_included_index != base_index:
                raise AssertionError("snapshot index diverges from compacted applied boundary")
            if snapshot.last_included_term != base_term:
                raise AssertionError("snapshot term diverges from compacted applied boundary")
            return

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
