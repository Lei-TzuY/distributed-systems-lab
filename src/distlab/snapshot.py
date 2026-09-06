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
class SnapshotVotingConfiguration:
    """Committed Raft voting configuration represented by a snapshot boundary."""

    committed_index: int
    old_voters: tuple[str, ...]
    new_voters: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.committed_index < 0:
            raise ValueError("snapshot membership commit index must be non-negative")
        if not self.old_voters:
            raise ValueError("snapshot membership requires at least one old voter")
        if tuple(sorted(self.old_voters)) != self.old_voters:
            raise ValueError("snapshot old voters must be canonically sorted")
        if len(set(self.old_voters)) != len(self.old_voters):
            raise ValueError("snapshot old voters must be unique")
        if self.new_voters is not None:
            if not self.new_voters:
                raise ValueError("snapshot joint membership requires at least one new voter")
            if tuple(sorted(self.new_voters)) != self.new_voters:
                raise ValueError("snapshot new voters must be canonically sorted")
            if len(set(self.new_voters)) != len(self.new_voters):
                raise ValueError("snapshot new voters must be unique")


@dataclass(frozen=True, slots=True)
class KVSnapshot:
    """Durable replicated-state-machine checkpoint at an applied Raft index."""

    last_included_index: int
    last_included_term: int
    state: tuple[tuple[str, str], ...]
    client_requests: tuple[SnapshotClientRequest, ...]
    voting_configuration: SnapshotVotingConfiguration | None = None

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
        membership = self.voting_configuration
        if membership is not None and membership.committed_index > self.last_included_index:
            raise ValueError("snapshot membership cannot exceed the snapshot boundary")


class KVSnapshotStore:
    """Create, validate, compact, and install durable KV snapshots.

    Snapshot compaction advances both the durable Raft log boundary and the
    durable state-machine boundary. KV state plus client deduplication identities
    therefore become the source of truth for the discarded applied prefix, while
    later applied entries remain explicit and replayable. Reconfigurable clusters
    additionally checkpoint the committed voting configuration effective at the
    snapshot boundary so membership history remains recoverable after compaction.
    """

    _PERSISTENT_KEY = "kv_snapshot"
    _MEMBERSHIP_COMMIT_INDEX = "membership_commit_index"

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
            voting_configuration=self._snapshot_voting_configuration(last_applied),
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
            membership_commit_index=(
                snapshot.voting_configuration.committed_index
                if snapshot.voting_configuration is not None
                else None
            ),
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

    def install(
        self,
        node_id: str,
        snapshot: KVSnapshot,
        *,
        preserve_matching_suffix: bool = False,
    ) -> None:
        """Install a newer durable snapshot into a follower behind its boundary."""
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

        retained_suffix = ()
        if node.last_log_index > snapshot.last_included_index:
            if not preserve_matching_suffix:
                raise ValueError(
                    "snapshot install cannot discard a retained suffix beyond its boundary"
                )
            if not node.log_view.prefix_matches(
                snapshot.last_included_index,
                snapshot.last_included_term,
            ):
                raise ValueError(
                    "snapshot install cannot preserve retained suffix without a matching boundary"
                )
            retained_suffix = node.log_view.suffix_from(snapshot.last_included_index + 1)

        previous_applied_index = self.kv.applier.last_applied(node_id)
        if previous_applied_index > snapshot.last_included_index:
            raise ValueError("snapshot install cannot roll back applied state beyond its boundary")

        self._validate_snapshot_membership(snapshot)
        persistent = self.sim.persistent_state[node_id]
        previous_log_base_index = node.log_base_index
        previous_last_log_index = node.last_log_index

        persistent[self._PERSISTENT_KEY] = snapshot
        persistent["log"] = retained_suffix
        persistent["log_base_index"] = snapshot.last_included_index
        persistent["log_base_term"] = snapshot.last_included_term
        persistent["state_machine_applied"] = ()
        persistent["state_machine_base_index"] = snapshot.last_included_index
        persistent["state_machine_base_term"] = snapshot.last_included_term
        if snapshot.voting_configuration is not None:
            previous_membership_index = int(persistent.get(self._MEMBERSHIP_COMMIT_INDEX, 0))
            persistent[self._MEMBERSHIP_COMMIT_INDEX] = max(
                previous_membership_index,
                snapshot.voting_configuration.committed_index,
            )

        applier = self.kv.applier
        applier._base_index[node_id] = snapshot.last_included_index
        applier._base_term[node_id] = snapshot.last_included_term
        applier._applied[node_id] = []
        applier._last_applied[node_id] = snapshot.last_included_index

        self.kv._state[node_id] = dict(snapshot.state)
        self.kv._requests[node_id] = {
            (item.client_id, item.request_id): item.operation
            for item in snapshot.client_requests
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
            retained_count=len(retained_suffix),
            key_count=len(snapshot.state),
            client_request_count=len(snapshot.client_requests),
            membership_commit_index=(
                snapshot.voting_configuration.committed_index
                if snapshot.voting_configuration is not None
                else None
            ),
        )
        self.cluster.assert_log_matching()
        applier.assert_state_machine_safety()

    def _snapshot_voting_configuration(
        self, last_applied: int
    ) -> SnapshotVotingConfiguration | None:
        from .membership import ReconfigurableRaftCluster

        if not isinstance(self.cluster, ReconfigurableRaftCluster):
            return None
        committed_index = max(
            int(self.sim.persistent_state[node_id].get(self._MEMBERSHIP_COMMIT_INDEX, 0))
            for node_id in self.cluster.node_ids
        )
        if committed_index > last_applied:
            raise ValueError(
                "cannot snapshot before the latest committed membership configuration is applied"
            )
        configuration = self.cluster.voting_configuration
        return SnapshotVotingConfiguration(
            committed_index=committed_index,
            old_voters=tuple(sorted(configuration.old_voters)),
            new_voters=(
                tuple(sorted(configuration.new_voters))
                if configuration.new_voters is not None
                else None
            ),
        )

    def _validate_snapshot_membership(self, snapshot: KVSnapshot) -> None:
        from .membership import ReconfigurableRaftCluster

        membership = snapshot.voting_configuration
        if isinstance(self.cluster, ReconfigurableRaftCluster):
            if membership is None:
                raise ValueError("reconfigurable Raft snapshot requires voting configuration")
            voters = set(membership.old_voters)
            if membership.new_voters is not None:
                voters.update(membership.new_voters)
            if not voters <= set(self.cluster.node_ids):
                raise ValueError("snapshot membership references unknown Raft nodes")
        elif membership is not None:
            raise ValueError("non-reconfigurable Raft snapshot cannot carry voting configuration")

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
        self._validate_snapshot_membership(snapshot)
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

        from .membership_log import JointConsensusCommand, StableConsensusCommand

        state: dict[str, str] = {}
        requests: dict[tuple[str, int], SnapshotClientRequest] = {}
        for entry in included:
            command = entry.command
            if isinstance(command, (JointConsensusCommand, StableConsensusCommand)):
                continue
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
