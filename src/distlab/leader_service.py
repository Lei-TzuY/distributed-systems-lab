from __future__ import annotations

from dataclasses import dataclass

from .client_history import KVClientHistory
from .commit_recovery import CommitRecoveryBarrier
from .kv import ClientRequest, ClientRequestConflict, Delete, Put, ReplicatedKV
from .leader_authority import LeaderGenerationGuard
from .leader_runtime import LeaderRuntimeIdentity, LeaderRuntimeSupervisor
from .linearizability import Get
from .linearizable_read import LinearizableKVReader, ReadBarrierEvidence
from .membership import ReconfigurableRaftCluster
from .membership_replication import MembershipAwareLeaderReplicator
from .raft import LogEntry, RaftNode
from .replication import LeaderReplicator, ReplicationError, ReplicationResponseMissing


class LeaderKVServiceError(RuntimeError):
    """Base error for generation-fenced leader data-plane failures."""


class LeaderWriteQuorumUnavailable(LeaderKVServiceError):
    """Raised when a client write cannot reach the active commit quorum."""


class LeaderReadBarrierUnavailable(LeaderKVServiceError):
    """Raised when a leader cannot establish a current-term read barrier."""


@dataclass(frozen=True, slots=True)
class LeaderWriteResult:
    leader_id: str
    term: int
    generation: int
    request: ClientRequest
    log_index: int | None
    commit_index: int


@dataclass(frozen=True, slots=True)
class LeaderReadResult:
    leader_id: str
    term: int
    generation: int
    key: str
    value: str | None
    commit_index: int


class LeaderKVService:
    """Fence client-visible KV work with one explicit leader runtime generation.

    The service deliberately consumes only the public runtime identity API. It
    does not own heartbeat scheduling, so it can coexist with an independently
    managed leader-runtime tick loop. Every operation validates generation
    authority before protocol mutation and again before publishing a client
    response. A committed operation whose generation disappears before response
    therefore remains pending and can be retried exactly on the next leader.
    """

    def __init__(
        self,
        supervisor: LeaderRuntimeSupervisor,
        kv: ReplicatedKV,
        clients: KVClientHistory | None = None,
    ) -> None:
        if kv.cluster is not supervisor.cluster:
            raise ValueError("KV state and runtime supervisor must share one cluster")
        if clients is not None and clients.kv is not kv:
            raise ValueError("client history and service must share one KV state")
        self.supervisor = supervisor
        self.cluster = supervisor.cluster
        self.sim = self.cluster.sim
        self.authority = LeaderGenerationGuard(self.cluster)
        self.kv = kv
        self.clients = clients if clients is not None else KVClientHistory(kv)

    def write(
        self,
        leader_id: str,
        expected_generation: int,
        *,
        operation_id: str,
        client_id: str,
        request_id: int,
        operation: Put | Delete,
        max_attempts_per_peer: int = 8,
    ) -> LeaderWriteResult:
        identity = self._require_generation(leader_id, expected_generation)
        request = self.clients.invoke_write(
            operation_id,
            client_id,
            request_id,
            operation,
        )
        return self._commit_write(
            identity,
            operation_id,
            request,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def retry_write(
        self,
        leader_id: str,
        expected_generation: int,
        *,
        operation_id: str,
        max_attempts_per_peer: int = 8,
    ) -> LeaderWriteResult:
        identity = self._require_generation(leader_id, expected_generation)
        request = self.clients.retry_write(operation_id)
        return self._commit_write(
            identity,
            operation_id,
            request,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def linearizable_read(
        self,
        leader_id: str,
        expected_generation: int,
        *,
        operation_id: str,
        client_id: str,
        key: str,
        max_attempts_per_peer: int = 8,
    ) -> LeaderReadResult:
        if not key:
            raise ValueError("KV key must be non-empty")
        identity = self._require_generation(leader_id, expected_generation)
        self.clients.history.invoke(operation_id, client_id, Get(key))
        self.sim._record(
            "client-invoke",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            node=leader_id,
            consistency="linearizable",
            generation=identity.generation,
        )
        return self._complete_read(
            identity,
            operation_id,
            client_id,
            key,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def retry_linearizable_read(
        self,
        leader_id: str,
        expected_generation: int,
        *,
        operation_id: str,
        client_id: str,
        key: str,
        max_attempts_per_peer: int = 8,
    ) -> LeaderReadResult:
        identity = self._require_generation(leader_id, expected_generation)
        pending = {item.operation_id: item for item in self.clients.history.pending()}.get(
            operation_id
        )
        if pending is None:
            raise ValueError(f"unknown pending linearizable read {operation_id!r}")
        if pending.client_id != client_id or pending.operation != Get(key):
            raise ValueError("linearizable read retry must match the pending invocation")
        if self.clients.history.is_abandoned(operation_id):
            raise ValueError(f"abandoned linearizable read {operation_id!r} cannot be retried")
        self.sim._record(
            "client-retry",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            operation="get",
            key=key,
            node=leader_id,
            consistency="linearizable",
            generation=identity.generation,
        )
        return self._complete_read(
            identity,
            operation_id,
            client_id,
            key,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def linearizable_barrier(
        self,
        leader_id: str,
        expected_generation: int,
        *,
        max_attempts_per_peer: int = 8,
    ) -> ReadBarrierEvidence:
        """Confirm generation-fenced leader authority without creating client history."""

        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")
        identity = self._require_generation(leader_id, expected_generation)
        leader = self.cluster.node(leader_id)
        replicator = self._new_replicator(leader)
        self._ensure_current_term_commit(
            identity,
            replicator,
            max_attempts_per_peer=max_attempts_per_peer,
        )
        reader = LinearizableKVReader(self.kv, replicator)
        evidence = reader.barrier(max_attempts_per_peer=max_attempts_per_peer)
        self._require_generation(leader_id, expected_generation)
        self.sim._record(
            "raft-leader-service-barrier",
            leader=leader_id,
            term=identity.term,
            generation=identity.generation,
            commit_index=evidence.commit_index,
            acknowledged_voters=evidence.acknowledged_voters,
            quorum_mode=evidence.quorum_mode,
        )
        return evidence

    def _commit_write(
        self,
        identity: LeaderRuntimeIdentity,
        operation_id: str,
        request: ClientRequest,
        *,
        max_attempts_per_peer: int,
    ) -> LeaderWriteResult:
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")
        self._require_generation(identity.leader_id, identity.generation)
        leader = self.cluster.node(identity.leader_id)

        self.kv.apply_committed(identity.leader_id)
        if self.kv.has_applied_request(
            identity.leader_id, request.client_id, request.request_id
        ):
            applied = self.kv.client_requests(identity.leader_id)[
                (request.client_id, request.request_id)
            ]
            if applied != request.operation:
                raise ClientRequestConflict(
                    "applied request identity conflicts with pending client write"
                )
            self._require_generation(identity.leader_id, identity.generation)
            self.clients.complete_write(operation_id, identity.leader_id)
            return self._write_result(identity, request, self._request_log_index(leader, request))

        replicator = self._new_replicator(leader)
        request_index = self._request_log_index(leader, request)
        if request_index is None:
            request_index = self._append_command(leader, request, trace_kind="raft-client-append")

        target_commit_index = request_index
        if leader.log_view.term_at(request_index) != leader.current_term:
            target_commit_index = self._current_term_target_after(leader, request_index)
            if target_commit_index is None:
                target_commit_index = self._append_command(
                    leader,
                    CommitRecoveryBarrier(),
                    trace_kind="raft-current-term-barrier",
                )

        self._drive_commit(
            identity,
            replicator,
            target_commit_index,
            max_attempts_per_peer=max_attempts_per_peer,
            failure_type=LeaderWriteQuorumUnavailable,
        )
        self.kv.apply_committed(identity.leader_id)
        if not self.kv.has_applied_request(
            identity.leader_id, request.client_id, request.request_id
        ):
            raise AssertionError("committed client request was not applied on the leader")

        self._require_generation(identity.leader_id, identity.generation)
        self.clients.complete_write(operation_id, identity.leader_id)
        result = self._write_result(identity, request, request_index)
        self.sim._record(
            "raft-leader-service-write",
            leader=result.leader_id,
            term=result.term,
            generation=result.generation,
            log_index=result.log_index,
            commit_index=result.commit_index,
            client_id=request.client_id,
            request_id=request.request_id,
        )
        return result

    def _complete_read(
        self,
        identity: LeaderRuntimeIdentity,
        operation_id: str,
        client_id: str,
        key: str,
        *,
        max_attempts_per_peer: int,
    ) -> LeaderReadResult:
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")
        leader = self.cluster.node(identity.leader_id)
        replicator = self._new_replicator(leader)
        self._ensure_current_term_commit(
            identity,
            replicator,
            max_attempts_per_peer=max_attempts_per_peer,
        )
        reader = LinearizableKVReader(self.kv, replicator)
        value = reader.get(key, max_attempts_per_peer=max_attempts_per_peer)
        self._require_generation(identity.leader_id, identity.generation)
        self.clients.history.respond(operation_id, value)
        self.sim._record(
            "client-response",
            operation_id=operation_id,
            client_id=client_id,
            request_id=None,
            node=identity.leader_id,
            result=value,
            consistency="linearizable",
            generation=identity.generation,
        )
        result = LeaderReadResult(
            leader_id=identity.leader_id,
            term=identity.term,
            generation=identity.generation,
            key=key,
            value=value,
            commit_index=leader.commit_index,
        )
        self.sim._record(
            "raft-leader-service-read",
            leader=result.leader_id,
            term=result.term,
            generation=result.generation,
            commit_index=result.commit_index,
            key=key,
            value=value,
        )
        return result

    def _ensure_current_term_commit(
        self,
        identity: LeaderRuntimeIdentity,
        replicator: LeaderReplicator,
        *,
        max_attempts_per_peer: int,
    ) -> None:
        leader = replicator.leader
        if (
            leader.commit_index > 0
            and leader.log_view.term_at(leader.commit_index) == leader.current_term
        ):
            return
        target = self._append_command(
            leader,
            CommitRecoveryBarrier(),
            trace_kind="raft-current-term-barrier",
        )
        self._drive_commit(
            identity,
            replicator,
            target,
            max_attempts_per_peer=max_attempts_per_peer,
            failure_type=LeaderReadBarrierUnavailable,
        )
        self.kv.apply_committed(identity.leader_id)

    def _drive_commit(
        self,
        identity: LeaderRuntimeIdentity,
        replicator: LeaderReplicator,
        target_index: int,
        *,
        max_attempts_per_peer: int,
        failure_type: type[LeaderKVServiceError],
    ) -> None:
        leader = replicator.leader
        if leader.commit_index >= target_index:
            return
        for peer in self._replication_peers(leader):
            self._require_generation(identity.leader_id, identity.generation)
            try:
                replicator.replicate(peer, max_attempts=max_attempts_per_peer)
            except ReplicationResponseMissing:
                continue
            except ReplicationError as exc:
                self._require_generation(identity.leader_id, identity.generation)
                raise LeaderKVServiceError(str(exc)) from exc
            self._require_generation(identity.leader_id, identity.generation)
            if leader.commit_index >= target_index:
                return

        self.sim._record(
            "raft-leader-service-quorum-failed",
            leader=identity.leader_id,
            term=identity.term,
            generation=identity.generation,
            target_index=target_index,
            commit_index=leader.commit_index,
            operation="write"
            if failure_type is LeaderWriteQuorumUnavailable
            else "read-barrier",
        )
        raise failure_type(
            f"leader {identity.leader_id!r} could not commit index {target_index}"
        )

    def _require_generation(
        self, leader_id: str, expected_generation: int
    ) -> LeaderRuntimeIdentity:
        return self.authority.require(leader_id, expected_generation)

    def _new_replicator(self, leader: RaftNode) -> LeaderReplicator:
        if isinstance(self.cluster, ReconfigurableRaftCluster):
            return MembershipAwareLeaderReplicator(leader)
        return LeaderReplicator(leader)

    def _replication_peers(self, leader: RaftNode) -> tuple[str, ...]:
        if isinstance(self.cluster, ReconfigurableRaftCluster):
            voters = self.cluster.voting_configuration.voters
            return tuple(sorted(voters - {leader.node_id}))
        return tuple(sorted(leader.peers))

    def _append_command(self, leader: RaftNode, command: object, *, trace_kind: str) -> int:
        previous = leader.log
        leader._persist_log((*previous, LogEntry(term=leader.current_term, command=command)))
        index = leader.log_view.last_index
        details: dict[str, object] = {
            "leader": leader.node_id,
            "term": leader.current_term,
            "index": index,
        }
        if isinstance(command, ClientRequest):
            details.update(
                client_id=command.client_id,
                request_id=command.request_id,
            )
        self.sim._record(trace_kind, **details)
        if leader.log[: len(previous)] != previous:
            raise AssertionError("Leader Append-Only violated while appending service command")
        return index

    @staticmethod
    def _request_log_index(leader: RaftNode, request: ClientRequest) -> int | None:
        log = leader.log_view
        for index in range(log.first_retained_index, log.last_index + 1):
            command = log.entry_at(index).command
            if not isinstance(command, ClientRequest):
                continue
            if command.client_id != request.client_id or command.request_id != request.request_id:
                continue
            if command.operation != request.operation:
                raise ClientRequestConflict(
                    "leader log contains conflicting client request identity"
                )
            return index
        return None

    @staticmethod
    def _current_term_target_after(leader: RaftNode, request_index: int) -> int | None:
        log = leader.log_view
        for index in range(log.last_index, request_index, -1):
            if log.term_at(index) == leader.current_term:
                return index
        return None

    def _write_result(
        self,
        identity: LeaderRuntimeIdentity,
        request: ClientRequest,
        log_index: int | None,
    ) -> LeaderWriteResult:
        return LeaderWriteResult(
            leader_id=identity.leader_id,
            term=identity.term,
            generation=identity.generation,
            request=request,
            log_index=log_index,
            commit_index=self.cluster.node(identity.leader_id).commit_index,
        )
