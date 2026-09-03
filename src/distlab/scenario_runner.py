from __future__ import annotations

from dataclasses import dataclass

from .client_history import KVClientHistory
from .kv import ClientRequest, Delete, Put, ReplicatedKV
from .lifecycle import NodeLifecycleKind, SeededLifecycleSchedule
from .linearizability import (
    LinearizabilityResult,
    OperationHistory,
    SingleKeyKVLinearizabilityChecker,
)
from .raft import LogEntry, RaftCluster, RaftRole
from .raft_invariants import RaftSafetyHarness
from .randomized_faults import SeededFaultSchedule
from .randomized_workload import ClientOperationKind, SeededClientWorkloadSchedule
from .replication import LeaderReplicator, ReplicationResponseMissing
from .simulator import Simulator, TraceRecord


class ScenarioExecutionError(RuntimeError):
    """Raised when an explicit replay schedule cannot be executed as specified."""


@dataclass(frozen=True, slots=True)
class ReplicatedKVScenarioResult:
    history: OperationHistory
    linearizability: LinearizabilityResult
    trace: tuple[TraceRecord, ...]
    snapshots: dict[str, dict[str, str]]


class ReplicatedKVScenarioRunner:
    """Replay explicit client, lifecycle, and fault schedules through Raft/KV.

    The runner is intentionally bounded to the current single-key linearizability
    foundation. Randomness is never consulted here: workload, lifecycle, and
    fault inputs must already be compiled into explicit schedules.
    """

    def __init__(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        *,
        lifecycle: SeededLifecycleSchedule | None = None,
        node_ids: tuple[str, ...] = ("n1", "n2", "n3"),
        leader_id: str = "n1",
    ) -> None:
        if leader_id not in node_ids:
            raise ValueError("leader_id must name a cluster node")
        if len(node_ids) < 1 or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be non-empty and unique")
        unknown = sorted({action.node_id for action in workload.actions} - set(node_ids))
        if unknown:
            raise ValueError(f"workload references unknown nodes: {unknown!r}")
        keys = {action.key for action in workload.actions}
        if len(keys) > 1:
            raise ValueError("scenario runner currently supports a single KV key")

        lifecycle = lifecycle or SeededLifecycleSchedule.empty(workload.seed)
        unknown_lifecycle = sorted(
            {action.node_id for action in lifecycle.actions} - set(node_ids)
        )
        if unknown_lifecycle:
            raise ValueError(
                f"lifecycle schedule references unknown nodes: {unknown_lifecycle!r}"
            )
        if any(
            action.before_action_index > len(workload.actions)
            for action in lifecycle.actions
        ):
            raise ValueError("lifecycle action references a workload boundary out of range")

        self.workload = workload
        self.faults = faults
        self.lifecycle = lifecycle
        self.node_ids = node_ids
        self.leader_id = leader_id

    def run(self) -> ReplicatedKVScenarioResult:
        sim = Simulator(fault_plan=self.faults.to_fault_plan())
        cluster = RaftCluster(sim, self.node_ids)
        safety = RaftSafetyHarness(cluster)
        leader = cluster.node(self.leader_id)
        leader.start_election()
        sim.run()
        if leader.role is not RaftRole.LEADER:
            raise ScenarioExecutionError(
                "configured leader could not win the deterministic election"
            )
        safety.checkpoint()

        replicator = LeaderReplicator(leader)
        kv = ReplicatedKV(cluster, applier=safety.state_machine)
        clients = KVClientHistory(kv)
        lifecycle_position = 0

        for action_index, action in enumerate(self.workload.actions):
            lifecycle_position = self._apply_lifecycle_boundary(
                action_index,
                lifecycle_position,
                sim,
                safety,
            )
            if action.kind is ClientOperationKind.GET:
                clients.read(
                    action.operation_id,
                    action.client_id,
                    action.node_id,
                    action.key,
                )
                continue

            if action.kind is ClientOperationKind.RETRY:
                assert action.retry_of is not None
                request = clients.pending_write(action.retry_of)
                if request is None:
                    sim._record(
                        "client-retry-suppressed",
                        attempt_operation_id=action.operation_id,
                        retry_of=action.retry_of,
                        client_id=action.client_id,
                        node=action.node_id,
                    )
                    continue
                if request.client_id != action.client_id:
                    raise ScenarioExecutionError(
                        "retry action client does not match original write"
                    )
                request = clients.retry_write(action.retry_of)
                self._drive_write_attempt(
                    leader,
                    replicator,
                    safety,
                    kv,
                    clients,
                    action.retry_of,
                    action.node_id,
                    request,
                )
                continue

            operation = (
                Put(action.key, action.value)
                if action.kind is ClientOperationKind.PUT
                else Delete(action.key)
            )
            assert action.request_id is not None
            request = clients.invoke_write(
                action.operation_id,
                action.client_id,
                action.request_id,
                operation,
            )
            self._drive_write_attempt(
                leader,
                replicator,
                safety,
                kv,
                clients,
                action.operation_id,
                action.node_id,
                request,
            )

        self._apply_lifecycle_boundary(
            len(self.workload.actions),
            lifecycle_position,
            sim,
            safety,
        )
        linearizability = SingleKeyKVLinearizabilityChecker().check(clients.history)
        return ReplicatedKVScenarioResult(
            history=clients.history,
            linearizability=linearizability,
            trace=tuple(sim.trace),
            snapshots={node_id: kv.snapshot(node_id) for node_id in self.node_ids},
        )

    def _apply_lifecycle_boundary(
        self,
        boundary: int,
        position: int,
        sim: Simulator,
        safety: RaftSafetyHarness,
    ) -> int:
        while position < len(self.lifecycle.actions):
            action = self.lifecycle.actions[position]
            if action.before_action_index != boundary:
                break
            if action.kind is NodeLifecycleKind.CRASH:
                if not sim.is_alive(action.node_id):
                    raise ScenarioExecutionError(
                        f"cannot crash already crashed node {action.node_id!r}"
                    )
                sim.crash(action.node_id)
            else:
                if sim.is_alive(action.node_id):
                    raise ScenarioExecutionError(
                        f"cannot restart live node {action.node_id!r}"
                    )
                sim.restart(action.node_id)
            sim._record(
                "scenario-lifecycle",
                action_id=action.action_id,
                node=action.node_id,
                action=action.kind.value,
                before_action_index=boundary,
            )
            safety.checkpoint()
            position += 1
        return position

    def _drive_write_attempt(
        self,
        leader,
        replicator: LeaderReplicator,
        safety: RaftSafetyHarness,
        kv: ReplicatedKV,
        clients: KVClientHistory,
        operation_id: str,
        response_node: str,
        request: ClientRequest,
    ) -> None:
        self._append_to_leader(leader, request)
        self._replicate_round(replicator)
        safety.checkpoint()
        self._replicate_round(replicator)
        safety.checkpoint()
        for node_id in self.node_ids:
            if leader.sim.is_alive(node_id):
                kv.apply_committed(node_id)
        safety.checkpoint()
        if kv.has_applied_request(
            response_node,
            request.client_id,
            request.request_id,
        ):
            clients.complete_write(operation_id, response_node)

        kv.assert_replica_consistency()

    @staticmethod
    def _append_to_leader(leader, request: ClientRequest) -> None:
        if leader.role is not RaftRole.LEADER:
            raise ScenarioExecutionError("client write requires the configured leader")
        previous = leader.log
        entry = LogEntry(term=leader.current_term, command=request)
        leader.sim.persistent_state[leader.node_id]["log"] = (*previous, entry)
        leader.sim._record(
            "raft-client-append",
            leader=leader.node_id,
            term=leader.current_term,
            index=len(previous) + 1,
            client_id=request.client_id,
            request_id=request.request_id,
        )
        if leader.log[: len(previous)] != previous:
            raise AssertionError("Leader Append-Only violated while appending client command")

    def _replicate_round(self, replicator: LeaderReplicator) -> None:
        for peer in replicator.leader.peers:
            try:
                replicator.replicate(peer, max_attempts=1)
            except ReplicationResponseMissing:
                continue
