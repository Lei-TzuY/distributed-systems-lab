from __future__ import annotations

from dataclasses import dataclass

from .client_history import KVClientHistory
from .kv import ClientRequest, Delete, Put, ReplicatedKV
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
    """Replay explicit client and fault schedules through a small Raft/KV cluster.

    The runner is intentionally bounded to the current single-key linearizability
    foundation. Randomness is never consulted here: both workload and fault input
    must already be compiled into explicit schedules.
    """

    def __init__(
        self,
        workload: SeededClientWorkloadSchedule,
        faults: SeededFaultSchedule,
        *,
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
        self.workload = workload
        self.faults = faults
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
        kv = ReplicatedKV(cluster)
        clients = KVClientHistory(kv)

        for action in self.workload.actions:
            if action.kind is ClientOperationKind.GET:
                clients.read(
                    action.operation_id,
                    action.client_id,
                    action.node_id,
                    action.key,
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
            self._append_to_leader(leader, request)
            self._replicate_round(replicator)
            safety.checkpoint()
            self._replicate_round(replicator)
            safety.checkpoint()
            for node_id in self.node_ids:
                kv.apply_committed(node_id)
            if kv.has_applied_request(
                action.node_id,
                action.client_id,
                action.request_id,
            ):
                clients.complete_write(action.operation_id, action.node_id)

            kv.assert_replica_consistency()

        linearizability = SingleKeyKVLinearizabilityChecker().check(clients.history)
        return ReplicatedKVScenarioResult(
            history=clients.history,
            linearizability=linearizability,
            trace=tuple(sim.trace),
            snapshots={node_id: kv.snapshot(node_id) for node_id in self.node_ids},
        )

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
