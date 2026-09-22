from __future__ import annotations

from dataclasses import dataclass

from .heartbeat import HeartbeatRound, LeaderHeartbeatController
from .leader_liveness import LeaderQuorumMonitor
from .membership import ReconfigurableRaftCluster
from .membership_replication import MembershipAwareLeaderReplicator
from .raft import (
    LeadershipLifecycleEvent,
    LeadershipLifecycleKind,
    RaftCluster,
    RaftRole,
)
from .replication import LeaderReplicator


class LeaderRuntimeError(RuntimeError):
    """Base error for leader-runtime ownership failures."""


class LeaderRuntimeUnavailable(LeaderRuntimeError):
    """Raised when a node has no active leader runtime generation."""


class StaleLeaderRuntimeGeneration(LeaderRuntimeError):
    """Raised when a caller presents an already retired runtime generation."""


@dataclass(frozen=True, slots=True)
class LeaderRuntimeIdentity:
    leader_id: str
    term: int
    generation: int


@dataclass(slots=True)
class _LeaderRuntime:
    identity: LeaderRuntimeIdentity
    replicator: LeaderReplicator
    monitor: LeaderQuorumMonitor
    heartbeat: LeaderHeartbeatController


class LeaderRuntimeSupervisor:
    """Own heartbeat/CheckQuorum runtime state for every node-local leader generation.

    Raft can temporarily contain leaders from different terms on opposite sides of
    a partition. Runtime ownership therefore follows each node's local authority
    instead of selecting one simulator-global leader. A runtime is created only
    after that node wins an election and is retired when the same node locally
    loses leader authority, crashes, or restarts.
    """

    def __init__(
        self,
        cluster: RaftCluster,
        *,
        heartbeat_interval: int,
        response_timeout: int,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        if response_timeout > heartbeat_interval:
            raise ValueError("response_timeout cannot exceed heartbeat_interval")

        self.cluster = cluster
        self.sim = cluster.sim
        self.heartbeat_interval = heartbeat_interval
        self.response_timeout = response_timeout
        self._runtimes: dict[str, _LeaderRuntime] = {}
        cluster.register_leadership_observer(self._handle_lifecycle_event)

    @property
    def active_runtimes(self) -> tuple[LeaderRuntimeIdentity, ...]:
        return tuple(
            runtime.identity
            for runtime in sorted(
                self._runtimes.values(),
                key=lambda runtime: runtime.identity.generation,
            )
        )

    def runtime_identity(self, leader_id: str) -> LeaderRuntimeIdentity:
        runtime = self._runtimes.get(leader_id)
        if runtime is None:
            raise LeaderRuntimeUnavailable(
                f"node {leader_id!r} has no active leader runtime"
            )
        return runtime.identity

    def run_heartbeats(
        self,
        leader_id: str,
        *,
        rounds: int,
        expected_generation: int | None = None,
    ) -> tuple[HeartbeatRound, ...]:
        runtime = self._runtimes.get(leader_id)
        if runtime is None:
            raise LeaderRuntimeUnavailable(
                f"node {leader_id!r} has no active leader runtime"
            )
        identity = runtime.identity
        if expected_generation is not None and expected_generation != identity.generation:
            raise StaleLeaderRuntimeGeneration(
                f"leader runtime generation {expected_generation} is stale; "
                f"active generation is {identity.generation}"
            )

        node = self.cluster.node(leader_id)
        if (
            not self.sim.is_alive(leader_id)
            or node.role is not RaftRole.LEADER
            or node.current_term != identity.term
        ):
            raise LeaderRuntimeUnavailable(
                f"node {leader_id!r} no longer owns runtime generation "
                f"{identity.generation}"
            )
        return runtime.heartbeat.run(rounds=rounds)

    def _handle_lifecycle_event(self, event: LeadershipLifecycleEvent) -> None:
        if event.kind is LeadershipLifecycleKind.ACQUIRED:
            self._start_runtime(event)
            return
        if event.kind is LeadershipLifecycleKind.RETIRED:
            self._retire_runtime(event)
            return
        raise ValueError(f"unsupported leadership lifecycle kind {event.kind!r}")

    def _start_runtime(self, event: LeadershipLifecycleEvent) -> None:
        node = self.cluster.node(event.node_id)
        if node.role is not RaftRole.LEADER or node.current_term != event.term:
            raise LeaderRuntimeError(
                "leader runtime acquisition requires matching node authority"
            )

        existing = self._runtimes.get(event.node_id)
        if existing is not None:
            if existing.identity.generation == event.generation:
                return
            raise LeaderRuntimeError(
                f"node {event.node_id!r} still owns runtime generation "
                f"{existing.identity.generation}"
            )

        if isinstance(self.cluster, ReconfigurableRaftCluster):
            replicator = MembershipAwareLeaderReplicator(node)
        else:
            replicator = LeaderReplicator(node)
        monitor = LeaderQuorumMonitor(replicator)
        heartbeat = LeaderHeartbeatController(
            monitor,
            heartbeat_interval=self.heartbeat_interval,
            response_timeout=self.response_timeout,
        )
        identity = LeaderRuntimeIdentity(
            leader_id=event.node_id,
            term=event.term,
            generation=event.generation,
        )
        self._runtimes[event.node_id] = _LeaderRuntime(
            identity=identity,
            replicator=replicator,
            monitor=monitor,
            heartbeat=heartbeat,
        )
        self.sim._record(
            "raft-leader-runtime-start",
            node=identity.leader_id,
            term=identity.term,
            generation=identity.generation,
            reason=event.reason,
            membership_aware=isinstance(replicator, MembershipAwareLeaderReplicator),
        )

    def _retire_runtime(self, event: LeadershipLifecycleEvent) -> None:
        runtime = self._runtimes.get(event.node_id)
        if runtime is None:
            return
        if runtime.identity.generation != event.generation:
            return
        self._runtimes.pop(event.node_id)
        self.sim._record(
            "raft-leader-runtime-stop",
            node=event.node_id,
            term=event.term,
            generation=event.generation,
            reason=event.reason,
        )
