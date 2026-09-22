from __future__ import annotations

from .leader_runtime import (
    LeaderRuntimeIdentity,
    LeaderRuntimeUnavailable,
    StaleLeaderRuntimeGeneration,
)
from .raft import RaftCluster, RaftRole


class LeaderGenerationGuard:
    """Validate node-local leader generation authority without owning runtime ticks."""

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.sim = cluster.sim

    def current(self, leader_id: str) -> LeaderRuntimeIdentity:
        active = self.cluster.active_leader_generations.get(leader_id)
        if active is None:
            raise LeaderRuntimeUnavailable(
                f"node {leader_id!r} has no active leader runtime generation"
            )
        term, generation = active
        node = self.cluster.node(leader_id)
        if (
            not self.sim.is_alive(leader_id)
            or node.role is not RaftRole.LEADER
            or node.current_term != term
        ):
            raise LeaderRuntimeUnavailable(
                f"node {leader_id!r} no longer owns runtime generation {generation}"
            )
        return LeaderRuntimeIdentity(
            leader_id=leader_id,
            term=term,
            generation=generation,
        )

    def require(
        self,
        leader_id: str,
        expected_generation: int,
    ) -> LeaderRuntimeIdentity:
        identity = self.current(leader_id)
        if identity.generation != expected_generation:
            raise StaleLeaderRuntimeGeneration(
                f"leader runtime generation {expected_generation} is stale; "
                f"active generation is {identity.generation}"
            )
        return identity

    def validate(self, identity: LeaderRuntimeIdentity) -> LeaderRuntimeIdentity:
        current = self.require(identity.leader_id, identity.generation)
        if current.term != identity.term:
            raise LeaderRuntimeUnavailable(
                f"node {identity.leader_id!r} changed term while retaining generation "
                f"{identity.generation}"
            )
        return current
