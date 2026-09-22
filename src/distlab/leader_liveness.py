from __future__ import annotations

from dataclasses import dataclass

from .membership import ReconfigurableRaftCluster
from .raft import RaftRole
from .replication import LeaderReplicator, ReplicationError


class LeaderLivenessError(RuntimeError):
    """Base error for deterministic leader-liveness failures."""


class LeaderAuthorityLost(LeaderLivenessError):
    """Raised when the monitored node stops being the same-term leader."""


class LeaderQuorumUnavailable(LeaderLivenessError):
    """Raised after CheckQuorum retires a leader that cannot reach voter quorum."""


class LeaderQuorumMembershipChanged(LeaderLivenessError):
    """Raised when quorum identity changes while a liveness round is in flight."""


@dataclass(frozen=True, slots=True)
class LeaderQuorumEvidence:
    leader: str
    term: int
    acknowledged_peers: tuple[str, ...]
    acknowledged_voters: tuple[str, ...]
    quorum_mode: str
    old_majority: int
    new_majority: int | None


class LeaderQuorumMonitor:
    """Confirm same-term voter activity and retire isolated leaders deterministically.

    The monitor is an explicit bounded control-plane round rather than a wall-clock
    background loop. Callers decide when to run the round; the simulator remains
    finite and replayable. A same-term AppendEntries rejection still counts as
    activity because CheckQuorum is proving reachability/term authority, not log
    convergence.
    """

    def __init__(self, replicator: LeaderReplicator) -> None:
        self.replicator = replicator
        self.leader = replicator.leader
        self.sim = self.leader.sim
        self._term = replicator.term

    def check(self, *, max_attempts_per_peer: int = 1) -> LeaderQuorumEvidence:
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")

        self._require_current_leader()
        cluster = self.leader.cluster
        configuration = (
            cluster.voting_configuration
            if isinstance(cluster, ReconfigurableRaftCluster)
            else None
        )
        active_voters = (
            configuration.voters
            if configuration is not None
            else frozenset(cluster.node_ids)
        )
        if self.leader.node_id not in active_voters:
            self.leader.step_down(reason="check-quorum-leader-not-voter")
            raise LeaderAuthorityLost("current leader is not an active voter")

        acknowledged = {self.leader.node_id}
        acknowledged_peers: list[str] = []

        def require_same_configuration() -> None:
            if configuration is None:
                return
            if cluster.voting_configuration is configuration:
                return
            self.sim._record(
                "raft-check-quorum-membership-changed",
                leader=self.leader.node_id,
                term=self._term,
                acknowledged_voters=tuple(sorted(acknowledged & configuration.voters)),
            )
            if self.leader.role is RaftRole.LEADER and self.leader.current_term == self._term:
                self.leader.step_down(reason="check-quorum-membership-changed")
            raise LeaderQuorumMembershipChanged(
                "voting configuration changed during CheckQuorum round"
            )

        def has_quorum() -> bool:
            require_same_configuration()
            if configuration is not None:
                return configuration.has_quorum(acknowledged)
            return len(acknowledged) >= len(cluster.node_ids) // 2 + 1

        for peer in sorted(active_voters - {self.leader.node_id}):
            if has_quorum():
                break
            for _ in range(max_attempts_per_peer):
                self._require_current_leader()
                require_same_configuration()
                try:
                    active = self.replicator.probe_activity(peer)
                except ReplicationError as exc:
                    self._raise_authority_loss(exc)
                if active:
                    acknowledged.add(peer)
                    acknowledged_peers.append(peer)
                    require_same_configuration()
                    break

        self._require_current_leader()
        require_same_configuration()
        acknowledged_voters = tuple(sorted(acknowledged & active_voters))
        quorum_mode = (
            "joint"
            if configuration is not None and configuration.is_joint
            else "stable"
        )
        old_voters = (
            configuration.old_voters
            if configuration is not None
            else frozenset(cluster.node_ids)
        )
        old_majority = len(old_voters) // 2 + 1
        new_majority = (
            len(configuration.new_voters) // 2 + 1
            if configuration is not None and configuration.new_voters is not None
            else None
        )

        if not has_quorum():
            self.sim._record(
                "raft-check-quorum-failed",
                leader=self.leader.node_id,
                term=self._term,
                acknowledged_peers=tuple(acknowledged_peers),
                acknowledged_voters=acknowledged_voters,
                quorum_mode=quorum_mode,
                old_majority=old_majority,
                new_majority=new_majority,
            )
            self.leader.step_down(reason="check-quorum-failed")
            raise LeaderQuorumUnavailable(
                f"leader {self.leader.node_id!r} could not confirm the active voting quorum"
            )

        evidence = LeaderQuorumEvidence(
            leader=self.leader.node_id,
            term=self._term,
            acknowledged_peers=tuple(acknowledged_peers),
            acknowledged_voters=acknowledged_voters,
            quorum_mode=quorum_mode,
            old_majority=old_majority,
            new_majority=new_majority,
        )
        self.sim._record(
            "raft-check-quorum",
            leader=evidence.leader,
            term=evidence.term,
            acknowledged_peers=evidence.acknowledged_peers,
            acknowledged_voters=evidence.acknowledged_voters,
            quorum_mode=evidence.quorum_mode,
            old_majority=evidence.old_majority,
            new_majority=evidence.new_majority,
        )
        return evidence

    def check_window(self, *, response_timeout: int) -> LeaderQuorumEvidence:
        """Confirm quorum from concurrent probes inside a bounded logical-time window."""
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")

        self._require_current_leader()
        cluster = self.leader.cluster
        configuration = (
            cluster.voting_configuration
            if isinstance(cluster, ReconfigurableRaftCluster)
            else None
        )
        active_voters = (
            configuration.voters
            if configuration is not None
            else frozenset(cluster.node_ids)
        )
        if self.leader.node_id not in active_voters:
            self.leader.step_down(reason="check-quorum-leader-not-voter")
            raise LeaderAuthorityLost("current leader is not an active voter")

        probes = {}
        for peer in sorted(active_voters - {self.leader.node_id}):
            try:
                probes[peer] = self.replicator.begin_activity_probe(peer)
            except ReplicationError as exc:
                self._raise_authority_loss(exc)

        deadline = self.sim.time + response_timeout
        self.sim._record(
            "raft-check-quorum-window",
            leader=self.leader.node_id,
            term=self._term,
            deadline=deadline,
            voter_count=len(active_voters),
        )
        self.sim.run_until_time(deadline)

        try:
            self._require_current_leader()
        except LeaderAuthorityLost as exc:
            self.sim._record(
                "raft-check-quorum-authority-lost",
                leader=self.leader.node_id,
                monitor_term=self._term,
                current_term=self.leader.current_term,
                role=self.leader.role.value,
                reason=str(exc),
            )
            raise

        if configuration is not None and cluster.voting_configuration is not configuration:
            self.sim._record(
                "raft-check-quorum-membership-changed",
                leader=self.leader.node_id,
                term=self._term,
                acknowledged_voters=(self.leader.node_id,),
            )
            self.leader.step_down(reason="check-quorum-membership-changed")
            raise LeaderQuorumMembershipChanged(
                "voting configuration changed during CheckQuorum round"
            )

        acknowledged = {self.leader.node_id}
        acknowledged_peers: list[str] = []
        for peer, probe in probes.items():
            try:
                active = self.replicator.finish_activity_probe(probe)
            except ReplicationError as exc:
                self._raise_authority_loss(exc)
            if active:
                acknowledged.add(peer)
                acknowledged_peers.append(peer)

        if configuration is not None and cluster.voting_configuration is not configuration:
            self.leader.step_down(reason="check-quorum-membership-changed")
            raise LeaderQuorumMembershipChanged(
                "voting configuration changed during CheckQuorum round"
            )

        quorum_mode = (
            "joint"
            if configuration is not None and configuration.is_joint
            else "stable"
        )
        old_voters = (
            configuration.old_voters
            if configuration is not None
            else frozenset(cluster.node_ids)
        )
        old_majority = len(old_voters) // 2 + 1
        new_majority = (
            len(configuration.new_voters) // 2 + 1
            if configuration is not None and configuration.new_voters is not None
            else None
        )
        acknowledged_voters = tuple(sorted(acknowledged & active_voters))
        has_quorum = (
            configuration.has_quorum(acknowledged)
            if configuration is not None
            else len(acknowledged) >= len(cluster.node_ids) // 2 + 1
        )

        if not has_quorum:
            self.sim._record(
                "raft-check-quorum-failed",
                leader=self.leader.node_id,
                term=self._term,
                acknowledged_peers=tuple(acknowledged_peers),
                acknowledged_voters=acknowledged_voters,
                quorum_mode=quorum_mode,
                old_majority=old_majority,
                new_majority=new_majority,
                deadline=deadline,
            )
            self.leader.step_down(reason="check-quorum-failed")
            raise LeaderQuorumUnavailable(
                f"leader {self.leader.node_id!r} could not confirm the active voting quorum"
            )

        evidence = LeaderQuorumEvidence(
            leader=self.leader.node_id,
            term=self._term,
            acknowledged_peers=tuple(acknowledged_peers),
            acknowledged_voters=acknowledged_voters,
            quorum_mode=quorum_mode,
            old_majority=old_majority,
            new_majority=new_majority,
        )
        self.sim._record(
            "raft-check-quorum",
            leader=evidence.leader,
            term=evidence.term,
            acknowledged_peers=evidence.acknowledged_peers,
            acknowledged_voters=evidence.acknowledged_voters,
            quorum_mode=evidence.quorum_mode,
            old_majority=evidence.old_majority,
            new_majority=evidence.new_majority,
            deadline=deadline,
        )
        return evidence

    def _require_current_leader(self) -> None:
        if not self.sim.is_alive(self.leader.node_id):
            raise LeaderAuthorityLost("CheckQuorum requires a live leader")
        if self.leader.role is not RaftRole.LEADER:
            raise LeaderAuthorityLost("CheckQuorum requires leader role")
        if self.leader.current_term != self._term:
            raise LeaderAuthorityLost("CheckQuorum monitor term is stale")
        if self.replicator.term != self._term:
            raise LeaderAuthorityLost("CheckQuorum replicator term is stale")

    def _raise_authority_loss(self, exc: ReplicationError) -> None:
        self.sim._record(
            "raft-check-quorum-authority-lost",
            leader=self.leader.node_id,
            monitor_term=self._term,
            current_term=self.leader.current_term,
            role=self.leader.role.value,
            reason=str(exc),
        )
        raise LeaderAuthorityLost(str(exc)) from exc
