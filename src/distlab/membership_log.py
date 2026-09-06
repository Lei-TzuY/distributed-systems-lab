from __future__ import annotations

from dataclasses import dataclass

from .membership import (
    MembershipChangeError,
    ReconfigurableRaftCluster,
    VotingConfiguration,
)
from .membership_replication import MembershipAwareLeaderReplicator
from .raft import LogEntry, RaftNode, RaftRole


@dataclass(frozen=True, slots=True)
class JointConsensusCommand:
    """Durable Raft command proposing transition to a new voter set."""

    new_voters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.new_voters:
            raise ValueError("joint consensus requires at least one new voter")
        if len(set(self.new_voters)) != len(self.new_voters):
            raise ValueError("new voter ids must be unique")


@dataclass(frozen=True, slots=True)
class StableConsensusCommand:
    """Durable Raft command finalizing the active joint voter set."""

    voters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.voters:
            raise ValueError("stable consensus requires at least one voter")
        if len(set(self.voters)) != len(self.voters):
            raise ValueError("stable voter ids must be unique")


MembershipCommand = JointConsensusCommand | StableConsensusCommand
_MEMBERSHIP_COMMIT_INDEX = "membership_commit_index"


def recover_voting_configuration(
    cluster: ReconfigurableRaftCluster,
    initial: VotingConfiguration,
) -> VotingConfiguration:
    """Replay only durably committed membership commands after cluster recreation.

    Raft's ordinary ``commit_index`` is volatile, so membership activation stores a
    durable per-replica watermark once the cluster has objectively committed the
    corresponding entry. Recovery chooses the highest such watermark, replays the
    membership commands from a replica that durably contains that prefix, and
    rejects compacted or contradictory histories instead of guessing.
    """

    watermarks = {
        node_id: int(cluster.sim.persistent_state[node_id].get(_MEMBERSHIP_COMMIT_INDEX, 0))
        for node_id in cluster.node_ids
    }
    committed_index = max(watermarks.values(), default=0)
    if committed_index == 0:
        return initial

    recovered: VotingConfiguration | None = None
    source: str | None = None
    for node_id in sorted(cluster.node_ids):
        if watermarks[node_id] != committed_index:
            continue
        node = cluster.node(node_id)
        view = node.log_view
        if view.base_index > 0:
            raise MembershipChangeError(
                "cannot recover membership from compacted log without snapshot membership state"
            )
        if committed_index > view.last_index:
            raise MembershipChangeError(
                f"membership commit watermark {committed_index} exceeds {node_id!r} log"
            )
        candidate = _replay_membership_prefix(
            initial,
            node_ids=frozenset(cluster.node_ids),
            entries=tuple(view.entry_at(index) for index in range(1, committed_index + 1)),
        )
        if recovered is None:
            recovered = candidate
            source = node_id
        elif candidate != recovered:
            raise MembershipChangeError(
                "durable replicas disagree on committed membership configuration"
            )

    if recovered is None or source is None:
        raise MembershipChangeError(
            "membership commit watermark has no recoverable durable replica"
        )

    cluster.sim._record(
        "raft-membership-recovered",
        source=source,
        committed_index=committed_index,
        old_voters=tuple(sorted(recovered.old_voters)),
        new_voters=(
            tuple(sorted(recovered.new_voters)) if recovered.new_voters is not None else None
        ),
    )
    return recovered


def _replay_membership_prefix(
    initial: VotingConfiguration,
    *,
    node_ids: frozenset[str],
    entries: tuple[LogEntry, ...],
) -> VotingConfiguration:
    configuration = initial
    for entry in entries:
        command = entry.command
        if isinstance(command, JointConsensusCommand):
            if configuration.is_joint:
                raise MembershipChangeError(
                    "committed membership history starts a second joint configuration"
                )
            new_voters = frozenset(command.new_voters)
            if not new_voters <= node_ids:
                raise MembershipChangeError("committed membership history references unknown nodes")
            configuration = VotingConfiguration(configuration.old_voters, new_voters)
        elif isinstance(command, StableConsensusCommand):
            if configuration.new_voters is None:
                raise MembershipChangeError(
                    "committed membership history finalizes without active joint consensus"
                )
            voters = frozenset(command.voters)
            if voters != configuration.new_voters:
                raise MembershipChangeError(
                    "committed finalization does not match active joint new-voter set"
                )
            configuration = VotingConfiguration(voters)
    return configuration


def _persist_membership_commit_watermark(
    cluster: ReconfigurableRaftCluster,
    *,
    index: int,
    command: MembershipCommand,
) -> None:
    """Persist the cluster-level committed membership index on replicas holding it."""

    for node_id in cluster.node_ids:
        node = cluster.node(node_id)
        if index < node.log_view.first_retained_index or index > node.log_view.last_index:
            continue
        if node.log_view.entry_at(index).command != command:
            continue
        state = cluster.sim.persistent_state[node_id]
        previous = int(state.get(_MEMBERSHIP_COMMIT_INDEX, 0))
        if index > previous:
            state[_MEMBERSHIP_COMMIT_INDEX] = index


class ReplicatedMembershipTransition:
    """Drive one stable-to-joint-to-stable configuration transition through Raft.

    Both configuration changes are appended as ordinary durable current-term log
    entries. Stable-to-joint activation waits for the proposal to commit under the
    old stable quorum. Joint-to-stable finalization waits for its entry to commit
    under the active joint quorum. Committed membership watermarks make those
    decisions reconstructible after crash/restart or cluster recreation.
    """

    def __init__(self, leader: RaftNode) -> None:
        if not isinstance(leader.cluster, ReconfigurableRaftCluster):
            raise TypeError("membership transition requires a reconfigurable Raft cluster")
        self.leader = leader
        self.cluster = leader.cluster
        self.sim = leader.sim
        self._term = leader.current_term
        self._pending_index: int | None = None
        self._pending_command: MembershipCommand | None = None
        self._require_current_leader()

    @property
    def pending_index(self) -> int | None:
        return self._pending_index

    def propose_joint(self, new_voters: tuple[str, ...]) -> int:
        self._require_current_leader()
        self._require_no_pending_command()
        if self.cluster.voting_configuration.is_joint:
            raise MembershipChangeError("joint consensus is already active")

        proposed = frozenset(new_voters)
        if not proposed:
            raise ValueError("new voter configuration cannot be empty")
        if not proposed <= frozenset(self.cluster.node_ids):
            raise ValueError("new voters must be pre-provisioned cluster nodes")

        command = JointConsensusCommand(tuple(sorted(proposed)))
        return self._append_pending(command, trace_kind="raft-membership-proposed")

    def propose_finalize(self) -> int:
        self._require_current_leader()
        self._require_no_pending_command()
        configuration = self.cluster.voting_configuration
        if configuration.new_voters is None:
            raise MembershipChangeError("no joint consensus configuration is active")
        if self.leader.node_id not in configuration.new_voters:
            raise MembershipChangeError("current leader must belong to the new voter configuration")

        command = StableConsensusCommand(tuple(sorted(configuration.new_voters)))
        return self._append_pending(command, trace_kind="raft-membership-finalize-proposed")

    def activate_if_committed(self, replicator: MembershipAwareLeaderReplicator) -> bool:
        index, command = self._committed_pending(replicator)
        if index is None or command is None:
            return False
        if not isinstance(command, JointConsensusCommand):
            raise MembershipChangeError("pending membership command is not a joint proposal")

        _persist_membership_commit_watermark(self.cluster, index=index, command=command)
        self.cluster.begin_joint_consensus(self.leader.node_id, command.new_voters)
        self.sim._record(
            "raft-membership-committed",
            leader=self.leader.node_id,
            term=self._term,
            index=index,
            new_voters=command.new_voters,
        )
        self._clear_pending()
        return True

    def finalize_if_committed(self, replicator: MembershipAwareLeaderReplicator) -> bool:
        index, command = self._committed_pending(replicator)
        if index is None or command is None:
            return False
        if not isinstance(command, StableConsensusCommand):
            raise MembershipChangeError("pending membership command is not a finalization proposal")

        configuration = self.cluster.voting_configuration
        if configuration.new_voters != frozenset(command.voters):
            raise MembershipChangeError("active joint configuration changed before finalization")
        _persist_membership_commit_watermark(self.cluster, index=index, command=command)
        self.cluster.finalize_membership(self.leader.node_id)
        self.sim._record(
            "raft-membership-finalized",
            leader=self.leader.node_id,
            term=self._term,
            index=index,
            voters=command.voters,
        )
        self._clear_pending()
        return True

    def replicate_and_activate(self, *, max_attempts_per_peer: int = 2) -> bool:
        command = self._require_pending_command(JointConsensusCommand)
        return self._replicate_until_applied(
            command,
            self.activate_if_committed,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def replicate_and_finalize(self, *, max_attempts_per_peer: int = 3) -> bool:
        command = self._require_pending_command(StableConsensusCommand)
        return self._replicate_until_applied(
            command,
            self.finalize_if_committed,
            max_attempts_per_peer=max_attempts_per_peer,
        )

    def _append_pending(self, command: MembershipCommand, *, trace_kind: str) -> int:
        previous = self.leader.log
        entry = LogEntry(term=self._term, command=command)
        self.sim.persistent_state[self.leader.node_id]["log"] = (*previous, entry)
        index = self.leader.log_view.last_index
        self._pending_index = index
        self._pending_command = command
        self.sim._record(
            trace_kind,
            leader=self.leader.node_id,
            term=self._term,
            index=index,
            old_voters=tuple(sorted(self.cluster.voting_configuration.old_voters)),
            new_voters=(
                command.new_voters
                if isinstance(command, JointConsensusCommand)
                else command.voters
            ),
        )

        if self.leader.log[: len(previous)] != previous:
            raise AssertionError("Leader Append-Only violated while appending membership proposal")
        if self.leader.log_view.entry_at(index) != entry:
            raise AssertionError("membership proposal was not durably appended")
        return index

    def _committed_pending(
        self, replicator: MembershipAwareLeaderReplicator
    ) -> tuple[int | None, MembershipCommand | None]:
        self._require_current_leader()
        if replicator.leader is not self.leader:
            raise MembershipChangeError("replicator must belong to the transition leader")
        index = self._pending_index
        command = self._pending_command
        if index is None or command is None:
            raise MembershipChangeError("no membership proposal is pending")
        if replicator.commit_index < index:
            return None, None
        if self.leader.log_view.entry_at(index).command != command:
            raise MembershipChangeError("pending membership entry changed before activation")
        return index, command

    def _replicate_until_applied(
        self,
        command: MembershipCommand,
        apply_committed,
        *,
        max_attempts_per_peer: int,
    ) -> bool:
        if max_attempts_per_peer <= 0:
            raise ValueError("max_attempts_per_peer must be positive")
        self._require_current_leader()

        replicator = MembershipAwareLeaderReplicator(self.leader)
        replicator.advance_commit_index()
        if apply_committed(replicator):
            return True

        voters = self.cluster.voting_configuration.voters
        for peer in sorted(voters - {self.leader.node_id}):
            if not self.sim.is_alive(peer):
                continue
            replicator.replicate(peer, max_attempts=max_attempts_per_peer)
            if apply_committed(replicator):
                return True
        return False

    def _require_pending_command(self, expected_type):
        self._require_current_leader()
        command = self._pending_command
        if command is None or self._pending_index is None:
            raise MembershipChangeError("no membership proposal is pending")
        if not isinstance(command, expected_type):
            raise MembershipChangeError("pending membership command has a different phase")
        return command

    def _require_no_pending_command(self) -> None:
        if self._pending_index is not None:
            raise MembershipChangeError("a membership proposal is already pending")

    def _clear_pending(self) -> None:
        self._pending_index = None
        self._pending_command = None

    def _require_current_leader(self) -> None:
        if not self.sim.is_alive(self.leader.node_id):
            raise MembershipChangeError("membership transition requires a live current leader")
        if self.leader.role is not RaftRole.LEADER or self.leader.current_term != self._term:
            raise MembershipChangeError("membership transition leader is no longer current")
