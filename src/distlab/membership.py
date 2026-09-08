from __future__ import annotations

from dataclasses import dataclass

from .raft import (
    AppendEntries,
    AppendEntriesResponse,
    RaftCluster,
    RaftNode,
    RaftRole,
    RequestVote,
    RequestVoteResponse,
)
from .simulator import Simulator


class MembershipChangeError(RuntimeError):
    """Base error for deterministic Raft membership-transition failures."""


class NonVoterElectionError(MembershipChangeError):
    """Raised when a learner or removed node attempts to become a candidate."""


@dataclass(frozen=True, slots=True)
class VotingConfiguration:
    """Stable or joint Raft voter configuration.

    During joint consensus, quorum requires independent majorities of both the old
    and new voter sets. Transport nodes may exist outside either set as learners.
    """

    old_voters: frozenset[str]
    new_voters: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.old_voters:
            raise ValueError("voting configuration requires at least one old voter")
        if self.new_voters is not None and not self.new_voters:
            raise ValueError("joint voting configuration requires at least one new voter")

    @property
    def is_joint(self) -> bool:
        return self.new_voters is not None

    @property
    def voters(self) -> frozenset[str]:
        if self.new_voters is None:
            return self.old_voters
        return self.old_voters | self.new_voters

    def has_quorum(self, votes: frozenset[str] | set[str]) -> bool:
        observed = frozenset(votes)
        if not _has_majority(self.old_voters, observed):
            return False
        return self.new_voters is None or _has_majority(self.new_voters, observed)


def _has_majority(voters: frozenset[str], votes: frozenset[str]) -> bool:
    return len(voters & votes) >= (len(voters) // 2 + 1)


class ReconfigurableRaftCluster(RaftCluster):
    """Raft cluster whose elections obey durable stable/joint configurations.

    All ``node_ids`` are pre-provisioned transport participants. ``voters`` selects
    the bootstrap stable voting set; other nodes are learners until a committed
    configuration transition includes them. If persistent state already contains
    committed membership history, construction recovers the active configuration
    before any election timer is armed.
    """

    def __init__(
        self,
        sim: Simulator,
        node_ids: tuple[str, ...],
        *,
        voters: tuple[str, ...] | None = None,
        election_timeouts: dict[str, int] | None = None,
    ) -> None:
        if not node_ids:
            raise ValueError("Raft cluster requires at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Raft node ids must be unique")
        initial_voters = frozenset(node_ids if voters is None else voters)
        if not initial_voters:
            raise ValueError("Raft cluster requires at least one voter")
        if not initial_voters <= frozenset(node_ids):
            raise ValueError("voters must be pre-provisioned cluster nodes")
        if election_timeouts is not None:
            if set(election_timeouts) != set(node_ids):
                raise ValueError("election_timeouts must specify every Raft node exactly once")
            if any(timeout <= 0 for timeout in election_timeouts.values()):
                raise ValueError("election timeouts must be positive")

        self.sim = sim
        self.node_ids = node_ids
        self._leaders_by_term: dict[int, str] = {}
        initial_configuration = VotingConfiguration(initial_voters)
        self._voting_configuration = initial_configuration
        self.nodes = {
            node_id: ReconfigurableRaftNode(
                cluster=self,
                node_id=node_id,
                peers=tuple(peer for peer in node_ids if peer != node_id),
                election_timeout=(
                    election_timeouts[node_id] if election_timeouts is not None else None
                ),
            )
            for node_id in node_ids
        }

        from .membership_log import recover_voting_configuration

        self._voting_configuration = recover_voting_configuration(self, initial_configuration)
        for node_id, node in self.nodes.items():
            sim.register(node_id, node.handle_message, restart_handler=node.handle_restart)
        for node in self.nodes.values():
            node.reset_election_timeout(reason="initial")

    @property
    def voting_configuration(self) -> VotingConfiguration:
        return self._voting_configuration

    def node(self, node_id: str) -> ReconfigurableRaftNode:
        return self.nodes[node_id]

    def begin_joint_consensus(self, leader_id: str, new_voters: tuple[str, ...]) -> None:
        self._require_transition_leader(leader_id)
        if self._voting_configuration.is_joint:
            raise MembershipChangeError("joint consensus is already active")
        proposed = frozenset(new_voters)
        if not proposed:
            raise ValueError("new voter configuration cannot be empty")
        if not proposed <= frozenset(self.node_ids):
            raise ValueError("new voters must be pre-provisioned cluster nodes")
        old = self._voting_configuration.old_voters
        self._install_voting_configuration(
            VotingConfiguration(old, proposed), reason="membership-joint"
        )
        self.sim._record(
            "raft-membership-joint",
            leader=leader_id,
            term=self.node(leader_id).current_term,
            old_voters=tuple(sorted(old)),
            new_voters=tuple(sorted(proposed)),
        )

    def finalize_membership(self, leader_id: str) -> None:
        self._require_transition_leader(leader_id)
        configuration = self._voting_configuration
        if configuration.new_voters is None:
            raise MembershipChangeError("no joint consensus configuration is active")
        if leader_id not in configuration.new_voters:
            raise MembershipChangeError("current leader must belong to the new voter configuration")
        old = configuration.old_voters
        new = configuration.new_voters
        self._install_voting_configuration(VotingConfiguration(new), reason="membership-stable")
        self.sim._record(
            "raft-membership-stable",
            leader=leader_id,
            term=self.node(leader_id).current_term,
            old_voters=tuple(sorted(old)),
            voters=tuple(sorted(new)),
        )

    def is_voter(self, node_id: str) -> bool:
        return node_id in self._voting_configuration.voters

    def has_election_quorum(self, votes: frozenset[str] | set[str]) -> bool:
        return self._voting_configuration.has_quorum(votes)

    def _install_voting_configuration(
        self, configuration: VotingConfiguration, *, reason: str
    ) -> None:
        previous_voters = self._voting_configuration.voters
        self._voting_configuration = configuration
        current_voters = configuration.voters
        for node_id in sorted(previous_voters - current_voters):
            self.node(node_id).disable_election_timeout(reason=f"{reason}-voter-removed")
        for node_id in sorted(current_voters - previous_voters):
            self.node(node_id).reset_election_timeout(reason=f"{reason}-voter-added")

    def _require_transition_leader(self, leader_id: str) -> None:
        if leader_id not in self.nodes:
            raise MembershipChangeError(f"unknown membership-change leader {leader_id!r}")
        leader = self.node(leader_id)
        if not self.sim.is_alive(leader_id) or leader.role is not RaftRole.LEADER:
            raise MembershipChangeError("membership change requires a live current leader")


class ReconfigurableRaftNode(RaftNode):
    cluster: ReconfigurableRaftCluster

    def reset_election_timeout(self, *, reason: str) -> None:
        if not self.cluster.is_voter(self.node_id):
            self.disable_election_timeout(reason=f"{reason}-non-voter")
            return
        super().reset_election_timeout(reason=reason)

    def disable_election_timeout(self, *, reason: str) -> None:
        self._election_timer_generation += 1
        volatile = self.sim.volatile_state[self.node_id]
        if self.role is RaftRole.CANDIDATE:
            volatile["role"] = RaftRole.FOLLOWER.value
            volatile["votes_received"] = set()
            self.sim._record(
                "raft-election-abort",
                node=self.node_id,
                term=self.current_term,
                reason="candidate-not-voter",
            )
        self.sim._record(
            "raft-election-timeout-disabled",
            node=self.node_id,
            generation=self._election_timer_generation,
            reason=reason,
        )

    def start_election(self) -> None:
        if not self.sim.is_alive(self.node_id):
            raise RuntimeError(f"crashed node {self.node_id!r} cannot start an election")
        if not self.cluster.is_voter(self.node_id):
            raise NonVoterElectionError(f"non-voter {self.node_id!r} cannot start an election")
        term = self.current_term + 1
        self._persist_term_and_vote(term=term, voted_for=self.node_id)
        volatile = self.sim.volatile_state[self.node_id]
        volatile["role"] = RaftRole.CANDIDATE.value
        volatile["votes_received"] = {self.node_id}
        self.sim._record(
            "raft-election-start",
            node=self.node_id,
            term=term,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )
        self.reset_election_timeout(reason="election-start")
        if self.cluster.has_election_quorum({self.node_id}):
            self._become_leader(term)
            return
        request = RequestVote(
            term=term,
            candidate_id=self.node_id,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)

    def _handle_request_vote(self, src: str, request: RequestVote) -> None:
        voter_eligible = self.cluster.is_voter(self.node_id)
        candidate_eligible = self.cluster.is_voter(request.candidate_id)
        if candidate_eligible and request.term > self.current_term:
            self._advance_term(request.term)
        log_up_to_date = self._candidate_log_is_up_to_date(request)
        grant = False
        if (
            voter_eligible
            and candidate_eligible
            and request.term == self.current_term
            and log_up_to_date
        ):
            voted_for = self.voted_for
            if voted_for is None or voted_for == request.candidate_id:
                self._persist_term_and_vote(term=request.term, voted_for=request.candidate_id)
                self.sim.volatile_state[self.node_id]["role"] = RaftRole.FOLLOWER.value
                grant = True
                self.reset_election_timeout(reason="vote-granted")
        self.sim._record(
            "raft-vote",
            voter=self.node_id,
            candidate=request.candidate_id,
            term=request.term,
            granted=grant,
            voter_eligible=voter_eligible,
            candidate_eligible=candidate_eligible,
            log_up_to_date=log_up_to_date,
            candidate_last_log_index=request.last_log_index,
            candidate_last_log_term=request.last_log_term,
            voter_last_log_index=self.last_log_index,
            voter_last_log_term=self.last_log_term,
        )
        self.sim.send(
            self.node_id,
            src,
            RequestVoteResponse(term=self.current_term, voter_id=self.node_id, vote_granted=grant),
        )

    def _handle_append_entries(self, src: str, request: AppendEntries) -> None:
        if self.cluster.is_voter(request.leader_id):
            super()._handle_append_entries(src, request)
            return
        self.sim._record(
            "raft-append-entries-rejected",
            follower=self.node_id,
            leader=request.leader_id,
            term=request.term,
            current_term=self.current_term,
            reason="leader-not-voter",
        )
        self.sim.send(
            self.node_id,
            src,
            AppendEntriesResponse(
                term=self.current_term,
                follower_id=self.node_id,
                success=False,
                match_index=0,
            ),
        )

    def _handle_request_vote_response(self, response: RequestVoteResponse) -> None:
        voter_eligible = self.cluster.is_voter(response.voter_id)
        recipient_eligible = self.cluster.is_voter(self.node_id)
        if not voter_eligible or not recipient_eligible:
            self.sim._record(
                "raft-vote-response-rejected",
                node=self.node_id,
                voter=response.voter_id,
                term=response.term,
                current_term=self.current_term,
                reason=("voter-not-voter" if not voter_eligible else "recipient-not-voter"),
            )
            return
        if response.term > self.current_term:
            self._advance_term(response.term)
            return
        if response.term != self.current_term or self.role is not RaftRole.CANDIDATE:
            return
        if not response.vote_granted:
            return
        votes = self.sim.volatile_state[self.node_id].setdefault("votes_received", set())
        votes.add(response.voter_id)
        if self.cluster.has_election_quorum(votes):
            self._become_leader(response.term)

    def _handle_append_entries_response(self, response: AppendEntriesResponse) -> None:
        if response.term > self.current_term:
            follower_eligible = self.cluster.is_voter(response.follower_id)
            recipient_eligible = self.cluster.is_voter(self.node_id)
            if not follower_eligible or not recipient_eligible:
                self.sim._record(
                    "raft-append-response-rejected",
                    node=self.node_id,
                    follower=response.follower_id,
                    term=response.term,
                    current_term=self.current_term,
                    reason=(
                        "follower-not-voter"
                        if not follower_eligible
                        else "recipient-not-voter"
                    ),
                )
                return
        super()._handle_append_entries_response(response)
