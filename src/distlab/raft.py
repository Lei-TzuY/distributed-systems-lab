from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .simulator import Message, Simulator


class RaftRole(StrEnum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True, slots=True)
class RequestVote:
    term: int
    candidate_id: str


@dataclass(frozen=True, slots=True)
class RequestVoteResponse:
    term: int
    voter_id: str
    vote_granted: bool


class ElectionSafetyViolation(AssertionError):
    pass


class RaftCluster:
    """Minimal Raft election layer over the deterministic simulator.

    This milestone intentionally models only term/vote persistence and RequestVote.
    Log freshness, heartbeats, and log replication are separate later layers.
    """

    def __init__(self, sim: Simulator, node_ids: tuple[str, ...]) -> None:
        if not node_ids:
            raise ValueError("Raft cluster requires at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Raft node ids must be unique")

        self.sim = sim
        self.node_ids = node_ids
        self._leaders_by_term: dict[int, str] = {}
        self.nodes = {
            node_id: RaftNode(
                cluster=self,
                node_id=node_id,
                peers=tuple(peer for peer in node_ids if peer != node_id),
            )
            for node_id in node_ids
        }
        for node_id, node in self.nodes.items():
            sim.register(node_id, node.handle_message)

    def node(self, node_id: str) -> RaftNode:
        return self.nodes[node_id]

    def record_leader(self, term: int, node_id: str) -> None:
        existing = self._leaders_by_term.get(term)
        if existing is not None and existing != node_id:
            raise ElectionSafetyViolation(
                f"Election Safety violated in term {term}: {existing!r} and {node_id!r}"
            )
        self._leaders_by_term[term] = node_id

    @property
    def leaders_by_term(self) -> dict[int, str]:
        return dict(self._leaders_by_term)


class RaftNode:
    def __init__(self, *, cluster: RaftCluster, node_id: str, peers: tuple[str, ...]) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
        persistent = self.sim.persistent_state[node_id]
        persistent.setdefault("current_term", 0)
        persistent.setdefault("voted_for", None)
        self._reset_volatile_defaults()

    @property
    def current_term(self) -> int:
        return int(self.sim.persistent_state[self.node_id].get("current_term", 0))

    @property
    def voted_for(self) -> str | None:
        value = self.sim.persistent_state[self.node_id].get("voted_for")
        return value if isinstance(value, str) else None

    @property
    def role(self) -> RaftRole:
        value = self.sim.volatile_state[self.node_id].get("role", RaftRole.FOLLOWER.value)
        return RaftRole(value)

    @property
    def votes_received(self) -> frozenset[str]:
        votes = self.sim.volatile_state[self.node_id].get("votes_received", set())
        return frozenset(votes)

    def start_election(self) -> None:
        if not self.sim.is_alive(self.node_id):
            raise RuntimeError(f"crashed node {self.node_id!r} cannot start an election")

        term = self.current_term + 1
        self._persist_term_and_vote(term=term, voted_for=self.node_id)
        volatile = self.sim.volatile_state[self.node_id]
        volatile["role"] = RaftRole.CANDIDATE.value
        volatile["votes_received"] = {self.node_id}
        self.sim._record("raft-election-start", node=self.node_id, term=term)

        if self._has_majority(1):
            self._become_leader(term)
            return

        request = RequestVote(term=term, candidate_id=self.node_id)
        for peer in self.peers:
            self.sim.send(self.node_id, peer, request)

    def handle_message(self, sim: Simulator, message: Message) -> None:
        self._reset_volatile_defaults()
        payload = message.payload
        if isinstance(payload, RequestVote):
            self._handle_request_vote(message.src, payload)
        elif isinstance(payload, RequestVoteResponse):
            self._handle_request_vote_response(payload)
        else:
            raise TypeError(f"unsupported Raft message {type(payload).__name__}")

    def _handle_request_vote(self, src: str, request: RequestVote) -> None:
        if request.term > self.current_term:
            self._advance_term(request.term)

        grant = False
        if request.term == self.current_term:
            voted_for = self.voted_for
            if voted_for is None or voted_for == request.candidate_id:
                self._persist_term_and_vote(term=request.term, voted_for=request.candidate_id)
                self.sim.volatile_state[self.node_id]["role"] = RaftRole.FOLLOWER.value
                grant = True

        self.sim._record(
            "raft-vote",
            voter=self.node_id,
            candidate=request.candidate_id,
            term=request.term,
            granted=grant,
        )
        self.sim.send(
            self.node_id,
            src,
            RequestVoteResponse(
                term=self.current_term,
                voter_id=self.node_id,
                vote_granted=grant,
            ),
        )

    def _handle_request_vote_response(self, response: RequestVoteResponse) -> None:
        if response.term > self.current_term:
            self._advance_term(response.term)
            return
        if response.term != self.current_term or self.role is not RaftRole.CANDIDATE:
            return
        if not response.vote_granted:
            return

        votes = self.sim.volatile_state[self.node_id].setdefault("votes_received", set())
        votes.add(response.voter_id)
        if self._has_majority(len(votes)):
            self._become_leader(response.term)

    def _advance_term(self, term: int) -> None:
        if term <= self.current_term:
            return
        self._persist_term_and_vote(term=term, voted_for=None)
        volatile = self.sim.volatile_state[self.node_id]
        volatile["role"] = RaftRole.FOLLOWER.value
        volatile["votes_received"] = set()
        self.sim._record("raft-term-advance", node=self.node_id, term=term)

    def _persist_term_and_vote(self, *, term: int, voted_for: str | None) -> None:
        persistent = self.sim.persistent_state[self.node_id]
        persistent["current_term"] = term
        persistent["voted_for"] = voted_for
        self.sim._record(
            "raft-persist-term-vote",
            node=self.node_id,
            term=term,
            voted_for=voted_for,
        )

    def _become_leader(self, term: int) -> None:
        if term != self.current_term or self.role is not RaftRole.CANDIDATE:
            return
        self.cluster.record_leader(term, self.node_id)
        self.sim.volatile_state[self.node_id]["role"] = RaftRole.LEADER.value
        self.sim._record("raft-leader", node=self.node_id, term=term)

    def _has_majority(self, votes: int) -> bool:
        return votes >= (len(self.cluster.node_ids) // 2 + 1)

    def _reset_volatile_defaults(self) -> None:
        volatile = self.sim.volatile_state[self.node_id]
        volatile.setdefault("role", RaftRole.FOLLOWER.value)
        volatile.setdefault("votes_received", set())
