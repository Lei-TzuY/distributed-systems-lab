from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .simulator import Message, Simulator


class RaftRole(StrEnum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True, slots=True)
class LogEntry:
    term: int
    command: Any = None

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("log entry term must be non-negative")


@dataclass(frozen=True, slots=True)
class RequestVote:
    term: int
    candidate_id: str
    last_log_index: int = 0
    last_log_term: int = 0

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if self.last_log_index < 0:
            raise ValueError("last_log_index must be non-negative")
        if self.last_log_term < 0:
            raise ValueError("last_log_term must be non-negative")


@dataclass(frozen=True, slots=True)
class RequestVoteResponse:
    term: int
    voter_id: str
    vote_granted: bool


@dataclass(frozen=True, slots=True)
class AppendEntries:
    term: int
    leader_id: str
    prev_log_index: int = 0
    prev_log_term: int = 0
    entries: tuple[LogEntry, ...] = ()
    leader_commit: int = 0

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if self.prev_log_index < 0:
            raise ValueError("prev_log_index must be non-negative")
        if self.prev_log_term < 0:
            raise ValueError("prev_log_term must be non-negative")
        if self.leader_commit < 0:
            raise ValueError("leader_commit must be non-negative")
        if self.prev_log_index == 0 and self.prev_log_term != 0:
            raise ValueError("prev_log_term must be zero when prev_log_index is zero")
        if not all(isinstance(entry, LogEntry) for entry in self.entries):
            raise TypeError("entries must contain only LogEntry values")


@dataclass(frozen=True, slots=True)
class AppendEntriesResponse:
    term: int
    follower_id: str
    success: bool
    match_index: int

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if self.match_index < 0:
            raise ValueError("match_index must be non-negative")


@dataclass(frozen=True, slots=True)
class _ElectionTimeout:
    generation: int


class ElectionSafetyViolation(AssertionError):
    pass


class LogMatchingViolation(AssertionError):
    pass


class RaftCluster:
    """Raft election, log matching, and commit propagation over the simulator."""

    def __init__(
        self,
        sim: Simulator,
        node_ids: tuple[str, ...],
        *,
        election_timeouts: dict[str, int] | None = None,
    ) -> None:
        if not node_ids:
            raise ValueError("Raft cluster requires at least one node")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Raft node ids must be unique")
        if election_timeouts is not None:
            if set(election_timeouts) != set(node_ids):
                raise ValueError("election_timeouts must specify every Raft node exactly once")
            if any(timeout <= 0 for timeout in election_timeouts.values()):
                raise ValueError("election timeouts must be positive")

        self.sim = sim
        self.node_ids = node_ids
        self._leaders_by_term: dict[int, str] = {}
        self.nodes = {
            node_id: RaftNode(
                cluster=self,
                node_id=node_id,
                peers=tuple(peer for peer in node_ids if peer != node_id),
                election_timeout=(
                    election_timeouts[node_id] if election_timeouts is not None else None
                ),
            )
            for node_id in node_ids
        }
        for node_id, node in self.nodes.items():
            sim.register(
                node_id,
                node.handle_message,
                restart_handler=node.handle_restart,
            )
        for node in self.nodes.values():
            node.reset_election_timeout(reason="initial")

    def node(self, node_id: str) -> RaftNode:
        return self.nodes[node_id]

    def record_leader(self, term: int, node_id: str) -> None:
        existing = self._leaders_by_term.get(term)
        if existing is not None and existing != node_id:
            raise ElectionSafetyViolation(
                f"Election Safety violated in term {term}: {existing!r} and {node_id!r}"
            )
        self._leaders_by_term[term] = node_id

    def assert_log_matching(self) -> None:
        nodes = tuple(self.nodes.values())
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 :]:
                shared = min(left.last_log_index, right.last_log_index)
                for index in range(1, shared + 1):
                    left_entry = left.log[index - 1]
                    right_entry = right.log[index - 1]
                    if left_entry.term != right_entry.term:
                        continue
                    if left.log[:index] != right.log[:index]:
                        raise LogMatchingViolation(
                            "Log Matching violated at "
                            f"index {index}, term {left_entry.term}: "
                            f"{left.node_id!r} and {right.node_id!r}"
                        )

    @property
    def leaders_by_term(self) -> dict[int, str]:
        return dict(self._leaders_by_term)


class RaftNode:
    def __init__(
        self,
        *,
        cluster: RaftCluster,
        node_id: str,
        peers: tuple[str, ...],
        election_timeout: int | None,
    ) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
        self._election_timeout = election_timeout
        self._election_timer_generation = 0
        persistent = self.sim.persistent_state[node_id]
        persistent.setdefault("current_term", 0)
        persistent.setdefault("voted_for", None)
        persistent.setdefault("log", ())
        self._validate_persistent_log()
        self._reset_volatile_defaults()

    @property
    def current_term(self) -> int:
        return int(self.sim.persistent_state[self.node_id].get("current_term", 0))

    @property
    def voted_for(self) -> str | None:
        value = self.sim.persistent_state[self.node_id].get("voted_for")
        return value if isinstance(value, str) else None

    @property
    def log(self) -> tuple[LogEntry, ...]:
        value = self.sim.persistent_state[self.node_id].get("log", ())
        return tuple(value)

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    @property
    def role(self) -> RaftRole:
        value = self.sim.volatile_state[self.node_id].get("role", RaftRole.FOLLOWER.value)
        return RaftRole(value)

    @property
    def votes_received(self) -> frozenset[str]:
        votes = self.sim.volatile_state[self.node_id].get("votes_received", set())
        return frozenset(votes)

    @property
    def election_timeout(self) -> int | None:
        return self._election_timeout

    @property
    def commit_index(self) -> int:
        return int(self.sim.volatile_state[self.node_id].get("commit_index", 0))

    def advance_commit_index(self, index: int, *, source: str) -> int:
        """Monotonically advance this node's volatile commit index."""
        if index < 0:
            raise ValueError("commit index must be non-negative")
        if index > self.last_log_index:
            raise ValueError("commit index cannot exceed the local log")
        previous = self.commit_index
        if index <= previous:
            return previous
        self.sim.volatile_state[self.node_id]["commit_index"] = index
        self.sim._record(
            "raft-commit-index",
            node=self.node_id,
            previous_commit_index=previous,
            commit_index=index,
            source=source,
        )
        return index

    def reset_election_timeout(self, *, reason: str) -> None:
        if self._election_timeout is None or self.role is RaftRole.LEADER:
            return
        self._election_timer_generation += 1
        generation = self._election_timer_generation
        timeout = self._election_timeout
        self.sim._record(
            "raft-election-timeout-reset",
            node=self.node_id,
            generation=generation,
            deadline=self.sim.time + timeout,
            reason=reason,
        )
        self.sim._schedule(
            Message(
                src=self.node_id,
                dst=self.node_id,
                payload=_ElectionTimeout(generation),
                ordinal=0,
            ),
            timeout,
        )

    def handle_restart(self, sim: Simulator) -> None:
        """Reconstruct Raft volatile state at the simulator restart boundary."""
        if sim is not self.sim:
            raise ValueError("restart callback invoked by a different simulator")
        self._validate_persistent_log()
        self._election_timer_generation += 1
        self._reset_volatile_defaults()
        self.sim._record(
            "raft-restart",
            node=self.node_id,
            term=self.current_term,
            voted_for=self.voted_for,
            last_log_index=self.last_log_index,
            last_log_term=self.last_log_term,
        )
        self.reset_election_timeout(reason="restart")

    def start_election(self) -> None:
        if not self.sim.is_alive(self.node_id):
            raise RuntimeError(f"crashed node {self.node_id!r} cannot start an election")
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
        if self._has_majority(1):
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

    def send_append_entries(
        self,
        peer: str,
        *,
        prev_log_index: int | None = None,
        entries: tuple[LogEntry, ...] = (),
        leader_commit: int | None = None,
    ) -> None:
        if not self.sim.is_alive(self.node_id):
            raise RuntimeError(f"crashed node {self.node_id!r} cannot send AppendEntries")
        if self.role is not RaftRole.LEADER:
            raise RuntimeError("only the current leader role can send AppendEntries")
        if peer not in self.peers:
            raise ValueError(f"unknown peer {peer!r}")
        if prev_log_index is None:
            prev_log_index = self.last_log_index
        if prev_log_index < 0 or prev_log_index > self.last_log_index:
            raise ValueError("prev_log_index must reference the leader log")
        if leader_commit is None:
            leader_commit = self.commit_index
        if leader_commit < 0 or leader_commit > self.last_log_index:
            raise ValueError("leader_commit must reference the leader log")
        from .log_index import RaftLogView

        prev_log_term = RaftLogView.uncompacted(self.log).term_at(prev_log_index)
        self.sim.send(
            self.node_id,
            peer,
            AppendEntries(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=entries,
                leader_commit=leader_commit,
            ),
        )

    def handle_message(self, sim: Simulator, message: Message) -> None:
        self._reset_volatile_defaults()
        payload = message.payload
        if isinstance(payload, _ElectionTimeout):
            self._handle_election_timeout(payload)
        elif isinstance(payload, RequestVote):
            self._handle_request_vote(message.src, payload)
        elif isinstance(payload, RequestVoteResponse):
            self._handle_request_vote_response(payload)
        elif isinstance(payload, AppendEntries):
            self._handle_append_entries(message.src, payload)
        elif isinstance(payload, AppendEntriesResponse):
            self._handle_append_entries_response(payload)
        else:
            raise TypeError(f"unsupported Raft message {type(payload).__name__}")

    def _handle_election_timeout(self, timeout: _ElectionTimeout) -> None:
        if timeout.generation != self._election_timer_generation:
            self.sim._record(
                "raft-election-timeout-stale",
                node=self.node_id,
                generation=timeout.generation,
                current_generation=self._election_timer_generation,
            )
            return
        if self.role is RaftRole.LEADER:
            return
        self.sim._record("raft-election-timeout", node=self.node_id, generation=timeout.generation)
        self.start_election()

    def _handle_request_vote(self, src: str, request: RequestVote) -> None:
        if request.term > self.current_term:
            self._advance_term(request.term)
        log_up_to_date = self._candidate_log_is_up_to_date(request)
        grant = False
        if request.term == self.current_term and log_up_to_date:
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

    def _handle_append_entries(self, src: str, request: AppendEntries) -> None:
        if request.term > self.current_term:
            self._advance_term(request.term)
        success = False
        match_index = 0
        if request.term == self.current_term:
            volatile = self.sim.volatile_state[self.node_id]
            volatile["role"] = RaftRole.FOLLOWER.value
            volatile["votes_received"] = set()
            self.reset_election_timeout(reason="append-entries")
            if self._prefix_matches(request.prev_log_index, request.prev_log_term):
                self._merge_entries(request.prev_log_index, request.entries)
                success = True
                match_index = request.prev_log_index + len(request.entries)
                if request.leader_commit > self.commit_index:
                    self.advance_commit_index(
                        min(request.leader_commit, match_index), source=request.leader_id
                    )
                self.cluster.assert_log_matching()
        self.sim._record(
            "raft-append-entries",
            follower=self.node_id,
            leader=request.leader_id,
            term=request.term,
            prev_log_index=request.prev_log_index,
            prev_log_term=request.prev_log_term,
            entry_count=len(request.entries),
            leader_commit=request.leader_commit,
            commit_index=self.commit_index,
            success=success,
            match_index=match_index,
        )
        self.sim.send(
            self.node_id,
            src,
            AppendEntriesResponse(
                term=self.current_term,
                follower_id=self.node_id,
                success=success,
                match_index=match_index,
            ),
        )

    def _handle_append_entries_response(self, response: AppendEntriesResponse) -> None:
        if response.term > self.current_term:
            self._advance_term(response.term)
            return
        self.sim._record(
            "raft-append-response",
            leader=self.node_id,
            follower=response.follower_id,
            term=response.term,
            success=response.success,
            match_index=response.match_index,
        )

    def _candidate_log_is_up_to_date(self, request: RequestVote) -> bool:
        if request.last_log_term != self.last_log_term:
            return request.last_log_term > self.last_log_term
        return request.last_log_index >= self.last_log_index

    def _prefix_matches(self, prev_log_index: int, prev_log_term: int) -> bool:
        from .log_index import RaftLogView

        return RaftLogView.uncompacted(self.log).prefix_matches(prev_log_index, prev_log_term)

    def _merge_entries(self, prev_log_index: int, entries: tuple[LogEntry, ...]) -> None:
        if not entries:
            return
        from .log_index import RaftLogView

        log = RaftLogView.uncompacted(self.log).merge_after(prev_log_index, entries)
        self._persist_log(log)

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
            "raft-persist-term-vote", node=self.node_id, term=term, voted_for=voted_for
        )

    def _persist_log(self, log: tuple[LogEntry, ...]) -> None:
        self.sim.persistent_state[self.node_id]["log"] = log
        self.sim._record("raft-persist-log", node=self.node_id, log=log)

    def _become_leader(self, term: int) -> None:
        if term != self.current_term or self.role is not RaftRole.CANDIDATE:
            return
        self.cluster.record_leader(term, self.node_id)
        self.sim.volatile_state[self.node_id]["role"] = RaftRole.LEADER.value
        self._election_timer_generation += 1
        self.sim._record("raft-leader", node=self.node_id, term=term)

    def _has_majority(self, votes: int) -> bool:
        return votes >= (len(self.cluster.node_ids) // 2 + 1)

    def _validate_persistent_log(self) -> None:
        log = self.sim.persistent_state[self.node_id].get("log", ())
        if not isinstance(log, (tuple, list)):
            raise TypeError("persistent Raft log must be a sequence of LogEntry values")
        if not all(isinstance(entry, LogEntry) for entry in log):
            raise TypeError("persistent Raft log must contain only LogEntry values")

    def _reset_volatile_defaults(self) -> None:
        volatile = self.sim.volatile_state[self.node_id]
        volatile.setdefault("role", RaftRole.FOLLOWER.value)
        volatile.setdefault("votes_received", set())
        volatile.setdefault("commit_index", 0)
