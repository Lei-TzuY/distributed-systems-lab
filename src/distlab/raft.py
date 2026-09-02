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

    def __post_init__(self) -> None:
        if self.term < 0:
            raise ValueError("term must be non-negative")
        if self.prev_log_index < 0:
            raise ValueError("prev_log_index must be non-negative")
        if self.prev_log_term < 0:
            raise ValueError("prev_log_term must be non-negative")
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


class ElectionSafetyViolation(AssertionError):
    pass


class LogMatchingViolation(AssertionError):
    pass


class RaftCluster:
    """Raft election and follower-side log-matching layer over the simulator.

    The current layer models persistent term/vote/log state, RequestVote, and
    AppendEntries consistency checks plus conflict repair. Leader next-index,
    commit advancement, and state-machine application remain later milestones.
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

    def assert_log_matching(self) -> None:
        """Assert Raft Log Matching over all currently persisted node logs.

        If two logs contain an entry with the same index and term, their prefix
        through that index must be identical. This executable invariant catches
        any conflict-repair bug that would leave divergent histories sharing the
        same Raft index/term identity.
        """

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
    def __init__(self, *, cluster: RaftCluster, node_id: str, peers: tuple[str, ...]) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self.node_id = node_id
        self.peers = peers
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
    ) -> None:
        """Send one explicit AppendEntries probe from a live leader.

        This deliberately does not implement next-index tracking yet. Tests and
        later leader replication code can drive exact probes while follower-side
        Raft consistency and repair semantics are established first.
        """

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
        prev_log_term = self.log[prev_log_index - 1].term if prev_log_index else 0
        self.sim.send(
            self.node_id,
            peer,
            AppendEntries(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=entries,
            ),
        )

    def handle_message(self, sim: Simulator, message: Message) -> None:
        self._reset_volatile_defaults()
        payload = message.payload
        if isinstance(payload, RequestVote):
            self._handle_request_vote(message.src, payload)
        elif isinstance(payload, RequestVoteResponse):
            self._handle_request_vote_response(payload)
        elif isinstance(payload, AppendEntries):
            self._handle_append_entries(message.src, payload)
        elif isinstance(payload, AppendEntriesResponse):
            self._handle_append_entries_response(payload)
        else:
            raise TypeError(f"unsupported Raft message {type(payload).__name__}")

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

    def _handle_append_entries(self, src: str, request: AppendEntries) -> None:
        if request.term > self.current_term:
            self._advance_term(request.term)

        success = False
        match_index = 0
        if request.term == self.current_term:
            self.sim.volatile_state[self.node_id]["role"] = RaftRole.FOLLOWER.value
            self.sim.volatile_state[self.node_id]["votes_received"] = set()
            if self._prefix_matches(request.prev_log_index, request.prev_log_term):
                self._merge_entries(request.prev_log_index, request.entries)
                success = True
                match_index = request.prev_log_index + len(request.entries)
                self.cluster.assert_log_matching()

        self.sim._record(
            "raft-append-entries",
            follower=self.node_id,
            leader=request.leader_id,
            term=request.term,
            prev_log_index=request.prev_log_index,
            prev_log_term=request.prev_log_term,
            entry_count=len(request.entries),
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
        if prev_log_index == 0:
            return prev_log_term == 0
        if prev_log_index > self.last_log_index:
            return False
        return self.log[prev_log_index - 1].term == prev_log_term

    def _merge_entries(self, prev_log_index: int, entries: tuple[LogEntry, ...]) -> None:
        if not entries:
            return
        log = list(self.log)
        insert_at = prev_log_index
        incoming_offset = 0
        while incoming_offset < len(entries) and insert_at < len(log):
            existing = log[insert_at]
            incoming = entries[incoming_offset]
            if existing.term != incoming.term:
                del log[insert_at:]
                break
            if existing != incoming:
                raise LogMatchingViolation(
                    "same index/term identifies different entries at "
                    f"index {insert_at + 1}, term {existing.term}"
                )
            insert_at += 1
            incoming_offset += 1

        if incoming_offset < len(entries):
            log.extend(entries[incoming_offset:])
        self._persist_log(tuple(log))

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

    def _persist_log(self, log: tuple[LogEntry, ...]) -> None:
        self.sim.persistent_state[self.node_id]["log"] = log
        self.sim._record("raft-persist-log", node=self.node_id, log=log)

    def _become_leader(self, term: int) -> None:
        if term != self.current_term or self.role is not RaftRole.CANDIDATE:
            return
        self.cluster.record_leader(term, self.node_id)
        self.sim.volatile_state[self.node_id]["role"] = RaftRole.LEADER.value
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
