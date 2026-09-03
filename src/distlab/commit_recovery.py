from __future__ import annotations

from typing import Any

from .raft import LogEntry, RaftNode, RaftRole


class CommitRecoveryError(RuntimeError):
    """Raised when a current-term commit recovery barrier cannot be appended."""


def append_current_term_barrier(leader: RaftNode, *, command: Any = None) -> int:
    """Append one durable current-term entry to a live Raft leader.

    A restarted leader intentionally loses its volatile commit index. Raft does not
    allow that leader to infer commitment of entries from older terms merely from
    replica counts. Appending and then majority-replicating a current-term entry
    provides an explicit barrier: once that entry commits, every preceding entry in
    the leader's log is committed as part of the same prefix.

    The helper only appends locally. Callers must use ``LeaderReplicator`` to drive
    deterministic majority replication and commit propagation.
    """
    if not leader.sim.is_alive(leader.node_id):
        raise CommitRecoveryError("commit recovery requires a live leader")
    if leader.role is not RaftRole.LEADER:
        raise CommitRecoveryError("commit recovery requires leader role")

    previous = leader.log
    entry = LogEntry(term=leader.current_term, command=command)
    durable = (*previous, entry)
    leader.sim.persistent_state[leader.node_id]["log"] = durable
    index = len(durable)
    leader.sim._record(
        "raft-current-term-barrier",
        leader=leader.node_id,
        term=leader.current_term,
        index=index,
    )

    if leader.log[: len(previous)] != previous:
        raise AssertionError("Leader Append-Only violated while appending commit barrier")
    if leader.log[index - 1] != entry:
        raise AssertionError("commit recovery barrier was not durably appended")
    return index
