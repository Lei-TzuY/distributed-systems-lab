from __future__ import annotations

from dataclasses import dataclass

from .raft import LogEntry, RaftCluster


class StateMachineSafetyViolation(AssertionError):
    """Raised when two state machines apply different entries at the same index."""


@dataclass(frozen=True, slots=True)
class AppliedEntry:
    node_id: str
    index: int
    entry: LogEntry


class StateMachineApplier:
    """Deterministically apply committed Raft log entries and assert safety.

    Applied progress is persisted alongside each node's Raft durable state. This
    models a durable state-machine boundary: a crash may clear volatile
    ``commitIndex`` but must not cause an already-applied command to execute a
    second time after restart.
    """

    _PERSISTENT_KEY = "state_machine_applied"

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self._last_applied: dict[str, int] = {}
        self._applied: dict[str, list[AppliedEntry]] = {}
        self._canonical_by_index: dict[int, AppliedEntry] = {}

        for node_id in cluster.node_ids:
            durable_entries = self._load_durable_history(node_id)
            records = [
                AppliedEntry(node_id=node_id, index=index, entry=entry)
                for index, entry in enumerate(durable_entries, start=1)
            ]
            self._applied[node_id] = records
            self._last_applied[node_id] = len(records)
            for record in records:
                self._record_canonical(record)

    def last_applied(self, node_id: str) -> int:
        self._require_node(node_id)
        return self._last_applied[node_id]

    def applied_entries(self, node_id: str) -> tuple[LogEntry, ...]:
        self._require_node(node_id)
        return tuple(record.entry for record in self._applied[node_id])

    def apply_committed(self, node_id: str) -> tuple[AppliedEntry, ...]:
        """Apply every newly committed local entry in index order exactly once.

        ``commitIndex`` is volatile in Raft. Immediately after restart it may be
        lower than the durable applied prefix until leadership communication
        re-establishes the commit point. In that state there is simply no new
        work to apply; durable application history must never be rolled back.
        """

        self._require_node(node_id)
        node = self.cluster.node(node_id)
        previous = self._last_applied[node_id]
        commit_index = node.commit_index

        if commit_index <= previous:
            return ()
        if commit_index > node.last_log_index:
            raise AssertionError("commit index cannot exceed the local Raft log")

        applied_now: list[AppliedEntry] = []
        for index in range(previous + 1, commit_index + 1):
            entry = node.log[index - 1]
            record = AppliedEntry(node_id=node_id, index=index, entry=entry)
            self._record_canonical(record)

            durable = tuple(self.sim.persistent_state[node_id][self._PERSISTENT_KEY])
            self.sim.persistent_state[node_id][self._PERSISTENT_KEY] = (*durable, entry)
            self._applied[node_id].append(record)
            self._last_applied[node_id] = index
            applied_now.append(record)
            self.sim._record(
                "raft-state-machine-apply",
                node=node_id,
                index=index,
                term=entry.term,
                command=entry.command,
            )

        return tuple(applied_now)

    def assert_state_machine_safety(self) -> None:
        """Re-check every applied history against State Machine Safety."""

        seen: dict[int, AppliedEntry] = {}
        for node_id in self.cluster.node_ids:
            expected_index = 1
            for record in self._applied[node_id]:
                if record.index != expected_index:
                    raise AssertionError(
                        f"non-contiguous applied history for {node_id!r}: "
                        f"expected index {expected_index}, got {record.index}"
                    )
                canonical = seen.get(record.index)
                if canonical is not None and canonical.entry != record.entry:
                    raise StateMachineSafetyViolation(
                        "State Machine Safety violated at "
                        f"index {record.index}: {canonical.node_id!r} applied "
                        f"{canonical.entry!r}, but {node_id!r} applied {record.entry!r}"
                    )
                seen.setdefault(record.index, record)
                expected_index += 1

    def _load_durable_history(self, node_id: str) -> tuple[LogEntry, ...]:
        persistent = self.sim.persistent_state[node_id]
        value = persistent.setdefault(self._PERSISTENT_KEY, ())
        if not isinstance(value, tuple) or not all(isinstance(entry, LogEntry) for entry in value):
            raise TypeError("durable state-machine history must be a tuple of LogEntry values")

        log = self.cluster.node(node_id).log
        if len(value) > len(log):
            raise StateMachineSafetyViolation(
                f"durable applied history for {node_id!r} exceeds the persistent Raft log"
            )
        if log[: len(value)] != value:
            raise StateMachineSafetyViolation(
                f"durable applied history for {node_id!r} diverges from the persistent Raft log"
            )
        return value

    def _record_canonical(self, record: AppliedEntry) -> None:
        canonical = self._canonical_by_index.get(record.index)
        if canonical is not None and canonical.entry != record.entry:
            raise StateMachineSafetyViolation(
                "State Machine Safety violated at "
                f"index {record.index}: {canonical.node_id!r} applied {canonical.entry!r}, "
                f"but {record.node_id!r} would apply {record.entry!r}"
            )
        self._canonical_by_index.setdefault(record.index, record)

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._last_applied:
            raise ValueError(f"unknown Raft node {node_id!r}")
