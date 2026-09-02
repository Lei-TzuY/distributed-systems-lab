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
    """Deterministically apply committed Raft log entries and assert safety."""

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self._last_applied = {node_id: 0 for node_id in cluster.node_ids}
        self._applied = {node_id: [] for node_id in cluster.node_ids}
        self._canonical_by_index: dict[int, AppliedEntry] = {}

    def last_applied(self, node_id: str) -> int:
        self._require_node(node_id)
        return self._last_applied[node_id]

    def applied_entries(self, node_id: str) -> tuple[LogEntry, ...]:
        self._require_node(node_id)
        return tuple(record.entry for record in self._applied[node_id])

    def apply_committed(self, node_id: str) -> tuple[AppliedEntry, ...]:
        """Apply every newly committed local entry in index order exactly once."""

        self._require_node(node_id)
        node = self.cluster.node(node_id)
        previous = self._last_applied[node_id]
        commit_index = node.commit_index

        if commit_index < previous:
            raise RuntimeError(
                f"commit index regressed below lastApplied for {node_id!r}: "
                f"commit_index={commit_index}, last_applied={previous}"
            )
        if commit_index > node.last_log_index:
            raise AssertionError("commit index cannot exceed the local Raft log")

        applied_now: list[AppliedEntry] = []
        for index in range(previous + 1, commit_index + 1):
            entry = node.log[index - 1]
            record = AppliedEntry(node_id=node_id, index=index, entry=entry)
            canonical = self._canonical_by_index.get(index)
            if canonical is not None and canonical.entry != entry:
                raise StateMachineSafetyViolation(
                    "State Machine Safety violated at "
                    f"index {index}: {canonical.node_id!r} applied {canonical.entry!r}, "
                    f"but {node_id!r} would apply {entry!r}"
                )
            if canonical is None:
                self._canonical_by_index[index] = record

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

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._last_applied:
            raise ValueError(f"unknown Raft node {node_id!r}")
