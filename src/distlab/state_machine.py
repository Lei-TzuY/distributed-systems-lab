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

    Applied progress is persisted alongside each node's Raft durable state. A
    durable snapshot boundary can replace the already-checkpointed applied
    prefix, while entries after that boundary remain explicit and replayable.
    This lets restart and future InstallSnapshot recovery advance from a real
    checkpoint without fabricating commands that have already been compacted.
    """

    _PERSISTENT_KEY = "state_machine_applied"
    _BASE_INDEX_KEY = "state_machine_base_index"
    _BASE_TERM_KEY = "state_machine_base_term"

    def __init__(self, cluster: RaftCluster) -> None:
        self.cluster = cluster
        self.sim = cluster.sim
        self._base_index: dict[str, int] = {}
        self._base_term: dict[str, int] = {}
        self._last_applied: dict[str, int] = {}
        self._applied: dict[str, list[AppliedEntry]] = {}
        self._canonical_by_index: dict[int, AppliedEntry] = {}

        for node_id in cluster.node_ids:
            base_index, base_term, durable_entries = self._load_durable_state(node_id)
            records = [
                AppliedEntry(node_id=node_id, index=index, entry=entry)
                for index, entry in enumerate(durable_entries, start=base_index + 1)
            ]
            self._base_index[node_id] = base_index
            self._base_term[node_id] = base_term
            self._applied[node_id] = records
            self._last_applied[node_id] = base_index + len(records)
            for record in records:
                self._record_canonical(record)

    def last_applied(self, node_id: str) -> int:
        self._require_node(node_id)
        return self._last_applied[node_id]

    def applied_base_index(self, node_id: str) -> int:
        self._require_node(node_id)
        return self._base_index[node_id]

    def applied_base_term(self, node_id: str) -> int:
        self._require_node(node_id)
        return self._base_term[node_id]

    def applied_entries(self, node_id: str) -> tuple[LogEntry, ...]:
        """Return the durable applied suffix retained after the snapshot boundary."""
        self._require_node(node_id)
        return tuple(record.entry for record in self._applied[node_id])

    def compact_through(self, node_id: str, index: int, term: int) -> None:
        """Discard applied history through an already-durable snapshot boundary."""
        self._require_node(node_id)
        if index < 0 or term < 0:
            raise ValueError("state-machine snapshot boundary must be non-negative")
        base_index = self._base_index[node_id]
        base_term = self._base_term[node_id]
        last_applied = self._last_applied[node_id]
        if index < base_index:
            raise ValueError("state-machine snapshot boundary cannot move backwards")
        if index > last_applied:
            raise ValueError("state-machine snapshot boundary cannot exceed last applied")
        if index == base_index:
            if term != base_term:
                raise StateMachineSafetyViolation(
                    f"state-machine snapshot term diverges at existing boundary for {node_id!r}"
                )
            return

        offset = index - base_index - 1
        boundary = self._applied[node_id][offset]
        if boundary.entry.term != term:
            raise StateMachineSafetyViolation(
                f"state-machine snapshot term diverges from applied entry for {node_id!r}"
            )

        retained = self._applied[node_id][index - base_index :]
        persistent = self.sim.persistent_state[node_id]
        persistent[self._PERSISTENT_KEY] = tuple(record.entry for record in retained)
        persistent[self._BASE_INDEX_KEY] = index
        persistent[self._BASE_TERM_KEY] = term
        self._base_index[node_id] = index
        self._base_term[node_id] = term
        self._applied[node_id] = list(retained)
        self.sim._record(
            "raft-state-machine-compact",
            node=node_id,
            previous_base_index=base_index,
            base_index=index,
            base_term=term,
            last_applied=last_applied,
            retained_count=len(retained),
        )

    def apply_committed(self, node_id: str) -> tuple[AppliedEntry, ...]:
        """Apply every newly committed local entry in index order exactly once."""
        self._require_node(node_id)
        node = self.cluster.node(node_id)
        previous = self._last_applied[node_id]
        commit_index = node.commit_index

        if commit_index <= previous:
            return ()
        if commit_index > node.last_log_index:
            raise AssertionError("commit index cannot exceed the local Raft log")

        log = node.log_view
        if previous < log.base_index:
            raise StateMachineSafetyViolation(
                "durable applied history does not reach the compacted Raft boundary for "
                f"{node_id!r}: last_applied={previous}, log_base_index={log.base_index}"
            )

        applied_now: list[AppliedEntry] = []
        for index in range(previous + 1, commit_index + 1):
            entry = log.entry_at(index)
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
        """Re-check retained durable applied histories against Raft safety."""
        seen: dict[int, AppliedEntry] = {}
        boundaries: dict[int, tuple[str, int]] = {}
        for node_id in self.cluster.node_ids:
            base_index = self._base_index[node_id]
            base_term = self._base_term[node_id]
            if base_index > 0:
                existing_boundary = boundaries.get(base_index)
                if existing_boundary is not None and existing_boundary[1] != base_term:
                    raise StateMachineSafetyViolation(
                        "State Machine Safety violated at snapshot boundary "
                        f"{base_index}: {existing_boundary[0]!r} has term "
                        f"{existing_boundary[1]}, but {node_id!r} has term {base_term}"
                    )
                boundaries.setdefault(base_index, (node_id, base_term))
                canonical = seen.get(base_index)
                if canonical is not None and canonical.entry.term != base_term:
                    raise StateMachineSafetyViolation(
                        f"snapshot boundary for {node_id!r} diverges from applied term at "
                        f"index {base_index}"
                    )

            expected_index = base_index + 1
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
                boundary = boundaries.get(record.index)
                if boundary is not None and boundary[1] != record.entry.term:
                    raise StateMachineSafetyViolation(
                        f"applied entry at index {record.index} diverges from snapshot boundary"
                    )
                expected_index += 1

            durable_base, durable_term, durable = self._load_durable_state(node_id)
            applied = self.applied_entries(node_id)
            if (
                durable_base != base_index
                or durable_term != base_term
                or durable != applied
            ):
                raise StateMachineSafetyViolation(
                    "durable state-machine history changed after application for "
                    f"{node_id!r}"
                )

    def _load_durable_state(self, node_id: str) -> tuple[int, int, tuple[LogEntry, ...]]:
        persistent = self.sim.persistent_state[node_id]
        value = persistent.setdefault(self._PERSISTENT_KEY, ())
        base_index = persistent.setdefault(self._BASE_INDEX_KEY, 0)
        base_term = persistent.setdefault(self._BASE_TERM_KEY, 0)
        if not isinstance(value, tuple) or not all(isinstance(entry, LogEntry) for entry in value):
            raise TypeError("durable state-machine history must be a tuple of LogEntry values")
        if not isinstance(base_index, int) or not isinstance(base_term, int):
            raise TypeError("durable state-machine boundary must contain integer metadata")
        if base_index < 0 or base_term < 0:
            raise ValueError("durable state-machine boundary must be non-negative")
        if base_index == 0 and base_term != 0:
            raise ValueError("durable state-machine base term must be zero at index zero")

        log = self.cluster.node(node_id).log_view
        last_applied = base_index + len(value)
        if base_index > log.base_index:
            raise StateMachineSafetyViolation(
                f"state-machine snapshot boundary for {node_id!r} exceeds compacted Raft boundary"
            )
        if last_applied > log.last_index:
            raise StateMachineSafetyViolation(
                f"durable applied history for {node_id!r} exceeds the persistent Raft log"
            )
        if base_index == log.base_index and base_index > 0 and base_term != log.base_term:
            raise StateMachineSafetyViolation(
                f"durable state-machine boundary for {node_id!r} diverges from Raft boundary"
            )
        if base_index < log.base_index <= last_applied:
            offset = log.base_index - base_index - 1
            if value[offset].term != log.base_term:
                raise StateMachineSafetyViolation(
                    f"durable applied history for {node_id!r} diverges at compacted boundary"
                )

        overlap_start = max(base_index + 1, log.first_retained_index)
        for index in range(overlap_start, last_applied + 1):
            entry = value[index - base_index - 1]
            if log.entry_at(index) != entry:
                raise StateMachineSafetyViolation(
                    f"durable applied history for {node_id!r} diverges from the persistent Raft log"
                )
        return base_index, base_term, value

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
