from __future__ import annotations

from dataclasses import dataclass

from .raft import LogEntry, LogMatchingViolation


@dataclass(frozen=True, slots=True)
class RaftLogView:
    """Address a retained Raft log suffix by absolute log index.

    ``base_index``/``base_term`` represent the compacted prefix boundary. The
    retained ``entries`` begin at ``base_index + 1``. Today Raft nodes still
    retain their complete logs, so callers construct this view with a zero
    boundary; the abstraction deliberately supports a non-zero boundary so
    protocol code can stop equating absolute indexes with tuple offsets before
    prefix compaction and InstallSnapshot are introduced.
    """

    base_index: int
    base_term: int
    entries: tuple[LogEntry, ...]

    def __post_init__(self) -> None:
        if self.base_index < 0:
            raise ValueError("log base index must be non-negative")
        if self.base_term < 0:
            raise ValueError("log base term must be non-negative")
        if self.base_index == 0 and self.base_term != 0:
            raise ValueError("log base term must be zero at index zero")
        if not all(isinstance(entry, LogEntry) for entry in self.entries):
            raise TypeError("log entries must contain only LogEntry values")

    @classmethod
    def uncompacted(cls, entries: tuple[LogEntry, ...]) -> RaftLogView:
        return cls(base_index=0, base_term=0, entries=entries)

    @property
    def first_retained_index(self) -> int:
        return self.base_index + 1

    @property
    def last_index(self) -> int:
        return self.base_index + len(self.entries)

    @property
    def last_term(self) -> int:
        return self.entries[-1].term if self.entries else self.base_term

    def term_at(self, index: int) -> int:
        """Return the term at an absolute index, including the compacted boundary."""
        if index == self.base_index:
            return self.base_term
        return self.entry_at(index).term

    def entry_at(self, index: int) -> LogEntry:
        """Return a retained entry by absolute index rather than tuple offset."""
        if index <= self.base_index:
            raise IndexError(
                f"log index {index} is at or before compacted boundary {self.base_index}"
            )
        if index > self.last_index:
            raise IndexError(f"log index {index} exceeds last index {self.last_index}")
        return self.entries[index - self.base_index - 1]

    def suffix_from(self, index: int) -> tuple[LogEntry, ...]:
        """Return retained entries beginning at absolute ``index`` (inclusive)."""
        if index < self.first_retained_index:
            raise IndexError(
                f"log index {index} precedes first retained index {self.first_retained_index}"
            )
        if index > self.last_index + 1:
            raise IndexError(f"log index {index} exceeds append position {self.last_index + 1}")
        return self.entries[index - self.first_retained_index :]

    def compact_through(self, index: int) -> RaftLogView:
        """Discard the retained prefix through ``index`` while preserving its term.

        Compaction is expressed entirely in absolute Raft indexes. The resulting
        boundary carries the term at ``index`` so vote freshness and
        AppendEntries prefix checks can continue to reason about the compacted
        entry without reconstructing discarded commands.
        """
        if index < self.base_index:
            raise IndexError(
                f"compaction index {index} precedes current boundary {self.base_index}"
            )
        if index > self.last_index:
            raise IndexError(f"compaction index {index} exceeds last index {self.last_index}")
        if index == self.base_index:
            return self
        boundary_term = self.term_at(index)
        retained = self.entries[index - self.base_index :]
        return RaftLogView(base_index=index, base_term=boundary_term, entries=retained)

    def prefix_matches(self, index: int, term: int) -> bool:
        """Check an AppendEntries-style absolute index/term boundary."""
        if term < 0:
            raise ValueError("log term must be non-negative")
        if index < self.base_index or index > self.last_index:
            return False
        return self.term_at(index) == term

    def merge_after(
        self,
        prev_index: int,
        incoming: tuple[LogEntry, ...],
    ) -> tuple[LogEntry, ...]:
        """Merge AppendEntries payload after an absolute prefix boundary.

        The returned tuple contains only the retained suffix represented by this
        view. Entries at or before ``base_index`` are compacted and therefore
        never reconstructed here.
        """
        if prev_index < self.base_index or prev_index > self.last_index:
            raise IndexError(
                f"previous log index {prev_index} is outside retained range "
                f"[{self.base_index}, {self.last_index}]"
            )
        if not all(isinstance(entry, LogEntry) for entry in incoming):
            raise TypeError("incoming log entries must contain only LogEntry values")
        if not incoming:
            return self.entries

        retained = list(self.entries)
        insert_at = prev_index - self.base_index
        incoming_offset = 0
        while incoming_offset < len(incoming) and insert_at < len(retained):
            existing = retained[insert_at]
            candidate = incoming[incoming_offset]
            absolute_index = self.base_index + insert_at + 1
            if existing.term != candidate.term:
                del retained[insert_at:]
                break
            if existing != candidate:
                raise LogMatchingViolation(
                    "same index/term identifies different entries at "
                    f"index {absolute_index}, term {existing.term}"
                )
            insert_at += 1
            incoming_offset += 1
        if incoming_offset < len(incoming):
            retained.extend(incoming[incoming_offset:])
        return tuple(retained)
