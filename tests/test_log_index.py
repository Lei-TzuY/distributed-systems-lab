import pytest

from distlab.log_index import RaftLogView
from distlab.raft import LogEntry, LogMatchingViolation


def test_uncompacted_log_view_preserves_existing_absolute_indexes() -> None:
    entries = (LogEntry(term=1, command="a"), LogEntry(term=2, command="b"))
    log = RaftLogView.uncompacted(entries)

    assert log.base_index == 0
    assert log.base_term == 0
    assert log.first_retained_index == 1
    assert log.last_index == 2
    assert log.last_term == 2
    assert log.term_at(0) == 0
    assert log.entry_at(1) == entries[0]
    assert log.entry_at(2) == entries[1]
    assert log.suffix_from(2) == (entries[1],)
    assert log.suffix_from(3) == ()
    assert log.prefix_matches(2, 2) is True
    assert log.prefix_matches(2, 1) is False


def test_compacted_log_view_addresses_retained_suffix_by_absolute_index() -> None:
    entries = (LogEntry(term=4, command="x"), LogEntry(term=5, command="y"))
    log = RaftLogView(base_index=7, base_term=3, entries=entries)

    assert log.first_retained_index == 8
    assert log.last_index == 9
    assert log.last_term == 5
    assert log.term_at(7) == 3
    assert log.entry_at(8) == entries[0]
    assert log.entry_at(9) == entries[1]
    assert log.suffix_from(8) == entries
    assert log.suffix_from(9) == (entries[1],)
    assert log.suffix_from(10) == ()
    assert log.prefix_matches(7, 3) is True
    assert log.prefix_matches(8, 4) is True
    assert log.prefix_matches(6, 3) is False


def test_compacted_boundary_is_not_exposed_as_a_retained_entry() -> None:
    log = RaftLogView(base_index=5, base_term=2, entries=(LogEntry(term=3),))

    with pytest.raises(IndexError, match="compacted boundary"):
        log.entry_at(5)
    with pytest.raises(IndexError, match="first retained"):
        log.suffix_from(5)
    with pytest.raises(IndexError, match="last index"):
        log.entry_at(7)


def test_log_view_rejects_invalid_boundary_metadata() -> None:
    with pytest.raises(ValueError, match="base index"):
        RaftLogView(base_index=-1, base_term=0, entries=())
    with pytest.raises(ValueError, match="base term"):
        RaftLogView(base_index=1, base_term=-1, entries=())
    with pytest.raises(ValueError, match="index zero"):
        RaftLogView(base_index=0, base_term=1, entries=())


def test_compact_through_preserves_absolute_boundary_and_retained_suffix() -> None:
    log = RaftLogView.uncompacted(
        (
            LogEntry(term=1, command="a"),
            LogEntry(term=2, command="b"),
            LogEntry(term=2, command="c"),
            LogEntry(term=3, command="d"),
        )
    )

    compacted = log.compact_through(2)

    assert compacted.base_index == 2
    assert compacted.base_term == 2
    assert compacted.entries == (
        LogEntry(term=2, command="c"),
        LogEntry(term=3, command="d"),
    )
    assert compacted.last_index == 4
    assert compacted.last_term == 3
    assert compacted.prefix_matches(2, 2) is True
    assert compacted.entry_at(3) == LogEntry(term=2, command="c")


def test_compact_through_can_advance_existing_compacted_boundary() -> None:
    log = RaftLogView(
        base_index=7,
        base_term=3,
        entries=(LogEntry(term=4, command="x"), LogEntry(term=5, command="y")),
    )

    compacted = log.compact_through(8)

    assert compacted == RaftLogView(
        base_index=8,
        base_term=4,
        entries=(LogEntry(term=5, command="y"),),
    )
    assert compacted.last_index == 9


def test_compact_through_rejects_boundary_regression_or_unknown_index() -> None:
    log = RaftLogView(base_index=7, base_term=3, entries=(LogEntry(term=4),))

    with pytest.raises(IndexError, match="precedes current boundary"):
        log.compact_through(6)
    with pytest.raises(IndexError, match="exceeds last index"):
        log.compact_through(9)
    assert log.compact_through(7) is log


def test_merge_after_compacted_boundary_appends_by_absolute_index() -> None:
    retained = (LogEntry(term=4, command="x"), LogEntry(term=5, command="y"))
    log = RaftLogView(base_index=7, base_term=3, entries=retained)
    incoming = (LogEntry(term=6, command="z"),)

    merged = log.merge_after(9, incoming)

    assert merged == (*retained, *incoming)


def test_merge_after_compacted_boundary_truncates_conflicting_retained_suffix() -> None:
    log = RaftLogView(
        base_index=7,
        base_term=3,
        entries=(
            LogEntry(term=4, command="x"),
            LogEntry(term=5, command="stale-y"),
            LogEntry(term=5, command="stale-z"),
        ),
    )
    incoming = (
        LogEntry(term=6, command="new-y"),
        LogEntry(term=6, command="new-z"),
    )

    merged = log.merge_after(8, incoming)

    assert merged == (LogEntry(term=4, command="x"), *incoming)


def test_merge_after_preserves_same_index_term_identity_invariant() -> None:
    log = RaftLogView(
        base_index=7,
        base_term=3,
        entries=(LogEntry(term=4, command="existing"),),
    )

    with pytest.raises(LogMatchingViolation, match="index 8"):
        log.merge_after(7, (LogEntry(term=4, command="different"),))


def test_merge_after_rejects_prefix_before_compacted_boundary() -> None:
    log = RaftLogView(base_index=7, base_term=3, entries=(LogEntry(term=4),))

    with pytest.raises(IndexError, match="outside retained range"):
        log.merge_after(6, (LogEntry(term=4),))
