import pytest

from distlab.log_index import RaftLogView
from distlab.raft import LogEntry


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
