from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def minimize_indexed_sequence(
    entries: tuple[T, ...],
    *,
    preserves_failure: Callable[[tuple[T, ...]], bool],
) -> tuple[tuple[tuple[int, T], ...], tuple[int, ...]]:
    """Return a deterministic 1-minimal subsequence and removed original indices.

    Candidates are considered in current sequence order. After every successful
    deletion the scan restarts from the beginning, matching the public schedule
    minimizers' historical semantics while keeping the reduction policy in one
    place.
    """

    current = list(enumerate(entries))
    removed: list[int] = []

    while True:
        changed = False
        for position in range(len(current)):
            candidate_entries = current[:position] + current[position + 1 :]
            candidate = tuple(item for _, item in candidate_entries)
            if preserves_failure(candidate):
                original_index, _ = current[position]
                removed.append(original_index)
                current = candidate_entries
                changed = True
                break
        if not changed:
            break

    return tuple(current), tuple(sorted(removed))
