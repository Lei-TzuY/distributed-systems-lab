from distlab._deletion_minimizer import minimize_indexed_sequence


def test_minimization_is_deterministic_and_one_minimal() -> None:
    current, removed = minimize_indexed_sequence(
        ("a", "b", "c", "d"),
        preserves_failure=lambda items: "b" in items and "d" in items,
    )

    assert current == ((1, "b"), (3, "d"))
    assert removed == (0, 2)

    kept = tuple(item for _, item in current)
    for position in range(len(kept)):
        candidate = kept[:position] + kept[position + 1 :]
        assert not ("b" in candidate and "d" in candidate)
