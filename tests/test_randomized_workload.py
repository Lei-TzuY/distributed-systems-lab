import pytest

from distlab import (
    ClientOperationKind,
    ClientWorkloadAction,
    SeededClientWorkloadGenerator,
    SeededClientWorkloadSchedule,
)


def _generator() -> SeededClientWorkloadGenerator:
    return SeededClientWorkloadGenerator(
        clients=("client-b", "client-a"),
        nodes=("n3", "n1", "n2"),
        keys=("beta", "alpha"),
        values=("v2", "v1"),
        put_rate=0.5,
        delete_rate=0.25,
    )


def test_same_seed_compiles_to_identical_explicit_actions() -> None:
    first = _generator().compile(1729, 32)
    second = SeededClientWorkloadGenerator(
        clients=("client-a", "client-b"),
        nodes=("n1", "n2", "n3"),
        keys=("alpha", "beta"),
        values=("v1", "v2"),
        put_rate=0.5,
        delete_rate=0.25,
    ).compile(1729, 32)

    assert first == second
    assert first.to_json() == second.to_json()
    assert [action.operation_id for action in first.actions] == [
        f"op-{index:06d}" for index in range(1, 33)
    ]


def test_persisted_schedule_round_trips_without_random_generation() -> None:
    generated = _generator().compile(8086, 24)
    replayed = SeededClientWorkloadSchedule.from_json(generated.to_json())

    assert replayed == generated
    assert replayed.to_json() == generated.to_json()


def test_write_request_ids_are_monotonic_per_client_and_reads_have_none() -> None:
    schedule = _generator().compile(99, 100)
    observed: dict[str, list[int]] = {}

    for action in schedule.actions:
        if action.kind is ClientOperationKind.GET:
            assert action.request_id is None
            assert action.value is None
        else:
            assert action.request_id is not None
            observed.setdefault(action.client_id, []).append(action.request_id)

    for request_ids in observed.values():
        assert request_ids == list(range(1, len(request_ids) + 1))


def test_different_seeds_are_independent_but_reproducible() -> None:
    seed_one = _generator().compile(1, 40)
    seed_two = _generator().compile(2, 40)

    assert seed_one == _generator().compile(1, 40)
    assert seed_two == _generator().compile(2, 40)
    assert seed_one.actions != seed_two.actions


def test_action_validation_rejects_invalid_write_and_read_shapes() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ClientWorkloadAction(
            "op-1", "c", "n", ClientOperationKind.PUT, "k", value="v", request_id=0
        )
    with pytest.raises(ValueError, match="string value"):
        ClientWorkloadAction(
            "op-2", "c", "n", ClientOperationKind.PUT, "k", request_id=1
        )
    with pytest.raises(ValueError, match="must not carry"):
        ClientWorkloadAction(
            "op-3", "c", "n", ClientOperationKind.GET, "k", request_id=1
        )


def test_generator_and_schedule_validation_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sum"):
        SeededClientWorkloadGenerator(("c",), ("n",), ("k",), ("v",), 0.8, 0.3)
    with pytest.raises(ValueError, match="unique"):
        SeededClientWorkloadGenerator(("c", "c"), ("n",), ("k",), ("v",))
    with pytest.raises(ValueError, match="non-negative"):
        _generator().compile(1, -1)
    with pytest.raises(ValueError, match="operation ids"):
        SeededClientWorkloadSchedule(
            1,
            (
                ClientWorkloadAction("op", "c", "n", ClientOperationKind.GET, "k"),
                ClientWorkloadAction("op", "c", "n", ClientOperationKind.GET, "k"),
            ),
        )
    with pytest.raises(ValueError, match="unsupported"):
        SeededClientWorkloadSchedule.from_json('{"version":2,"seed":1,"actions":[]}')
