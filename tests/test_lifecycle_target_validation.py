import pytest

from distlab.simulator import ScenarioAction, Simulator


def _simulator() -> Simulator:
    simulator = Simulator()
    simulator.register("n1", lambda _simulator, _message: None)
    return simulator


@pytest.mark.parametrize("operation", ["crash", "restart"])
def test_lifecycle_fault_rejects_unknown_node_without_mutating_state(operation: str) -> None:
    simulator = _simulator()
    before_trace = list(simulator.trace)
    before_alive = dict(simulator._alive)
    before_volatile = dict(simulator.volatile_state)

    with pytest.raises(ValueError, match=f"{operation} references unknown node 'ghost'"):
        getattr(simulator, operation)("ghost")

    assert simulator.trace == before_trace
    assert dict(simulator._alive) == before_alive
    assert dict(simulator.volatile_state) == before_volatile
    assert "ghost" not in simulator._alive
    assert "ghost" not in simulator.volatile_state


@pytest.mark.parametrize(
    "action",
    [ScenarioAction.crash("ghost"), ScenarioAction.restart("ghost")],
)
def test_replayable_lifecycle_action_rejects_unknown_node_without_trace(action: ScenarioAction) -> None:
    simulator = _simulator()

    with pytest.raises(ValueError, match="references unknown node 'ghost'"):
        simulator.run_scenario((action,))

    assert simulator.trace == []
    assert "ghost" not in simulator._alive
    assert "ghost" not in simulator.volatile_state


def test_registered_lifecycle_faults_still_clear_volatile_state_and_replay() -> None:
    actions = (ScenarioAction.crash("n1"), ScenarioAction.restart("n1"))

    first = _simulator()
    first.volatile_state["n1"]["term"] = 7
    first_trace = first.run_scenario(actions)

    second = _simulator()
    second.volatile_state["n1"]["term"] = 7
    second_trace = second.run_scenario(actions)

    assert first_trace == second_trace
    assert first.is_alive("n1")
    assert second.is_alive("n1")
    assert first.volatile_state["n1"] == {}
    assert second.volatile_state["n1"] == {}
