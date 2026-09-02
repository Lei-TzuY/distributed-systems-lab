from distlab import FaultAction, FaultPlan, FaultRule, ScenarioAction, Simulator


def recording_handler(deliveries: list[tuple[int, str, str, object]]):
    def handle(sim: Simulator, message) -> None:
        deliveries.append((sim.time, message.src, message.dst, message.payload))

    return handle


def test_equal_time_events_preserve_send_order() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator()
    sim.register("b", recording_handler(deliveries))

    sim.send("a", "b", "first", delay=3)
    sim.send("a", "b", "second", delay=3)
    sim.run()

    assert deliveries == [
        (3, "a", "b", "first"),
        (3, "a", "b", "second"),
    ]


def test_fault_plan_drops_exact_message_ordinal() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator(
        fault_plan=FaultPlan(
            (FaultRule(FaultAction.DROP, src="a", dst="b", ordinal=2),)
        )
    )
    sim.register("b", recording_handler(deliveries))

    sim.send("a", "b", 1)
    sim.send("a", "b", 2)
    sim.send("a", "b", 3)
    sim.run()

    assert [entry[3] for entry in deliveries] == [1, 3]
    assert [record.kind for record in sim.trace].count("drop") == 1


def test_delay_and_duplicate_are_deterministic() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DELAY,
                    src="a",
                    dst="b",
                    ordinal=1,
                    extra_delay=4,
                ),
                FaultRule(
                    FaultAction.DUPLICATE,
                    src="a",
                    dst="b",
                    ordinal=2,
                    extra_delay=2,
                ),
            )
        )
    )
    sim.register("b", recording_handler(deliveries))

    sim.send("a", "b", "delayed", delay=1)
    sim.send("a", "b", "duplicated", delay=1)
    sim.run()

    assert deliveries == [
        (1, "a", "b", "duplicated"),
        (3, "a", "b", "duplicated"),
        (5, "a", "b", "delayed"),
    ]


def test_crash_clears_volatile_but_preserves_persistent_state() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["term"] = 7
    sim.volatile_state["n1"]["leader"] = "n2"

    sim.crash("n1")
    assert not sim.is_alive("n1")
    assert sim.persistent_state["n1"] == {"term": 7}
    assert sim.volatile_state["n1"] == {}

    sim.restart("n1")
    assert sim.is_alive("n1")
    assert sim.persistent_state["n1"] == {"term": 7}
    assert sim.volatile_state["n1"] == {}


def test_delivery_to_crashed_node_is_discarded() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator()
    sim.register("b", recording_handler(deliveries))

    sim.send("a", "b", "lost", delay=2)
    sim.crash("b")
    sim.run()

    assert deliveries == []
    assert any(record.kind == "discard-crashed" for record in sim.trace)


def test_same_scenario_and_fault_plan_replay_to_identical_trace() -> None:
    actions = (
        ScenarioAction.send("a", "b", {"term": 1}, delay=2),
        ScenarioAction.send("a", "b", {"term": 2}, delay=2),
    )
    plan = FaultPlan(
        (
            FaultRule(
                FaultAction.DUPLICATE,
                src="a",
                dst="b",
                ordinal=2,
                extra_delay=1,
            ),
        )
    )

    traces = []
    for _ in range(2):
        sim = Simulator(fault_plan=plan)
        sim.register("b", lambda _sim, _message: None)
        traces.append(sim.run_scenario(actions))

    assert traces[0] == traces[1]


def test_handler_can_schedule_followup_without_losing_determinism() -> None:
    deliveries: list[tuple[int, str, str, object]] = []
    sim = Simulator()

    def b_handler(current: Simulator, message) -> None:
        deliveries.append((current.time, message.src, message.dst, message.payload))
        current.send("b", "c", "ack", delay=1)

    sim.register("b", b_handler)
    sim.register("c", recording_handler(deliveries))
    sim.send("a", "b", "request", delay=2)
    sim.run()

    assert deliveries == [
        (2, "a", "b", "request"),
        (3, "b", "c", "ack"),
    ]


def test_invalid_delays_and_unknown_actions_are_rejected() -> None:
    sim = Simulator()

    try:
        sim.send("a", "b", "x", delay=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative message delay must be rejected")

    try:
        sim.run_scenario((ScenarioAction(kind="unknown"),))
    except ValueError as exc:
        assert "unknown scenario action" in str(exc)
    else:
        raise AssertionError("unknown scenario action must be rejected")
