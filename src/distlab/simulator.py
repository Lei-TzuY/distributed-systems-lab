from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FaultAction(StrEnum):
    DELIVER = "deliver"
    DROP = "drop"
    DELAY = "delay"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class FaultRule:
    action: FaultAction
    src: str | None = None
    dst: str | None = None
    ordinal: int | None = None
    extra_delay: int = 0

    def __post_init__(self) -> None:
        if self.extra_delay < 0:
            raise ValueError("extra_delay must be non-negative")
        if self.ordinal is not None and self.ordinal <= 0:
            raise ValueError("ordinal must be positive when specified")

    def matches(self, message: Message) -> bool:
        return (
            (self.src is None or self.src == message.src)
            and (self.dst is None or self.dst == message.dst)
            and (self.ordinal is None or self.ordinal == message.ordinal)
        )


@dataclass(frozen=True, slots=True)
class FaultPlan:
    rules: tuple[FaultRule, ...] = ()

    def action_for(self, message: Message) -> FaultRule:
        for rule in self.rules:
            if rule.matches(message):
                return rule
        return FaultRule(FaultAction.DELIVER)


@dataclass(frozen=True, slots=True)
class Message:
    src: str
    dst: str
    payload: Any
    ordinal: int
    delivery_dst: str | None = None


@dataclass(frozen=True, slots=True)
class TraceRecord:
    time: int
    kind: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScenarioAction:
    kind: str
    src: str | None = None
    dst: str | None = None
    payload: Any = None
    delay: int = 1
    node: str | None = None
    max_events: int | None = None
    left: tuple[str, ...] = ()
    right: tuple[str, ...] = ()

    @classmethod
    def send(cls, src: str, dst: str, payload: Any, *, delay: int = 1) -> ScenarioAction:
        return cls(kind="send", src=src, dst=dst, payload=payload, delay=delay)

    @classmethod
    def crash(cls, node: str) -> ScenarioAction:
        return cls(kind="crash", node=node)

    @classmethod
    def restart(cls, node: str) -> ScenarioAction:
        return cls(kind="restart", node=node)

    @classmethod
    def partition(
        cls, left: tuple[str, ...], right: tuple[str, ...]
    ) -> ScenarioAction:
        return cls(kind="partition", left=left, right=right)

    @classmethod
    def heal_partition(
        cls, left: tuple[str, ...], right: tuple[str, ...]
    ) -> ScenarioAction:
        return cls(kind="heal-partition", left=left, right=right)

    @classmethod
    def run(cls, *, max_events: int | None = None) -> ScenarioAction:
        return cls(kind="run", max_events=max_events)


@dataclass(order=True, slots=True)
class _ScheduledDelivery:
    time: int
    sequence: int
    message: Message = field(compare=False)


Handler = Callable[["Simulator", Message], None]
RestartHandler = Callable[["Simulator"], None]


class Simulator:
    """Deterministic event-driven message simulator.

    Equal-time events are ordered by a monotonically increasing sequence number,
    making execution independent of dictionary/hash iteration order. Faults are
    explicit data rather than random side effects, so a scenario can be replayed
    exactly from the same action list and fault plan.
    """

    def __init__(self, *, fault_plan: FaultPlan | None = None) -> None:
        self.time = 0
        self._sequence = 0
        self._queue: list[_ScheduledDelivery] = []
        self._handlers: dict[str, Handler] = {}
        self._restart_handlers: dict[str, RestartHandler] = {}
        self._alive: dict[str, bool] = defaultdict(lambda: True)
        self._send_ordinals: dict[tuple[str, str], int] = defaultdict(int)
        self._blocked_links: set[tuple[str, str]] = set()
        self.fault_plan = fault_plan or FaultPlan()
        self.trace: list[TraceRecord] = []
        self.persistent_state: dict[str, dict[str, Any]] = defaultdict(dict)
        self.volatile_state: dict[str, dict[str, Any]] = defaultdict(dict)

    def register(
        self,
        node: str,
        handler: Handler,
        *,
        restart_handler: RestartHandler | None = None,
    ) -> None:
        if node in self._handlers:
            raise ValueError(f"handler already registered for node {node!r}")
        self._handlers[node] = handler
        if restart_handler is not None:
            self._restart_handlers[node] = restart_handler
        self._alive[node] = True

    def send(
        self,
        src: str,
        dst: str,
        payload: Any,
        *,
        delay: int = 1,
        delivery_dst: str | None = None,
    ) -> None:
        """Send over logical link ``src`` -> ``dst`` to an optional handler endpoint.

        Fault rules, ordinals, partitions, crash checks, and trace identities stay
        keyed by the logical link. ``delivery_dst`` only selects the registered
        handler that receives a delivered message, allowing protocol subchannels
        to share the same deterministic network semantics.
        """
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if delivery_dst is not None and delivery_dst not in self._handlers:
            raise ValueError(f"unknown delivery endpoint {delivery_dst!r}")
        key = (src, dst)
        self._send_ordinals[key] += 1
        message = Message(
            src=src,
            dst=dst,
            payload=payload,
            ordinal=self._send_ordinals[key],
            delivery_dst=delivery_dst,
        )
        if key in self._blocked_links:
            self._record(
                "send",
                src=src,
                dst=dst,
                ordinal=message.ordinal,
                payload=payload,
                action="partition-drop",
            )
            self._record(
                "partition-drop",
                src=src,
                dst=dst,
                ordinal=message.ordinal,
                payload=payload,
            )
            return

        rule = self.fault_plan.action_for(message)
        self._record(
            "send",
            src=src,
            dst=dst,
            ordinal=message.ordinal,
            payload=payload,
            action=rule.action.value,
        )

        if rule.action is FaultAction.DROP:
            self._record(
                "drop",
                src=src,
                dst=dst,
                ordinal=message.ordinal,
                payload=payload,
            )
            return

        effective_delay = delay + (rule.extra_delay if rule.action is FaultAction.DELAY else 0)
        self._schedule(message, effective_delay)

        if rule.action is FaultAction.DUPLICATE:
            self._schedule(message, effective_delay + rule.extra_delay)
            self._record(
                "duplicate",
                src=src,
                dst=dst,
                ordinal=message.ordinal,
                payload=payload,
                extra_delay=rule.extra_delay,
            )

    def crash(self, node: str) -> None:
        self._alive[node] = False
        self.volatile_state[node].clear()
        self._record("crash", node=node)

    def restart(self, node: str) -> None:
        self._alive[node] = True
        self.volatile_state[node].clear()
        self._record("restart", node=node)
        restart_handler = self._restart_handlers.get(node)
        if restart_handler is not None:
            restart_handler(self)

    def partition(self, left: tuple[str, ...], right: tuple[str, ...]) -> None:
        left_nodes, right_nodes = self._validate_partition_groups(left, right)
        for src in left_nodes:
            for dst in right_nodes:
                self._blocked_links.add((src, dst))
                self._blocked_links.add((dst, src))
        self._record("partition", left=left_nodes, right=right_nodes)

    def heal_partition(self, left: tuple[str, ...], right: tuple[str, ...]) -> None:
        left_nodes, right_nodes = self._validate_partition_groups(left, right)
        for src in left_nodes:
            for dst in right_nodes:
                self._blocked_links.discard((src, dst))
                self._blocked_links.discard((dst, src))
        self._record("heal-partition", left=left_nodes, right=right_nodes)

    def is_alive(self, node: str) -> bool:
        return self._alive[node]

    def run(self, *, max_events: int | None = None) -> int:
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must be non-negative")
        delivered = 0
        while self._queue and (max_events is None or delivered < max_events):
            event = heapq.heappop(self._queue)
            self.time = event.time
            message = event.message
            if not self._alive[message.dst]:
                self._record(
                    "discard-crashed",
                    src=message.src,
                    dst=message.dst,
                    ordinal=message.ordinal,
                    payload=message.payload,
                )
                delivered += 1
                continue

            delivery_dst = message.delivery_dst or message.dst
            handler = self._handlers.get(delivery_dst)
            if handler is None:
                raise KeyError(f"no handler registered for node {delivery_dst!r}")

            self._record(
                "deliver",
                src=message.src,
                dst=message.dst,
                ordinal=message.ordinal,
                payload=message.payload,
            )
            handler(self, message)
            delivered += 1
        return delivered

    def run_scenario(self, actions: tuple[ScenarioAction, ...]) -> list[TraceRecord]:
        for action in actions:
            if action.kind == "send":
                if action.src is None or action.dst is None:
                    raise ValueError("send action requires src and dst")
                self.send(action.src, action.dst, action.payload, delay=action.delay)
            elif action.kind == "crash":
                if action.node is None:
                    raise ValueError("crash action requires node")
                self.crash(action.node)
            elif action.kind == "restart":
                if action.node is None:
                    raise ValueError("restart action requires node")
                self.restart(action.node)
            elif action.kind == "partition":
                self.partition(action.left, action.right)
            elif action.kind == "heal-partition":
                self.heal_partition(action.left, action.right)
            elif action.kind == "run":
                self.run(max_events=action.max_events)
            else:
                raise ValueError(f"unknown scenario action {action.kind!r}")
        self.run()
        return list(self.trace)

    def _validate_partition_groups(
        self, left: tuple[str, ...], right: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not left or not right:
            raise ValueError("partition groups must be non-empty")
        if len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError("partition groups must not contain duplicate nodes")
        if set(left) & set(right):
            raise ValueError("partition groups must be disjoint")
        unknown = (set(left) | set(right)) - set(self._handlers)
        if unknown:
            raise ValueError(f"partition references unknown nodes: {sorted(unknown)!r}")
        return tuple(sorted(left)), tuple(sorted(right))

    def _schedule(self, message: Message, delay: int) -> None:
        self._sequence += 1
        delivery_time = self.time + delay
        heapq.heappush(
            self._queue,
            _ScheduledDelivery(time=delivery_time, sequence=self._sequence, message=message),
        )
        self._record(
            "schedule",
            src=message.src,
            dst=message.dst,
            ordinal=message.ordinal,
            payload=message.payload,
            delivery_time=delivery_time,
            sequence=self._sequence,
        )

    def _record(self, kind: str, **details: Any) -> None:
        self.trace.append(TraceRecord(time=self.time, kind=kind, details=dict(details)))
