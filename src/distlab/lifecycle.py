from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import StrEnum


class NodeLifecycleKind(StrEnum):
    CRASH = "crash"
    RESTART = "restart"


@dataclass(frozen=True, slots=True)
class NodeLifecycleAction:
    """One explicit node lifecycle transition at a workload boundary."""

    action_id: str
    node_id: str
    kind: NodeLifecycleKind
    before_action_index: int

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if (
            not isinstance(self.before_action_index, int)
            or isinstance(self.before_action_index, bool)
            or self.before_action_index < 0
        ):
            raise ValueError("before_action_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SeededLifecycleSchedule:
    """Persisted lifecycle transitions consumed without consulting randomness."""

    seed: int
    actions: tuple[NodeLifecycleAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("lifecycle action ids must be unique")
        boundaries = [action.before_action_index for action in self.actions]
        if boundaries != sorted(boundaries):
            raise ValueError("lifecycle actions must be ordered by workload boundary")

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": [
                {
                    "action_id": action.action_id,
                    "node_id": action.node_id,
                    "kind": action.kind.value,
                    "before_action_index": action.before_action_index,
                }
                for action in self.actions
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> SeededLifecycleSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded lifecycle schedule format")
        seed = raw.get("seed")
        actions = raw.get("actions")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")

        decoded: list[NodeLifecycleAction] = []
        for item in actions:
            if not isinstance(item, dict):
                raise ValueError("each lifecycle action must be an object")
            try:
                decoded.append(
                    NodeLifecycleAction(
                        action_id=item["action_id"],
                        node_id=item["node_id"],
                        kind=NodeLifecycleKind(item["kind"]),
                        before_action_index=item["before_action_index"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid lifecycle action") from exc
        return cls(seed=seed, actions=tuple(decoded))

    @classmethod
    def empty(cls, seed: int) -> SeededLifecycleSchedule:
        return cls(seed=seed, actions=())


@dataclass(frozen=True, slots=True)
class SeededLifecycleGenerator:
    """Compile bounded node crash/restart choices into an explicit schedule."""

    nodes: tuple[str, ...]
    crash_rate: float = 0.0
    restart_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.nodes or any(not node for node in self.nodes):
            raise ValueError("nodes must contain non-empty strings")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique")
        if self.crash_rate < 0 or self.restart_rate < 0:
            raise ValueError("lifecycle rates must be non-negative")
        if self.crash_rate + self.restart_rate > 1:
            raise ValueError("crash_rate and restart_rate must sum to at most 1")

    def compile(self, seed: int, boundary_count: int) -> SeededLifecycleSchedule:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(boundary_count, int) or isinstance(boundary_count, bool):
            raise ValueError("boundary_count must be an integer")
        if boundary_count < 0:
            raise ValueError("boundary_count must be non-negative")

        nodes = tuple(sorted(self.nodes))
        alive = set(nodes)
        crashed: set[str] = set()
        rng = random.Random(seed)
        actions: list[NodeLifecycleAction] = []

        for boundary in range(boundary_count + 1):
            sample = rng.random()
            kind: NodeLifecycleKind | None = None
            candidates: tuple[str, ...] = ()
            if sample < self.crash_rate and alive:
                kind = NodeLifecycleKind.CRASH
                candidates = tuple(sorted(alive))
            elif sample < self.crash_rate + self.restart_rate and crashed:
                kind = NodeLifecycleKind.RESTART
                candidates = tuple(sorted(crashed))
            if kind is None:
                continue

            node_id = rng.choice(candidates)
            if kind is NodeLifecycleKind.CRASH:
                alive.remove(node_id)
                crashed.add(node_id)
            else:
                crashed.remove(node_id)
                alive.add(node_id)
            actions.append(
                NodeLifecycleAction(
                    action_id=f"lifecycle-{len(actions) + 1:06d}",
                    node_id=node_id,
                    kind=kind,
                    before_action_index=boundary,
                )
            )

        return SeededLifecycleSchedule(seed=seed, actions=tuple(actions))
