from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import StrEnum

from .simulator import ScenarioAction


class LinkFaultKind(StrEnum):
    BLOCK = "block"
    HEAL = "heal"
    SET_DELAY = "set-delay"
    CLEAR_DELAY = "clear-delay"


@dataclass(frozen=True, slots=True)
class LinkFaultAction:
    """One directional link transition at a workload boundary."""

    action_id: str
    kind: LinkFaultKind
    src: str
    dst: str
    before_action_index: int
    extra_delay: int = 0

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not self.src or not self.dst:
            raise ValueError("link fault endpoints must be non-empty")
        if self.src == self.dst:
            raise ValueError("link fault endpoints must be distinct")
        if (
            not isinstance(self.before_action_index, int)
            or isinstance(self.before_action_index, bool)
            or self.before_action_index < 0
        ):
            raise ValueError("before_action_index must be a non-negative integer")
        if not isinstance(self.extra_delay, int) or isinstance(self.extra_delay, bool):
            raise ValueError("extra_delay must be an integer")
        if self.kind is LinkFaultKind.SET_DELAY:
            if self.extra_delay <= 0:
                raise ValueError("set-delay requires positive extra_delay")
        elif self.extra_delay != 0:
            raise ValueError("extra_delay is only valid for set-delay")

    def to_scenario_action(self) -> ScenarioAction:
        if self.kind is LinkFaultKind.BLOCK:
            return ScenarioAction.block_link(self.src, self.dst)
        if self.kind is LinkFaultKind.HEAL:
            return ScenarioAction.heal_link(self.src, self.dst)
        if self.kind is LinkFaultKind.SET_DELAY:
            return ScenarioAction.set_link_delay(
                self.src,
                self.dst,
                extra_delay=self.extra_delay,
            )
        return ScenarioAction.clear_link_delay(self.src, self.dst)


@dataclass(frozen=True, slots=True)
class SeededLinkFaultSchedule:
    """Persisted directional network transitions replayed without randomness."""

    seed: int
    actions: tuple[LinkFaultAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("link fault action ids must be unique")
        boundaries = [action.before_action_index for action in self.actions]
        if boundaries != sorted(boundaries):
            raise ValueError("link fault actions must be ordered by workload boundary")

    def actions_before(self, action_index: int) -> tuple[ScenarioAction, ...]:
        if not isinstance(action_index, int) or isinstance(action_index, bool):
            raise ValueError("action_index must be an integer")
        if action_index < 0:
            raise ValueError("action_index must be non-negative")
        return tuple(
            action.to_scenario_action()
            for action in self.actions
            if action.before_action_index == action_index
        )

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": [
                {
                    "action_id": action.action_id,
                    "kind": action.kind.value,
                    "src": action.src,
                    "dst": action.dst,
                    "before_action_index": action.before_action_index,
                    "extra_delay": action.extra_delay,
                }
                for action in self.actions
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> SeededLinkFaultSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded link fault schedule format")
        seed = raw.get("seed")
        actions = raw.get("actions")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")

        decoded: list[LinkFaultAction] = []
        for item in actions:
            if not isinstance(item, dict):
                raise ValueError("each link fault action must be an object")
            try:
                decoded.append(
                    LinkFaultAction(
                        action_id=item["action_id"],
                        kind=LinkFaultKind(item["kind"]),
                        src=item["src"],
                        dst=item["dst"],
                        before_action_index=item["before_action_index"],
                        extra_delay=item["extra_delay"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid link fault action") from exc
        return cls(seed=seed, actions=tuple(decoded))

    @classmethod
    def empty(cls, seed: int) -> SeededLinkFaultSchedule:
        return cls(seed=seed, actions=())


@dataclass(frozen=True, slots=True)
class SeededLinkFaultGenerator:
    """Compile bounded directional link faults into an explicit schedule."""

    nodes: tuple[str, ...]
    block_rate: float = 0.0
    heal_rate: float = 0.0
    delay_rate: float = 0.0
    clear_delay_rate: float = 0.0
    max_extra_delay: int = 3

    def __post_init__(self) -> None:
        if len(self.nodes) < 2 or any(not node for node in self.nodes):
            raise ValueError("nodes must contain at least two non-empty strings")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("nodes must be unique")
        rates = (
            self.block_rate,
            self.heal_rate,
            self.delay_rate,
            self.clear_delay_rate,
        )
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("link fault rates must be between 0 and 1")
        if sum(rates) > 1:
            raise ValueError("link fault rates must sum to at most 1")
        if self.max_extra_delay <= 0:
            raise ValueError("max_extra_delay must be positive")

    def compile(self, seed: int, boundary_count: int) -> SeededLinkFaultSchedule:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(boundary_count, int) or isinstance(boundary_count, bool):
            raise ValueError("boundary_count must be an integer")
        if boundary_count < 0:
            raise ValueError("boundary_count must be non-negative")

        nodes = tuple(sorted(self.nodes))
        links = tuple((src, dst) for src in nodes for dst in nodes if src != dst)
        blocked: set[tuple[str, str]] = set()
        delayed: dict[tuple[str, str], int] = {}
        rng = random.Random(seed)
        actions: list[LinkFaultAction] = []

        for boundary in range(boundary_count + 1):
            sample = rng.random()
            block_cutoff = self.block_rate
            heal_cutoff = block_cutoff + self.heal_rate
            delay_cutoff = heal_cutoff + self.delay_rate
            clear_cutoff = delay_cutoff + self.clear_delay_rate

            kind: LinkFaultKind | None = None
            candidates: tuple[tuple[str, str], ...] = ()
            if sample < block_cutoff:
                kind = LinkFaultKind.BLOCK
                candidates = tuple(link for link in links if link not in blocked)
            elif sample < heal_cutoff:
                kind = LinkFaultKind.HEAL
                candidates = tuple(sorted(blocked))
            elif sample < delay_cutoff:
                kind = LinkFaultKind.SET_DELAY
                candidates = links
            elif sample < clear_cutoff:
                kind = LinkFaultKind.CLEAR_DELAY
                candidates = tuple(sorted(delayed))
            if kind is None or not candidates:
                continue

            src, dst = rng.choice(candidates)
            extra_delay = 0
            if kind is LinkFaultKind.BLOCK:
                blocked.add((src, dst))
            elif kind is LinkFaultKind.HEAL:
                blocked.remove((src, dst))
            elif kind is LinkFaultKind.SET_DELAY:
                extra_delay = rng.randint(1, self.max_extra_delay)
                delayed[(src, dst)] = extra_delay
            else:
                delayed.pop((src, dst), None)

            actions.append(
                LinkFaultAction(
                    action_id=f"link-fault-{len(actions) + 1:06d}",
                    kind=kind,
                    src=src,
                    dst=dst,
                    before_action_index=boundary,
                    extra_delay=extra_delay,
                )
            )

        return SeededLinkFaultSchedule(seed=seed, actions=tuple(actions))
