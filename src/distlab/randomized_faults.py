from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .simulator import FaultAction, FaultPlan, FaultRule


@dataclass(frozen=True, order=True, slots=True)
class FaultOpportunity:
    """One concrete message ordinal that may receive a generated fault."""

    src: str
    dst: str
    ordinal: int

    def __post_init__(self) -> None:
        if not self.src:
            raise ValueError("src must be non-empty")
        if not self.dst:
            raise ValueError("dst must be non-empty")
        if self.ordinal <= 0:
            raise ValueError("ordinal must be positive")


@dataclass(frozen=True, slots=True)
class SeededFaultSchedule:
    """Persistable output of seeded fault generation.

    Replay never consults randomness: callers reconstruct the exact ``FaultPlan``
    encoded in ``rules``.
    """

    seed: int
    rules: tuple[FaultRule, ...]

    def to_fault_plan(self) -> FaultPlan:
        return FaultPlan(self.rules)

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "rules": [
                {
                    "action": rule.action.value,
                    "src": rule.src,
                    "dst": rule.dst,
                    "ordinal": rule.ordinal,
                    "extra_delay": rule.extra_delay,
                }
                for rule in self.rules
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> SeededFaultSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded fault schedule format")
        seed = raw.get("seed")
        rules = raw.get("rules")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(rules, list):
            raise ValueError("rules must be a list")

        decoded: list[FaultRule] = []
        for item in rules:
            if not isinstance(item, dict):
                raise ValueError("each rule must be an object")
            try:
                action = FaultAction(item["action"])
                src = item["src"]
                dst = item["dst"]
                ordinal = item["ordinal"]
                extra_delay = item["extra_delay"]
            except (KeyError, ValueError) as exc:
                raise ValueError("invalid fault rule") from exc
            if not isinstance(src, str) or not isinstance(dst, str):
                raise ValueError("fault rule endpoints must be strings")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise ValueError("fault rule ordinal must be an integer")
            if not isinstance(extra_delay, int) or isinstance(extra_delay, bool):
                raise ValueError("fault rule extra_delay must be an integer")
            decoded.append(
                FaultRule(
                    action=action,
                    src=src,
                    dst=dst,
                    ordinal=ordinal,
                    extra_delay=extra_delay,
                )
            )
        return cls(seed=seed, rules=tuple(decoded))


@dataclass(frozen=True, slots=True)
class SeededFaultGenerator:
    """Compile a seed into an explicit, persistable failure schedule."""

    drop_rate: float = 0.1
    delay_rate: float = 0.1
    duplicate_rate: float = 0.1
    max_extra_delay: int = 3

    def __post_init__(self) -> None:
        rates = (self.drop_rate, self.delay_rate, self.duplicate_rate)
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("fault rates must be between 0 and 1")
        if sum(rates) > 1:
            raise ValueError("fault rates must sum to at most 1")
        if self.max_extra_delay <= 0:
            raise ValueError("max_extra_delay must be positive")

    def compile(
        self,
        seed: int,
        opportunities: tuple[FaultOpportunity, ...],
    ) -> SeededFaultSchedule:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        canonical = tuple(sorted(opportunities))
        if len(set(canonical)) != len(canonical):
            raise ValueError("fault opportunities must be unique")

        rng = random.Random(seed)
        generated: list[FaultRule] = []
        drop_cutoff = self.drop_rate
        delay_cutoff = drop_cutoff + self.delay_rate
        duplicate_cutoff = delay_cutoff + self.duplicate_rate

        for opportunity in canonical:
            sample = rng.random()
            if sample < drop_cutoff:
                generated.append(
                    FaultRule(
                        FaultAction.DROP,
                        src=opportunity.src,
                        dst=opportunity.dst,
                        ordinal=opportunity.ordinal,
                    )
                )
            elif sample < delay_cutoff:
                generated.append(
                    FaultRule(
                        FaultAction.DELAY,
                        src=opportunity.src,
                        dst=opportunity.dst,
                        ordinal=opportunity.ordinal,
                        extra_delay=rng.randint(1, self.max_extra_delay),
                    )
                )
            elif sample < duplicate_cutoff:
                generated.append(
                    FaultRule(
                        FaultAction.DUPLICATE,
                        src=opportunity.src,
                        dst=opportunity.dst,
                        ordinal=opportunity.ordinal,
                        extra_delay=rng.randint(1, self.max_extra_delay),
                    )
                )

        return SeededFaultSchedule(seed=seed, rules=tuple(generated))
