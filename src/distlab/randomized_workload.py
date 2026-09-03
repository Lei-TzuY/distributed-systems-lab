from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import StrEnum


class ClientOperationKind(StrEnum):
    PUT = "put"
    DELETE = "delete"
    GET = "get"


@dataclass(frozen=True, slots=True)
class ClientWorkloadAction:
    """One explicit client operation in a replayable workload."""

    operation_id: str
    client_id: str
    node_id: str
    kind: ClientOperationKind
    key: str
    value: str | None = None
    request_id: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("operation_id", self.operation_id),
            ("client_id", self.client_id),
            ("node_id", self.node_id),
            ("key", self.key),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.kind is ClientOperationKind.PUT:
            if not isinstance(self.value, str):
                raise ValueError("put action requires a string value")
            self._validate_request_id()
        elif self.kind is ClientOperationKind.DELETE:
            if self.value is not None:
                raise ValueError("delete action must not carry a value")
            self._validate_request_id()
        elif self.kind is ClientOperationKind.GET:
            if self.value is not None or self.request_id is not None:
                raise ValueError("get action must not carry value or request_id")

    def _validate_request_id(self) -> None:
        if (
            not isinstance(self.request_id, int)
            or isinstance(self.request_id, bool)
            or self.request_id <= 0
        ):
            raise ValueError("write action requires a positive integer request_id")


@dataclass(frozen=True, slots=True)
class SeededClientWorkloadSchedule:
    seed: int
    actions: tuple[ClientWorkloadAction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        operation_ids = [action.operation_id for action in self.actions]
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("operation ids must be unique")

    def to_json(self) -> str:
        payload = {
            "version": 1,
            "seed": self.seed,
            "actions": [
                {
                    "operation_id": action.operation_id,
                    "client_id": action.client_id,
                    "node_id": action.node_id,
                    "kind": action.kind.value,
                    "key": action.key,
                    "value": action.value,
                    "request_id": action.request_id,
                }
                for action in self.actions
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, encoded: str) -> SeededClientWorkloadSchedule:
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("unsupported seeded client workload format")
        seed = raw.get("seed")
        actions = raw.get("actions")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")

        decoded: list[ClientWorkloadAction] = []
        for item in actions:
            if not isinstance(item, dict):
                raise ValueError("each action must be an object")
            try:
                kind = ClientOperationKind(item["kind"])
                action = ClientWorkloadAction(
                    operation_id=item["operation_id"],
                    client_id=item["client_id"],
                    node_id=item["node_id"],
                    kind=kind,
                    key=item["key"],
                    value=item["value"],
                    request_id=item["request_id"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid client workload action") from exc
            decoded.append(action)
        return cls(seed=seed, actions=tuple(decoded))


@dataclass(frozen=True, slots=True)
class SeededClientWorkloadGenerator:
    clients: tuple[str, ...]
    nodes: tuple[str, ...]
    keys: tuple[str, ...]
    values: tuple[str, ...]
    put_rate: float = 0.5
    delete_rate: float = 0.2

    def __post_init__(self) -> None:
        for name, domain in (
            ("clients", self.clients),
            ("nodes", self.nodes),
            ("keys", self.keys),
            ("values", self.values),
        ):
            if not domain or any(not value for value in domain):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(domain)) != len(domain):
                raise ValueError(f"{name} must be unique")
        if self.put_rate < 0 or self.delete_rate < 0:
            raise ValueError("operation rates must be non-negative")
        if self.put_rate + self.delete_rate > 1:
            raise ValueError("put_rate and delete_rate must sum to at most 1")

    def compile(self, seed: int, operation_count: int) -> SeededClientWorkloadSchedule:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not isinstance(operation_count, int) or isinstance(operation_count, bool):
            raise ValueError("operation_count must be an integer")
        if operation_count < 0:
            raise ValueError("operation_count must be non-negative")

        clients = tuple(sorted(self.clients))
        nodes = tuple(sorted(self.nodes))
        keys = tuple(sorted(self.keys))
        values = tuple(sorted(self.values))
        rng = random.Random(seed)
        next_request_id = {client: 0 for client in clients}
        actions: list[ClientWorkloadAction] = []

        for index in range(1, operation_count + 1):
            client_id = rng.choice(clients)
            node_id = rng.choice(nodes)
            key = rng.choice(keys)
            sample = rng.random()
            operation_id = f"op-{index:06d}"

            if sample < self.put_rate:
                next_request_id[client_id] += 1
                actions.append(
                    ClientWorkloadAction(
                        operation_id=operation_id,
                        client_id=client_id,
                        node_id=node_id,
                        kind=ClientOperationKind.PUT,
                        key=key,
                        value=rng.choice(values),
                        request_id=next_request_id[client_id],
                    )
                )
            elif sample < self.put_rate + self.delete_rate:
                next_request_id[client_id] += 1
                actions.append(
                    ClientWorkloadAction(
                        operation_id=operation_id,
                        client_id=client_id,
                        node_id=node_id,
                        kind=ClientOperationKind.DELETE,
                        key=key,
                        request_id=next_request_id[client_id],
                    )
                )
            else:
                actions.append(
                    ClientWorkloadAction(
                        operation_id=operation_id,
                        client_id=client_id,
                        node_id=node_id,
                        kind=ClientOperationKind.GET,
                        key=key,
                    )
                )

        return SeededClientWorkloadSchedule(seed=seed, actions=tuple(actions))
