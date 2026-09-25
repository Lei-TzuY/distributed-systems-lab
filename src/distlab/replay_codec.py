from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .simulator import TraceRecord


def encode_trace(trace: tuple[TraceRecord, ...]) -> str:
    payload = [
        {
            "time": record.time,
            "kind": record.kind,
            "details": json_value(record.details),
        }
        for record in trace
    ]
    return canonical_json(payload)


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{
                field.name: json_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot encode replay trace value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
