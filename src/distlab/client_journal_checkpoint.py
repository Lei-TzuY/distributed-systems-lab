"""Strict JSON codec for client-owned observations, never server commit evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .kv import Delete, Put
from .linearizability import Get, OperationHistory
from .multipaxos_state_machine import Append, MultiPaxosRequest
from .replay_codec import canonical_json

if TYPE_CHECKING:
    from .multipaxos_client import ClientJournal


_FIELDS = {"version", "requests", "events", "abandoned", "client_request_ids"}


def _fields(value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("unsupported or missing checkpoint fields")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint identity must be a nonempty string")
    return value


def _number(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("checkpoint number must be a nonnegative integer")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON checkpoint field")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-JSON checkpoint constant {value}")


def _encode_operation(operation: object) -> dict[str, Any]:
    kinds = {Put: "put", Delete: "delete", Get: "get", Append: "append"}
    kind = kinds.get(type(operation))
    if kind is None:
        raise ValueError("unsupported checkpoint operation")
    result = {"type": kind, "key": operation.key}
    if isinstance(operation, Put):
        result["value"] = operation.value
    elif isinstance(operation, Append):
        result["suffix"] = operation.suffix
    return result


def _decode_operation(raw: object) -> Put | Delete | Get | Append:
    if not isinstance(raw, dict):
        raise ValueError("invalid checkpoint operation")
    kind = raw.get("type")
    if kind == "put":
        data = _fields(raw, {"type", "key", "value"})
        return Put(data["key"], data["value"])
    if kind == "append":
        data = _fields(raw, {"type", "key", "suffix"})
        return Append(data["key"], data["suffix"])
    if kind in ("get", "delete"):
        data = _fields(raw, {"type", "key"})
        return Get(data["key"]) if kind == "get" else Delete(data["key"])
    raise ValueError("unsupported checkpoint operation type")


def decode_journal(encoded: str, journal_type: type[ClientJournal]) -> ClientJournal:
    """Build fresh objects through the history API after validating every field."""
    try:
        if not isinstance(encoded, str):
            raise ValueError("checkpoint must be a JSON string")
        raw = _fields(
            json.loads(encoded, object_pairs_hook=_unique_object, parse_constant=_constant), _FIELDS
        )
        if type(raw["version"]) is not int or raw["version"] != 1:
            raise ValueError("unsupported client journal checkpoint version")
        for field in ("requests", "events", "abandoned", "client_request_ids"):
            if not isinstance(raw[field], list):
                raise ValueError("checkpoint collections must be arrays")

        requests: dict[str, MultiPaxosRequest] = {}
        identities: set[tuple[str, int]] = set()
        for item in raw["requests"]:
            data = _fields(item, {"client_id", "request_id", "operation_id", "operation"})
            request = MultiPaxosRequest(
                _text(data["client_id"]),
                _number(data["request_id"]),
                _text(data["operation_id"]),
                _decode_operation(data["operation"]),
            )
            if request.operation_id in requests or request.identity in identities:
                raise ValueError("duplicate or conflicting checkpoint request identity")
            requests[request.operation_id] = request
            identities.add(request.identity)

        attachments: dict[str, int] = {}
        for item in raw["client_request_ids"]:
            data = _fields(item, {"operation_id", "request_id"})
            operation_id = _text(data["operation_id"])
            request_id = _number(data["request_id"])
            request = requests.get(operation_id)
            if (
                operation_id in attachments
                or request is None
                or not isinstance(request.operation, (Put, Delete))
                or request_id != request.request_id
            ):
                raise ValueError("conflicting checkpoint history request identity")
            attachments[operation_id] = request_id

        history = OperationHistory()
        for sequence, item in enumerate(raw["events"]):
            if not isinstance(item, dict) or item.get("type") not in ("invoke", "complete"):
                raise ValueError("invalid checkpoint event type")
            kind = item["type"]
            expected = {"sequence", "type", "operation_id"}
            if kind == "complete":
                expected.add("result")
            event = _fields(item, expected)
            if _number(event["sequence"]) != sequence:
                raise ValueError("invalid checkpoint event order")
            operation_id = _text(event["operation_id"])
            request = requests.get(operation_id)
            if request is None or isinstance(request.operation, Append):
                raise ValueError("checkpoint event lacks a history request")
            if kind == "invoke":
                history.invoke(operation_id, request.client_id, request.operation)
                if operation_id in attachments:
                    history.attach_client_request_id(operation_id, attachments[operation_id])
            else:
                history.respond(operation_id, event["result"])

        expected_invocations = [
            op for op, r in requests.items() if not isinstance(r.operation, Append)
        ]
        if [inv.operation_id for inv in history.invocations()] != expected_invocations:
            raise ValueError("checkpoint requests and invocation order disagree")
        abandoned: set[str] = set()
        for item in raw["abandoned"]:
            operation_id = _text(item)
            if operation_id in abandoned:
                raise ValueError("duplicate checkpoint abandonment")
            history.abandon(operation_id)
            abandoned.add(operation_id)
        return journal_type(history=history, requests=requests)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid client journal checkpoint: {exc}") from exc


def encode_journal(journal: ClientJournal) -> str:
    """Serialize validated observations in canonical order without changing them."""
    if not isinstance(journal.history, OperationHistory) or not isinstance(journal.requests, dict):
        raise ValueError("invalid client journal state")
    requests = []
    for operation_id, request in journal.requests.items():
        if not isinstance(request, MultiPaxosRequest) or operation_id != request.operation_id:
            raise ValueError("invalid client journal request table")
        requests.append(
            {
                "client_id": request.client_id,
                "request_id": request.request_id,
                "operation_id": operation_id,
                "operation": _encode_operation(request.operation),
            }
        )
    events = []
    attachments = []
    for invocation in journal.history.invocations():
        request = journal.requests.get(invocation.operation_id)
        if (
            request is None
            or request.client_id != invocation.client_id
            or request.operation != invocation.operation
        ):
            raise ValueError("client journal history conflicts with request contents")
        events.append(
            {
                "sequence": invocation.sequence,
                "type": "invoke",
                "operation_id": invocation.operation_id,
            }
        )
        request_id = journal.history.client_request_id(invocation.operation_id)
        if request_id is not None:
            attachments.append({"operation_id": invocation.operation_id, "request_id": request_id})
    events.extend(
        {
            "sequence": item.completion.sequence,
            "type": "complete",
            "operation_id": item.operation_id,
            "result": item.completion.result,
        }
        for item in journal.history.completed()
    )
    events.sort(key=lambda event: event["sequence"])
    encoded = canonical_json(
        {
            "version": 1,
            "requests": requests,
            "events": events,
            "abandoned": [inv.operation_id for inv in journal.history.abandoned()],
            "client_request_ids": attachments,
        }
    )
    # Refuse inconsistent in-memory journals too, rather than dropping history or
    # unsupported metadata during export. Import itself never uses server state.
    restored = decode_journal(encoded, type(journal))
    if vars(restored.history) != vars(journal.history) or list(restored.requests.items()) != list(
        journal.requests.items()
    ):
        raise ValueError("client journal state is not representable by checkpoint v1")
    return encoded
