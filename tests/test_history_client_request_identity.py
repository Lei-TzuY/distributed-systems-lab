import pytest

from distlab.kv import Delete, Put
from distlab.linearizability import Get, InvalidHistory, OperationHistory


def test_pending_client_request_identity_follows_write_history_lifecycle() -> None:
    history = OperationHistory()
    history.invoke("put-7", "writer", Put("x", "seven"))

    history.attach_client_request_id("put-7", 7)
    assert history.client_request_id("put-7") == 7

    history.respond("put-7")
    assert history.client_request_id("put-7") == 7
    assert history.retire_client_request_id("put-7") == 7
    assert history.client_request_id("put-7") is None


def test_client_request_identity_rejects_invalid_history_surfaces() -> None:
    history = OperationHistory()

    with pytest.raises(InvalidHistory, match="without invocation"):
        history.attach_client_request_id("missing", 1)

    history.invoke("read-1", "reader", Get("x"))
    with pytest.raises(InvalidHistory, match="Put/Delete"):
        history.attach_client_request_id("read-1", 1)

    history.invoke("delete-2", "writer", Delete("x"))
    with pytest.raises(ValueError, match="non-negative"):
        history.attach_client_request_id("delete-2", -1)

    history.attach_client_request_id("delete-2", 2)
    with pytest.raises(InvalidHistory, match="duplicate"):
        history.attach_client_request_id("delete-2", 2)
    with pytest.raises(InvalidHistory, match="before response"):
        history.retire_client_request_id("delete-2")

    history.respond("delete-2")
    history.retire_client_request_id("delete-2")
    with pytest.raises(InvalidHistory, match="after response"):
        history.attach_client_request_id("delete-2", 2)
