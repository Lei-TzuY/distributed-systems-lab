import pytest

from distlab import (
    ClientRequest,
    KVClientHistory,
    LogEntry,
    Put,
    RaftCluster,
    ReplicatedKV,
    Simulator,
    SingleKeyKVLinearizabilityChecker,
)


def _kv_with_logs(
    logs: dict[str, tuple[LogEntry, ...]],
) -> tuple[Simulator, RaftCluster, ReplicatedKV]:
    sim = Simulator()
    for node_id, log in logs.items():
        sim.persistent_state[node_id]["log"] = log
    cluster = RaftCluster(sim, tuple(logs))
    return sim, cluster, ReplicatedKV(cluster)


def test_completed_write_and_read_are_captured_automatically() -> None:
    request = ClientRequest("writer", 1, Put("x", "one"))
    sim, cluster, kv = _kv_with_logs({"n1": (LogEntry(term=1, command=request),)})
    clients = KVClientHistory(kv)

    submitted = clients.invoke_write("write", "writer", 1, Put("x", "one"))
    assert submitted == request
    cluster.node("n1").advance_commit_index(1, source="test")
    kv.apply_committed("n1")
    clients.complete_write("write", "n1")
    assert clients.read("read", "reader", "n1", "x") == "one"

    result = SingleKeyKVLinearizabilityChecker().check(clients.history)

    assert result.linearizable
    assert result.order == ("write", "read")
    assert [record.kind for record in sim.trace if record.kind.startswith("client-")] == [
        "client-invoke",
        "client-response",
        "client-invoke",
        "client-response",
    ]


def test_retry_reuses_one_logical_history_invocation() -> None:
    request = ClientRequest("writer", 7, Put("x", "once"))
    _, cluster, kv = _kv_with_logs(
        {
            "n1": (
                LogEntry(term=1, command=request),
                LogEntry(term=1, command=request),
            )
        }
    )
    clients = KVClientHistory(kv)

    clients.invoke_write("write", "writer", 7, Put("x", "once"))
    cluster.node("n1").advance_commit_index(2, source="test")
    kv.apply_committed("n1")
    clients.complete_write("write", "n1")

    assert len(clients.history.invocations()) == 1
    assert len(clients.history.completed()) == 1
    assert kv.snapshot("n1") == {"x": "once"}


def test_timeout_leaves_write_pending_and_linearizability_checker_can_omit_it() -> None:
    _, _, kv = _kv_with_logs({"n1": ()})
    clients = KVClientHistory(kv)

    clients.invoke_write("pending", "writer", 2, Put("x", "later"))

    assert clients.pending_write("pending") is not None
    assert tuple(item.operation_id for item in clients.history.pending()) == ("pending",)
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable


def test_write_cannot_complete_before_target_replica_applies_request() -> None:
    request = ClientRequest("writer", 3, Put("x", "one"))
    _, _, kv = _kv_with_logs({"n1": (LogEntry(term=1, command=request),)})
    clients = KVClientHistory(kv)
    clients.invoke_write("write", "writer", 3, Put("x", "one"))

    with pytest.raises(RuntimeError, match="before the target replica applied"):
        clients.complete_write("write", "n1")

    assert tuple(item.operation_id for item in clients.history.pending()) == ("write",)


def test_stale_replica_read_is_captured_as_non_linearizable() -> None:
    request = ClientRequest("writer", 4, Put("x", "one"))
    entry = LogEntry(term=1, command=request)
    _, cluster, kv = _kv_with_logs({"n1": (entry,), "n2": (entry,)})
    clients = KVClientHistory(kv)

    clients.invoke_write("write", "writer", 4, Put("x", "one"))
    cluster.node("n1").advance_commit_index(1, source="test")
    kv.apply_committed("n1")
    clients.complete_write("write", "n1")

    assert clients.read("stale-read", "reader", "n2", "x") is None
    result = SingleKeyKVLinearizabilityChecker().check(clients.history)

    assert not result.linearizable
    assert result.order == ()
