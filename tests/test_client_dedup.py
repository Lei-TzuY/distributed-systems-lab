import pytest

from distlab.kv import ClientRequest, ClientRequestConflict, Delete, Put, ReplicatedKV
from distlab.raft import LogEntry, RaftCluster
from distlab.simulator import Simulator


def _single_node_kv(log: tuple[LogEntry, ...]) -> tuple[Simulator, RaftCluster, ReplicatedKV]:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = log
    cluster = RaftCluster(sim, ("n1",))
    cluster.node("n1").advance_commit_index(len(log), source="test")
    return sim, cluster, ReplicatedKV(cluster)


def test_duplicate_client_request_executes_operation_once() -> None:
    request = ClientRequest("client-a", 7, Put("counter", "once"))
    sim, _, kv = _single_node_kv(
        (
            LogEntry(term=1, command=request),
            LogEntry(term=1, command=request),
        )
    )

    applied = kv.apply_committed("n1")

    assert len(applied) == 2
    assert kv.snapshot("n1") == {"counter": "once"}
    assert kv.has_applied_request("n1", "client-a", 7) is True
    traces = [record for record in sim.trace if record.kind == "kv-apply"]
    assert [record.details["duplicate"] for record in traces] == [False, True]


def test_same_request_id_is_scoped_by_client() -> None:
    log = (
        LogEntry(term=1, command=ClientRequest("client-a", 1, Put("a", "1"))),
        LogEntry(term=1, command=ClientRequest("client-b", 1, Put("b", "2"))),
    )
    _, _, kv = _single_node_kv(log)

    kv.apply_committed("n1")

    assert kv.snapshot("n1") == {"a": "1", "b": "2"}
    assert kv.has_applied_request("n1", "client-a", 1) is True
    assert kv.has_applied_request("n1", "client-b", 1) is True


def test_duplicate_delete_is_deterministic_noop_after_first_execution() -> None:
    delete = ClientRequest("client-a", 2, Delete("key"))
    log = (
        LogEntry(term=1, command=Put("key", "value")),
        LogEntry(term=1, command=delete),
        LogEntry(term=1, command=delete),
    )
    sim, _, kv = _single_node_kv(log)

    kv.apply_committed("n1")

    assert kv.snapshot("n1") == {}
    traces = [record for record in sim.trace if record.kind == "kv-apply"]
    assert traces[-1].details["duplicate"] is True


def test_conflicting_reuse_is_rejected_before_applied_progress_advances() -> None:
    first = ClientRequest("client-a", 3, Put("mode", "safe"))
    conflicting = ClientRequest("client-a", 3, Put("mode", "fast"))
    sim, _, kv = _single_node_kv(
        (
            LogEntry(term=1, command=first),
            LogEntry(term=1, command=conflicting),
        )
    )

    with pytest.raises(ClientRequestConflict, match="identity reused"):
        kv.apply_committed("n1")

    assert kv.applier.last_applied("n1") == 0
    assert sim.persistent_state["n1"]["state_machine_applied"] == ()
    assert kv.snapshot("n1") == {}


def test_restart_rebuilds_dedup_state_from_durable_applied_history() -> None:
    request = ClientRequest("client-a", 9, Put("survives", "restart"))
    sim, cluster, kv = _single_node_kv((LogEntry(term=1, command=request),))
    kv.apply_committed("n1")

    sim.crash("n1")
    sim.restart("n1")
    recovered = ReplicatedKV(cluster)

    assert recovered.snapshot("n1") == {"survives": "restart"}
    assert recovered.has_applied_request("n1", "client-a", 9) is True

    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1, command=request),
        LogEntry(term=2, command=request),
    )
    cluster.node("n1").advance_commit_index(2, source="test-recovery")
    recovered.apply_committed("n1")

    assert recovered.snapshot("n1") == {"survives": "restart"}
    traces = [record for record in sim.trace if record.kind == "kv-apply"]
    assert traces[-1].details["duplicate"] is True


def test_replica_dedup_state_converges_for_identical_applied_prefix() -> None:
    request = ClientRequest("client-a", 11, Put("x", "1"))
    log = (
        LogEntry(term=1, command=request),
        LogEntry(term=1, command=request),
    )
    sim = Simulator()
    for node_id in ("n1", "n2"):
        sim.persistent_state[node_id]["log"] = log
    cluster = RaftCluster(sim, ("n1", "n2"))
    cluster.node("n1").advance_commit_index(2, source="test")
    cluster.node("n2").advance_commit_index(2, source="test")
    kv = ReplicatedKV(cluster)

    kv.apply_committed("n1")
    kv.apply_committed("n2")

    assert kv.snapshot("n1") == kv.snapshot("n2") == {"x": "1"}
    kv.assert_replica_consistency()


def test_client_request_identity_validation() -> None:
    with pytest.raises(ValueError, match="client_id"):
        ClientRequest("", 1, Put("k", "v"))
    with pytest.raises(ValueError, match="request_id"):
        ClientRequest("client", -1, Put("k", "v"))
    with pytest.raises(TypeError, match="operation"):
        ClientRequest("client", 1, "not-an-operation")  # type: ignore[arg-type]
