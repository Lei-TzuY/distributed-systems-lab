import pytest

from distlab.client_history import KVClientHistory
from distlab.client_session import KVClientSession
from distlab.kv import Put, ReplicatedKV
from distlab.leader_runtime import (
    LeaderRuntimeSupervisor,
    LeaderRuntimeUnavailable,
    StaleLeaderRuntimeGeneration,
)
from distlab.leader_service import LeaderKVService, LeaderWriteQuorumUnavailable
from distlab.leadership_transfer import LeadershipTransfer
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.linearizable_read import ReadQuorumUnavailable
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _service_session() -> tuple[
    Simulator,
    RaftCluster,
    LeaderRuntimeSupervisor,
    ReplicatedKV,
    KVClientHistory,
    LeaderKVService,
    KVClientSession,
]:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    service = LeaderKVService(supervisor, kv, clients)
    session = KVClientSession(clients, "writer")
    return sim, cluster, supervisor, kv, clients, service, session


def test_service_backed_session_executes_program_ordered_write_and_read() -> None:
    _, cluster, supervisor, kv, clients, service, session = _service_session()
    identity = supervisor.runtime_identity("n1")

    written = session.write_via_service(
        service,
        "n1",
        identity.generation,
        "write-1",
        1,
        Put("x", "one"),
    )

    assert written.request.request_id == 1
    assert session.last_completed_request_id == 1
    assert session.pending_write() is None
    assert kv.get("n1", "x") == "one"

    read = session.linearizable_read_via_service(
        service,
        "n1",
        identity.generation,
        "read-1",
        "x",
    )

    assert read.value == "one"
    assert session.pending_read() is None
    assert [item.operation_id for item in clients.history.completed()] == [
        "write-1",
        "read-1",
    ]
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True
    kv.applier.assert_state_machine_safety()
    RaftSafetyHarness(cluster).checkpoint()


def test_stale_generation_rejection_leaves_no_phantom_session_operation() -> None:
    _, cluster, supervisor, _, clients, service, session = _service_session()
    node = cluster.node("n1")
    first = supervisor.runtime_identity("n1")
    assert node.step_down(reason="test-rollover")
    node.start_election()
    cluster.sim.run()
    second = supervisor.runtime_identity("n1")

    with pytest.raises(StaleLeaderRuntimeGeneration, match="stale"):
        session.write_via_service(
            service,
            "n1",
            first.generation,
            "stale-write",
            1,
            Put("x", "stale"),
        )

    assert second.generation > first.generation
    assert session.pending_write() is None
    assert session.last_completed_request_id == -1
    assert clients.history.invocations() == ()


def test_failed_session_write_retries_exactly_after_leadership_transfer() -> None:
    sim, cluster, supervisor, kv, clients, service, session = _service_session()
    first = supervisor.runtime_identity("n1")
    sim.crash("n2")
    sim.crash("n3")

    with pytest.raises(LeaderWriteQuorumUnavailable):
        session.write_via_service(
            service,
            "n1",
            first.generation,
            "write-1",
            1,
            Put("x", "one"),
            max_attempts_per_peer=1,
        )

    pending = session.pending_write()
    assert pending is not None
    assert pending.request.request_id == 1
    assert clients.pending_write("write-1") == pending.request
    assert session.last_completed_request_id == -1

    sim.restart("n2")
    sim.restart("n3")
    LeadershipTransfer(cluster.node("n1")).transfer("n2")
    second = supervisor.runtime_identity("n2")

    result = session.retry_write_via_service(
        service,
        "n2",
        second.generation,
        "write-1",
        max_attempts_per_peer=4,
    )

    assert result.leader_id == "n2"
    assert result.request == pending.request
    assert session.pending_write() is None
    assert session.last_completed_request_id == 1
    assert kv.get("n2", "x") == "one"
    assert [item.operation_id for item in clients.history.invocations()] == ["write-1"]
    assert [item.operation_id for item in clients.history.completed()] == ["write-1"]
    assert sum(
        record.kind == "raft-client-append"
        and record.details.get("client_id") == "writer"
        and record.details.get("request_id") == 1
        for record in sim.trace
    ) == 1
    RaftSafetyHarness(cluster).checkpoint()


def test_generation_loss_after_commit_preserves_session_write_for_retry(
    monkeypatch,
) -> None:
    sim, cluster, supervisor, kv, clients, service, session = _service_session()
    leader = cluster.node("n1")
    first = supervisor.runtime_identity("n1")
    original_apply = kv.apply_committed
    apply_calls = 0

    def apply_then_retire(node_id: str):
        nonlocal apply_calls
        applied = original_apply(node_id)
        apply_calls += 1
        if apply_calls == 2:
            assert leader.step_down(reason="test-after-commit")
        return applied

    monkeypatch.setattr(kv, "apply_committed", apply_then_retire)

    with pytest.raises(LeaderRuntimeUnavailable):
        session.write_via_service(
            service,
            "n1",
            first.generation,
            "write-1",
            1,
            Put("x", "one"),
        )

    pending = session.pending_write()
    assert pending is not None
    assert clients.pending_write("write-1") == pending.request
    assert kv.get("n1", "x") == "one"
    assert session.last_completed_request_id == -1

    monkeypatch.setattr(kv, "apply_committed", original_apply)
    leader.start_election()
    sim.run()
    second = supervisor.runtime_identity("n1")
    result = session.retry_write_via_service(
        service,
        "n1",
        second.generation,
        "write-1",
    )

    assert result.request == pending.request
    assert session.pending_write() is None
    assert session.last_completed_request_id == 1
    assert [item.operation_id for item in clients.history.completed()] == ["write-1"]


def test_failed_service_read_remains_single_flight_until_exact_retry() -> None:
    sim, cluster, supervisor, _, clients, service, session = _service_session()
    identity = supervisor.runtime_identity("n1")
    session.write_via_service(
        service,
        "n1",
        identity.generation,
        "write-1",
        1,
        Put("x", "one"),
    )
    sim.crash("n2")
    sim.crash("n3")

    with pytest.raises(ReadQuorumUnavailable):
        session.linearizable_read_via_service(
            service,
            "n1",
            identity.generation,
            "read-1",
            "x",
            max_attempts_per_peer=1,
        )

    assert session.pending_read() is not None
    assert [item.operation_id for item in clients.history.pending()] == ["read-1"]

    sim.restart("n2")
    sim.restart("n3")
    read = session.retry_linearizable_read_via_service(
        service,
        "n1",
        identity.generation,
        "read-1",
        max_attempts_per_peer=4,
    )

    assert read.value == "one"
    assert session.pending_read() is None
    assert [item.operation_id for item in clients.history.invocations()] == [
        "write-1",
        "read-1",
    ]
    assert [item.operation_id for item in clients.history.completed()] == [
        "write-1",
        "read-1",
    ]
    assert RaftSafetyHarness(cluster).checkpoint() is None


def test_generation_fenced_session_recovery_uses_current_leader_dedup_state() -> None:
    sim, cluster, supervisor, _, clients, service, session = _service_session()
    identity = supervisor.runtime_identity("n1")
    session.write_via_service(
        service,
        "n1",
        identity.generation,
        "write-5",
        5,
        Put("x", "five"),
    )

    stale = KVClientSession.recover(clients, "writer", "n3")
    assert stale.last_completed_request_id == -1

    recovered = KVClientSession.recover_generation_fenced(
        service,
        "writer",
        "n1",
        identity.generation,
    )

    assert recovered.last_completed_request_id == 5
    assert recovered.pending_write() is None
    events = [
        record
        for record in sim.trace
        if record.kind == "client-session-recover-generation-fenced"
    ]
    assert len(events) == 1
    assert events[0].details["node"] == "n1"
    assert events[0].details["generation"] == identity.generation
    assert [record for record in sim.trace if record.kind == "raft-leader-service-barrier"]
    RaftSafetyHarness(cluster).checkpoint()
