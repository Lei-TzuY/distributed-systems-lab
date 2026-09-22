import pytest

from distlab.client_history import KVClientHistory
from distlab.kv import Put, ReplicatedKV
from distlab.leader_runtime import (
    LeaderRuntimeSupervisor,
    LeaderRuntimeUnavailable,
    StaleLeaderRuntimeGeneration,
)
from distlab.leader_service import (
    LeaderKVService,
    LeaderWriteQuorumUnavailable,
)
from distlab.leadership_transfer import LeadershipTransfer
from distlab.linearizability import SingleKeyKVLinearizabilityChecker
from distlab.linearizable_read import ReadQuorumUnavailable
from distlab.membership import ReconfigurableRaftCluster
from distlab.raft import RaftCluster, RaftRole
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Simulator


def _service_cluster() -> tuple[
    Simulator,
    RaftCluster,
    LeaderRuntimeSupervisor,
    ReplicatedKV,
    KVClientHistory,
    LeaderKVService,
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
    return sim, cluster, supervisor, kv, clients, service


def test_generation_fenced_service_commits_write_and_linearizable_read() -> None:
    _, cluster, supervisor, kv, clients, service = _service_cluster()
    identity = supervisor.runtime_identity("n1")

    written = service.write(
        "n1",
        identity.generation,
        operation_id="write-1",
        client_id="client-a",
        request_id=1,
        operation=Put("x", "one"),
    )

    assert written.term == identity.term == 1
    assert written.generation == identity.generation
    assert written.commit_index >= written.log_index
    assert kv.get("n1", "x") == "one"

    read = service.linearizable_read(
        "n1",
        identity.generation,
        operation_id="read-1",
        client_id="client-a",
        key="x",
    )

    assert read.value == "one"
    assert read.term == identity.term
    assert read.generation == identity.generation
    assert [item.operation_id for item in clients.history.completed()] == [
        "write-1",
        "read-1",
    ]
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True
    kv.applier.assert_state_machine_safety()
    RaftSafetyHarness(cluster).checkpoint()


def test_stale_generation_is_rejected_before_client_history_mutation() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    node = cluster.node("n1")
    node.start_election()
    first = supervisor.runtime_identity("n1")
    assert node.step_down(reason="test-rollover")
    node.start_election()
    second = supervisor.runtime_identity("n1")
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    service = LeaderKVService(supervisor, kv, clients)

    with pytest.raises(StaleLeaderRuntimeGeneration, match="stale"):
        service.write(
            "n1",
            first.generation,
            operation_id="stale-write",
            client_id="client-a",
            request_id=1,
            operation=Put("x", "stale"),
        )

    assert second.generation > first.generation
    assert clients.history.invocations() == ()
    assert not [
        record
        for record in sim.trace
        if record.kind == "raft-client-append"
        and record.details.get("client_id") == "client-a"
    ]


def test_generation_loss_after_commit_keeps_write_pending_for_exact_retry(
    monkeypatch,
) -> None:
    sim, cluster, supervisor, kv, clients, service = _service_cluster()
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
        service.write(
            "n1",
            first.generation,
            operation_id="write-1",
            client_id="client-a",
            request_id=1,
            operation=Put("x", "one"),
        )

    assert kv.get("n1", "x") == "one"
    assert clients.pending_write("write-1") is not None
    assert clients.history.completed() == ()
    append_count = sum(
        record.kind == "raft-client-append"
        and record.details.get("client_id") == "client-a"
        and record.details.get("request_id") == 1
        for record in sim.trace
    )
    assert append_count == 1

    monkeypatch.setattr(kv, "apply_committed", original_apply)
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER
    second = supervisor.runtime_identity("n1")
    result = service.retry_write(
        "n1",
        second.generation,
        operation_id="write-1",
    )

    assert result.generation == second.generation
    assert clients.pending_write("write-1") is None
    assert [item.operation_id for item in clients.history.completed()] == ["write-1"]
    assert sum(
        record.kind == "raft-client-append"
        and record.details.get("client_id") == "client-a"
        and record.details.get("request_id") == 1
        for record in sim.trace
    ) == 1


def test_pending_write_retries_on_transferred_leader_with_current_term_barrier() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3", "n4", "n5"))
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
    first = supervisor.runtime_identity("n1")

    for node_id in ("n2", "n3", "n4", "n5"):
        sim.crash(node_id)

    with pytest.raises(LeaderWriteQuorumUnavailable):
        service.write(
            "n1",
            first.generation,
            operation_id="write-1",
            client_id="client-a",
            request_id=1,
            operation=Put("x", "one"),
            max_attempts_per_peer=1,
        )

    assert cluster.node("n1").commit_index == 0
    pending = clients.pending_write("write-1")
    assert pending is not None
    assert clients.history.completed() == ()

    sim.restart("n2")
    sim.restart("n3")
    transfer = LeadershipTransfer(cluster.node("n1")).transfer("n2")
    assert transfer.new_leader_id == "n2"
    assert transfer.commit_index == 0
    second = supervisor.runtime_identity("n2")

    result = service.retry_write(
        "n2",
        second.generation,
        operation_id="write-1",
        max_attempts_per_peer=4,
    )

    assert result.leader_id == "n2"
    assert result.term == 2
    assert result.commit_index >= 2
    assert kv.get("n2", "x") == "one"
    assert clients.pending_write("write-1") is None
    barriers = [
        record
        for record in sim.trace
        if record.kind == "raft-current-term-barrier"
        and record.details["leader"] == "n2"
        and record.details["term"] == 2
    ]
    assert barriers
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True
    kv.applier.assert_state_machine_safety()
    RaftSafetyHarness(cluster).checkpoint()

def test_leadership_transfer_fences_old_service_token_and_new_leader_can_read() -> None:
    _, cluster, supervisor, _, clients, service = _service_cluster()
    first = supervisor.runtime_identity("n1")
    service.write(
        "n1",
        first.generation,
        operation_id="write-1",
        client_id="client-a",
        request_id=1,
        operation=Put("x", "one"),
    )

    LeadershipTransfer(cluster.node("n1")).transfer("n2")
    second = supervisor.runtime_identity("n2")
    before = clients.history.invocations()

    with pytest.raises(LeaderRuntimeUnavailable):
        service.linearizable_read(
            "n1",
            first.generation,
            operation_id="stale-read",
            client_id="client-a",
            key="x",
        )

    assert clients.history.invocations() == before
    read = service.linearizable_read(
        "n2",
        second.generation,
        operation_id="read-2",
        client_id="client-a",
        key="x",
        max_attempts_per_peer=4,
    )

    assert read.value == "one"
    assert read.term == 2
    assert read.generation == second.generation
    assert SingleKeyKVLinearizabilityChecker().check(clients.history).linearizable is True


def test_joint_consensus_write_uses_membership_aware_commit_quorum() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run()
    identity = supervisor.runtime_identity("n1")
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    service = LeaderKVService(supervisor, kv, clients)
    cluster.begin_joint_consensus("n1", ("n1", "n4", "n5"))
    sim.crash("n4")
    sim.crash("n5")

    with pytest.raises(LeaderWriteQuorumUnavailable):
        service.write(
            "n1",
            identity.generation,
            operation_id="joint-write",
            client_id="client-a",
            request_id=1,
            operation=Put("x", "blocked"),
            max_attempts_per_peer=2,
        )

    assert cluster.node("n1").commit_index == 0
    assert clients.pending_write("joint-write") is not None
    assert clients.history.completed() == ()
    failure = [
        record
        for record in sim.trace
        if record.kind == "raft-leader-service-quorum-failed"
    ][-1]
    assert failure.details["operation"] == "write"
    RaftSafetyHarness(cluster).checkpoint()


def test_joint_consensus_linearizable_read_requires_new_voter_majority() -> None:
    sim = Simulator()
    cluster = ReconfigurableRaftCluster(
        sim,
        ("n1", "n2", "n3", "n4", "n5"),
        voters=("n1", "n2", "n3"),
    )
    supervisor = LeaderRuntimeSupervisor(
        cluster,
        heartbeat_interval=3,
        response_timeout=2,
    )
    cluster.node("n1").start_election()
    sim.run()
    identity = supervisor.runtime_identity("n1")
    kv = ReplicatedKV(cluster)
    clients = KVClientHistory(kv)
    service = LeaderKVService(supervisor, kv, clients)
    service.write(
        "n1",
        identity.generation,
        operation_id="write-1",
        client_id="client-a",
        request_id=1,
        operation=Put("x", "one"),
    )
    cluster.begin_joint_consensus("n1", ("n1", "n4", "n5"))
    sim.crash("n4")
    sim.crash("n5")

    with pytest.raises(ReadQuorumUnavailable):
        service.linearizable_read(
            "n1",
            identity.generation,
            operation_id="joint-read",
            client_id="client-a",
            key="x",
            max_attempts_per_peer=2,
        )

    pending = {item.operation_id for item in clients.history.pending()}
    assert "joint-read" in pending
    assert not [
        item for item in clients.history.completed() if item.operation_id == "joint-read"
    ]
    RaftSafetyHarness(cluster).checkpoint()
