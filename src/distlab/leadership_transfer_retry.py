from __future__ import annotations

from .leadership_transfer_transport import LeadershipTransferTransport, TimeoutNow
from .raft import RaftRole


def retry_timeout_now(transport: LeadershipTransferTransport, attempt_id: int) -> None:
    """Retransmit an existing TimeoutNow attempt without minting new authority.

    A dropped TimeoutNow must not force callers to create a second logical
    leadership-transfer attempt. Retrying preserves the original attempt id,
    leader term, and captured membership configuration while sending over the
    same logical Raft link, so deterministic drop/delay/partition semantics
    continue to apply.

    A crashed transferee is rejected before enqueueing the retry. The simulator
    discards messages to crashed logical destinations before delivery handlers
    run, so enqueueing here would consume retry budget without producing a
    protocol observation. The original attempt remains active and can still be
    retried after the transferee is restarted, provided leader authority and
    membership identity remain unchanged.
    """

    identity = transport._active_attempts.get(attempt_id)
    if identity is None:
        raise ValueError(f"unknown active leadership transfer attempt {attempt_id}")
    leader_id, transferee_id, term = identity
    leader = transport.cluster.node(leader_id)
    if not transport.sim.is_alive(leader_id):
        raise RuntimeError("leadership transfer retry requires a live source leader")
    if leader.role is not RaftRole.LEADER or leader.current_term != term:
        raise RuntimeError("leadership transfer retry requires original leader authority")
    if transport._attempt_configurations.get(attempt_id) is not transport._configuration_identity():
        raise RuntimeError("leadership transfer retry requires unchanged voting configuration")
    if not transport.sim.is_alive(transferee_id):
        raise RuntimeError("leadership transfer retry requires a live transferee")

    transport.sim._record(
        "raft-timeout-now-retry",
        leader=leader_id,
        transferee=transferee_id,
        term=term,
        attempt_id=attempt_id,
    )
    transport.sim.send(
        leader_id,
        transferee_id,
        TimeoutNow(
            term=term,
            leader_id=leader_id,
            transferee_id=transferee_id,
            attempt_id=attempt_id,
        ),
        delivery_dst=transport.endpoint(transferee_id),
    )
