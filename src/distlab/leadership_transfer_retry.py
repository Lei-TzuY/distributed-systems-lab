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

    A crashed, no-longer-caught-up, or log-divergent transferee is rejected
    before enqueueing the retry. The original attempt remains active and can
    still be retried after the transferee is restarted or repaired, provided
    leader authority and membership identity remain unchanged. Rejections of
    an active attempt are recorded in the deterministic trace before raising so
    replay artifacts retain the failed control-plane decision as evidence.
    """

    identity = transport._active_attempts.get(attempt_id)
    if identity is None:
        raise ValueError(f"unknown active leadership transfer attempt {attempt_id}")
    leader_id, transferee_id, term = identity

    def reject(reason: str, message: str) -> None:
        transport.sim._record(
            "raft-timeout-now-retry-rejected",
            leader=leader_id,
            transferee=transferee_id,
            term=term,
            attempt_id=attempt_id,
            reason=reason,
        )
        raise RuntimeError(message)

    leader = transport.cluster.node(leader_id)
    if not transport.sim.is_alive(leader_id):
        reject("source-leader-crashed", "leadership transfer retry requires a live source leader")
    if leader.role is not RaftRole.LEADER or leader.current_term != term:
        reject(
            "source-no-longer-current-leader",
            "leadership transfer retry requires original leader authority",
        )
    if transport._attempt_configurations.get(attempt_id) is not transport._configuration_identity():
        reject(
            "membership-configuration-changed",
            "leadership transfer retry requires unchanged voting configuration",
        )
    if not transport.sim.is_alive(transferee_id):
        reject("transferee-crashed", "leadership transfer retry requires a live transferee")

    transferee = transport.cluster.node(transferee_id)
    if (
        transferee.last_log_index != leader.last_log_index
        or transferee.last_log_term != leader.last_log_term
        or transferee.commit_index < leader.commit_index
    ):
        reject(
            "transferee-not-caught-up",
            "leadership transfer retry requires a caught-up transferee",
        )
    if not transport._retained_log_overlap_matches(leader, transferee):
        reject(
            "transferee-log-mismatch",
            "leadership transfer retry requires matching retained logs",
        )

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