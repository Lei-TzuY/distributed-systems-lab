from __future__ import annotations

from dataclasses import dataclass

from .leadership_transfer_retry import retry_timeout_now
from .leadership_transfer_transport import LeadershipTransferTransport
from .raft import RaftNode, RaftRole


class LeadershipTransferError(RuntimeError):
    """Base error for leadership-transfer failures."""


class LeadershipTransferTargetUnavailable(LeadershipTransferError):
    """Raised when the selected transferee is unavailable."""


class LeadershipTransferIncomplete(LeadershipTransferError):
    """Raised when transfer cannot complete within the configured attempts."""


@dataclass(frozen=True)
class LeadershipTransferResult:
    previous_leader_id: str
    new_leader_id: str
    previous_term: int
    new_term: int


class LeadershipTransfer:
    def __init__(self, leader: RaftNode) -> None:
        self.leader = leader
        self.sim = leader.sim
        self.cluster = leader.cluster
        self.transfer_transport = LeadershipTransferTransport(leader)

    def _record_failure(
        self,
        transferee_id: str,
        term: int,
        *,
        stage: str,
        reason: str,
    ) -> None:
        self.sim.record(
            "raft-leadership-transfer-failed",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=term,
            stage=stage,
            reason=reason,
        )

    def _validate_target(self, transferee_id: str) -> RaftNode:
        if self.leader.role is not RaftRole.LEADER:
            raise LeadershipTransferIncomplete("leadership transfer requires a current leader")
        if transferee_id == self.leader.node_id:
            raise ValueError("leadership transferee must differ from current leader")
        if transferee_id not in self.cluster.node_ids:
            raise ValueError(f"unknown leadership transferee {transferee_id!r}")
        if not self.sim.is_alive(transferee_id):
            raise LeadershipTransferTargetUnavailable(
                f"leadership transferee {transferee_id!r} is not alive"
            )

        active_membership = self.leader.active_membership()
        if not active_membership.is_voter(transferee_id):
            raise ValueError(
                f"leadership transferee {transferee_id!r} is not an active voter"
            )
        if active_membership.is_joint and transferee_id not in active_membership.new_voters:
            raise ValueError(
                f"leadership transferee {transferee_id!r} is not a voter in the incoming configuration"
            )
        return self.cluster.node(transferee_id)

    def _target_matches_leader_log(self, target: RaftNode) -> bool:
        if target.last_log_index != self.leader.last_log_index:
            return False
        if target.last_log_term != self.leader.last_log_term:
            return False
        if target.commit_index < self.leader.commit_index:
            return False
        overlap_start = max(
            self.leader.log_base_index + 1,
            target.log_base_index + 1,
        )
        for index in range(overlap_start, self.leader.last_log_index + 1):
            if target.entry_at(index) != self.leader.entry_at(index):
                return False
        return True

    def _catch_up_target(self, target: RaftNode, *, max_rounds: int) -> bool:
        for _ in range(max_rounds):
            if not self.sim.is_alive(target.node_id):
                return False
            if self._target_matches_leader_log(target):
                return True
            if self.leader.role is not RaftRole.LEADER:
                return False
            self.leader.replicate(target.node_id)
            self.sim.run()
        return self._target_matches_leader_log(target)

    def _transfer_elected(self, target: RaftNode, previous_term: int) -> bool:
        return (
            target.role is RaftRole.LEADER
            and target.current_term > previous_term
            and self.leader.role is not RaftRole.LEADER
        )

    def transfer(
        self,
        transferee_id: str,
        *,
        max_catch_up_rounds: int = 8,
        max_timeout_now_attempts: int = 1,
    ) -> LeadershipTransferResult:
        if max_catch_up_rounds <= 0:
            raise ValueError("max_catch_up_rounds must be positive")
        if max_timeout_now_attempts <= 0:
            raise ValueError("max_timeout_now_attempts must be positive")

        target = self._validate_target(transferee_id)
        previous_term = self.leader.current_term
        self.sim.record(
            "raft-leadership-transfer-started",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=previous_term,
        )

        if not self._catch_up_target(target, max_rounds=max_catch_up_rounds):
            if not self.sim.is_alive(transferee_id):
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="catch-up",
                    reason="transferee crashed during catch-up",
                )
                raise LeadershipTransferTargetUnavailable(
                    f"leadership transferee {transferee_id!r} crashed during catch-up"
                )
            self._record_failure(
                transferee_id,
                previous_term,
                stage="catch-up",
                reason="transferee did not catch up",
            )
            raise LeadershipTransferIncomplete(
                f"leadership transferee {transferee_id!r} did not catch up"
            )

        if self.leader.role is not RaftRole.LEADER or self.leader.current_term != previous_term:
            self._record_failure(
                transferee_id,
                previous_term,
                stage="catch-up",
                reason="source leader lost leadership during catch-up",
            )
            raise LeadershipTransferIncomplete(
                "source leader lost leadership during leadership-transfer catch-up"
            )

        self.sim.record(
            "raft-leadership-transfer-ready",
            leader=self.leader.node_id,
            transferee=transferee_id,
            term=previous_term,
            last_log_index=self.leader.last_log_index,
            last_log_term=self.leader.last_log_term,
            commit_index=self.leader.commit_index,
        )

        try:
            attempt_id = self.transfer_transport.send_timeout_now(transferee_id)
        except (RuntimeError, ValueError) as exc:
            target_unavailable = not self.sim.is_alive(transferee_id)
            reason = (
                "transferee crashed before TimeoutNow dispatch"
                if target_unavailable
                else f"TimeoutNow dispatch rejected: {exc}"
            )
            self._record_failure(
                transferee_id,
                previous_term,
                stage="dispatch",
                reason=reason,
            )
            if target_unavailable:
                raise LeadershipTransferTargetUnavailable(
                    f"leadership transferee {transferee_id!r} crashed before TimeoutNow dispatch"
                ) from exc
            raise LeadershipTransferIncomplete(
                f"failed to dispatch TimeoutNow to leadership transferee {transferee_id!r}"
            ) from exc

        self.sim.run()
        attempts = 1
        while (
            not self._transfer_elected(target, previous_term)
            and attempts < max_timeout_now_attempts
        ):
            attempts += 1
            if not self.sim.is_alive(transferee_id):
                self.transfer_transport.cancel_timeout_now(attempt_id)
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="retry",
                    reason="transferee crashed before retry catch-up",
                )
                raise LeadershipTransferTargetUnavailable(
                    f"leadership transferee {transferee_id!r} crashed before retry catch-up"
                )
            if not self._catch_up_target(target, max_rounds=max_catch_up_rounds):
                self.transfer_transport.cancel_timeout_now(attempt_id)
                if not self.sim.is_alive(transferee_id):
                    self._record_failure(
                        transferee_id,
                        previous_term,
                        stage="retry",
                        reason="transferee crashed during retry catch-up",
                    )
                    raise LeadershipTransferTargetUnavailable(
                        f"leadership transferee {transferee_id!r} crashed during retry catch-up"
                    )
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="retry",
                    reason="transferee did not catch up before retry",
                )
                raise LeadershipTransferIncomplete(
                    f"leadership transferee {transferee_id!r} did not catch up before retry"
                )
            if self.leader.role is not RaftRole.LEADER or self.leader.current_term != previous_term:
                self.transfer_transport.cancel_timeout_now(attempt_id)
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="retry",
                    reason="source leader lost leadership during retry catch-up",
                )
                raise LeadershipTransferIncomplete(
                    "source leader lost leadership during leadership-transfer retry catch-up"
                )
            try:
                retry_timeout_now(self.transfer_transport, attempt_id)
            except (RuntimeError, ValueError) as exc:
                self.transfer_transport.cancel_timeout_now(attempt_id)
                target_unavailable = not self.sim.is_alive(transferee_id)
                reason = (
                    "transferee crashed before TimeoutNow retry dispatch"
                    if target_unavailable
                    else f"TimeoutNow retry rejected: {exc}"
                )
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="retry",
                    reason=reason,
                )
                if target_unavailable:
                    raise LeadershipTransferTargetUnavailable(
                        "leadership transferee "
                        f"{transferee_id!r} crashed before TimeoutNow retry dispatch"
                    ) from exc
                raise LeadershipTransferIncomplete(
                    f"failed to retry TimeoutNow for leadership transferee {transferee_id!r}"
                ) from exc

        if not self._transfer_elected(target, previous_term):
            self.transfer_transport.cancel_timeout_now(attempt_id)
            source_lost_authority = (
                self.leader.role is not RaftRole.LEADER
                or self.leader.current_term != previous_term
            )
            transferee_did_not_start_next_term = target.current_term <= previous_term
            if source_lost_authority and transferee_did_not_start_next_term:
                self._record_failure(
                    transferee_id,
                    previous_term,
                    stage="election",
                    reason="source leader lost leadership during transferee election",
                )
                raise LeadershipTransferIncomplete(
                    "source leader lost leadership before transferee election completed"
                )
            self._record_failure(
                transferee_id,
                previous_term,
                stage="election",
                reason="transferee did not become leader",
            )
            raise LeadershipTransferIncomplete(
                f"leadership transferee {transferee_id!r} did not become leader"
            )

        self.sim.record(
            "raft-leadership-transfer-completed",
            previous_leader=self.leader.node_id,
            new_leader=target.node_id,
            previous_term=previous_term,
            new_term=target.current_term,
        )
        return LeadershipTransferResult(
            previous_leader_id=self.leader.node_id,
            new_leader_id=target.node_id,
            previous_term=previous_term,
            new_term=target.current_term,
        )
