from __future__ import annotations

from dataclasses import dataclass

from .leader_liveness import (
    LeaderLivenessError,
    LeaderQuorumEvidence,
    LeaderQuorumMonitor,
)


@dataclass(frozen=True, slots=True)
class HeartbeatRound:
    round_number: int
    scheduled_time: int
    completed_time: int
    evidence: LeaderQuorumEvidence


class LeaderHeartbeatController:
    """Run a finite deterministic heartbeat/CheckQuorum control loop.

    Heartbeat rounds are scheduled in logical time and each round has a bounded
    response window. No future heartbeat is enqueued in advance, so running a
    finite number of rounds always leaves the simulator free of self-perpetuating
    control events.
    """

    def __init__(
        self,
        monitor: LeaderQuorumMonitor,
        *,
        heartbeat_interval: int,
        response_timeout: int,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if response_timeout <= 0:
            raise ValueError("response_timeout must be positive")
        if response_timeout > heartbeat_interval:
            raise ValueError("response_timeout cannot exceed heartbeat_interval")

        self.monitor = monitor
        self.leader = monitor.leader
        self.sim = self.leader.sim
        self.heartbeat_interval = heartbeat_interval
        self.response_timeout = response_timeout
        self._next_heartbeat_time = self.sim.time + heartbeat_interval
        self._round_number = 0

    @property
    def next_heartbeat_time(self) -> int:
        return self._next_heartbeat_time

    def run(self, *, rounds: int) -> tuple[HeartbeatRound, ...]:
        if rounds <= 0:
            raise ValueError("rounds must be positive")

        completed: list[HeartbeatRound] = []
        for _ in range(rounds):
            scheduled_time = max(self._next_heartbeat_time, self.sim.time)
            if scheduled_time != self._next_heartbeat_time:
                self.sim._record(
                    "raft-heartbeat-round-delayed",
                    leader=self.leader.node_id,
                    term=self.leader.current_term,
                    planned_time=self._next_heartbeat_time,
                    actual_time=scheduled_time,
                )
            self.sim.run_until_time(scheduled_time)
            self._round_number += 1
            round_number = self._round_number
            self.sim._record(
                "raft-heartbeat-round-start",
                leader=self.leader.node_id,
                term=self.leader.current_term,
                round=round_number,
                scheduled_time=scheduled_time,
                response_timeout=self.response_timeout,
            )
            try:
                evidence = self.monitor.check_window(
                    response_timeout=self.response_timeout
                )
            except LeaderLivenessError as exc:
                self.sim._record(
                    "raft-heartbeat-round-failed",
                    leader=self.leader.node_id,
                    term=self.leader.current_term,
                    round=round_number,
                    scheduled_time=scheduled_time,
                    completed_time=self.sim.time,
                    reason=str(exc),
                )
                raise

            result = HeartbeatRound(
                round_number=round_number,
                scheduled_time=scheduled_time,
                completed_time=self.sim.time,
                evidence=evidence,
            )
            completed.append(result)
            self.sim._record(
                "raft-heartbeat-round-complete",
                leader=self.leader.node_id,
                term=evidence.term,
                round=round_number,
                scheduled_time=scheduled_time,
                completed_time=self.sim.time,
                acknowledged_voters=evidence.acknowledged_voters,
                quorum_mode=evidence.quorum_mode,
            )
            self._next_heartbeat_time = scheduled_time + self.heartbeat_interval

        return tuple(completed)
