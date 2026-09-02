# Deterministic leader replication progress

This milestone adds the leader-side `nextIndex` / `matchIndex` retry state needed to drive the existing follower-side AppendEntries log-matching implementation.

## Scope

`LeaderReplicator` is intentionally narrow:

- it snapshots the leader term when created after election;
- every follower starts at `nextIndex = leader.last_log_index + 1` and `matchIndex = 0`;
- a probe sends the leader suffix beginning at `nextIndex`;
- a rejection decrements `nextIndex` by one, never below 1, and retries deterministically;
- a success advances `matchIndex` monotonically and sets `nextIndex = matchIndex + 1`;
- every probe, backtrack, and successful advance is written to the simulator trace;
- bounded retry calls preserve their progress so a later call resumes from the same deterministic state.

This is deliberately **not** commit advancement. The component does not calculate a majority commit index, apply commands to a state machine, schedule periodic heartbeats, optimize conflict hints, or hide dropped-message retry behind wall-clock behavior. Those remain separate milestones so each correctness step can be tested independently.

## Why a separate controller first?

The current Raft core already has a deterministic simulator and correct follower-side AppendEntries prefix checking/conflict repair. Keeping leader replication progress in a small controller makes the retry state executable and testable without simultaneously changing commit semantics. Once the backtracking invariants are stable, the same per-peer progress can be integrated with leader commit advancement and heartbeat scheduling.

## Trace events

- `raft-replication-probe`: records the term, peer, current `nextIndex`, previous-log index, suffix size, and attempt number.
- `raft-replication-backtrack`: records a rejected probe and the monotonic one-step decrease of `nextIndex`.
- `raft-replication-advance`: records a successful response and the monotonic advance of `matchIndex` / `nextIndex`.

A deterministic trace therefore explains exactly which prefix mismatch caused each retry and can be replayed alongside the simulator's existing AppendEntries events.