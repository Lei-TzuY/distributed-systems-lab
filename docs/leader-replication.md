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

`LeaderReplicator` still does not own a wall-clock loop. Commit advancement and state-machine application remain separate concerns, while heartbeat scheduling is now composed above it by `LeaderHeartbeatController`. The controller uses explicit logical-time cadence and bounded response windows, so finite simulator scenarios remain replayable and terminating.

## Why a separate controller first?

The current Raft core already has a deterministic simulator and correct follower-side AppendEntries prefix checking/conflict repair. Keeping leader replication progress in a small controller makes the retry state executable and testable without simultaneously changing commit semantics. Once the backtracking invariants are stable, the same per-peer progress can be integrated with leader commit advancement and heartbeat scheduling.

## Trace events

- `raft-replication-probe`: records the term, peer, current `nextIndex`, previous-log index, suffix size, and attempt number.
- `raft-replication-backtrack`: records a rejected probe and the monotonic one-step decrease of `nextIndex`.
- `raft-replication-advance`: records a successful response and the monotonic advance of `matchIndex` / `nextIndex`.

A deterministic trace therefore explains exactly which prefix mismatch caused each retry and can be replayed alongside the simulator's existing AppendEntries events.
## Bounded heartbeat control loop

`LeaderHeartbeatController` composes leader replication and CheckQuorum without
introducing an unbounded background task.

- `Simulator.run_until_time()` advances only through events due inside an explicit
  logical-time window and leaves later events queued.
- heartbeat rounds are finite and scheduled at deterministic logical deadlines;
- each round sends correlated empty AppendEntries probes to active voters concurrently;
- same-term rejection responses still prove voter activity;
- the round evaluates stable or joint-consensus quorum after a bounded response window;
- quorum loss retires leader authority in the same term and re-arms election timeout state;
- observing a higher term uses the normal Raft term-advance path, which also re-arms
  the former leader's election timeout.

No future heartbeat is enqueued beyond the requested round budget, so test and
scenario callers retain explicit control over termination.

## Leader runtime lifecycle ownership

`LeaderRuntimeSupervisor` binds heartbeat and CheckQuorum state to explicit
node-local leader generations.

- every successful election acquires a new runtime generation for that node;
- same-term CheckQuorum step-down, higher-term authority loss, crash, and restart
  retire only that node's matching generation;
- leadership transfer retires the source generation and starts a new target
  generation when the transferee wins;
- partitions may temporarily leave different-term leaders with independent
  runtimes, matching distributed Raft semantics instead of choosing one
  simulator-global leader;
- reconfigurable clusters automatically use membership-aware replication and
  joint-consensus quorum rules;
- callers may fence work with an expected generation so stale leader-runtime
  authority fails explicitly.

Crash lifecycle callbacks run before volatile state is cleared, preserving enough
local authority state to retire the correct generation deterministically. A
retired leader that observes a higher-term but stale RequestVote also re-arms its
election timeout after rejecting the vote, so runtime retirement cannot strand the
node as a timerless follower.
