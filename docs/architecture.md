# Architecture

## Deterministic simulator foundation

The simulator is deliberately event-driven and single-threaded. Protocol code must observe simulated logical time rather than wall-clock time.

### Core invariants

1. **Stable ordering** — events scheduled for the same logical time are ordered by a monotonic sequence number assigned at scheduling time.
2. **Explicit faults** — drop, delay, and duplicate behavior is encoded in `FaultPlan`; it is not hidden inside timing or host randomness.
3. **Replayable scenarios** — an identical ordered `ScenarioAction` sequence and `FaultPlan` must produce an identical trace when handlers are deterministic.
4. **Crash boundary** — crash clears volatile node state but does not mutate persistent node state.
5. **No wall clock** — simulator progress is driven exclusively by queued events and logical time.
6. **Structured trace** — scheduling, delivery, faults, crash, and restart decisions are recorded as `TraceRecord` values suitable for regression assertions.

### Scenario interleavings

`ScenarioAction.run()` is an explicit deterministic barrier inside a replayable scenario. With no limit it drains the current event queue; with `max_events=N` it consumes at most `N` queued events before the next scenario action is applied. This allows failure schedules to encode interleavings such as send → crash → deliver/discard → restart without relying on wall-clock sleeps or host scheduling.

Every scenario still performs a final queue drain after the ordered action sequence, so barriers are needed only where the relative placement of crash/restart or later sends matters.

### Fault matching

`FaultRule` can match a source, destination, and per-link send ordinal. Rules are evaluated in declaration order; the first matching rule wins. This makes a failure schedule explicit and reviewable.

The current foundation supports:

- deliver
- drop
- delay by a deterministic logical-time delta
- duplicate with a deterministic second-delivery delta

Randomized fault generation is intentionally deferred. When introduced, it must compile a seed into an explicit fault schedule that can be persisted and replayed.

### Crash semantics

The simulator maintains separate dictionaries for persistent and volatile node state. `crash(node)` clears volatile state while preserving persistent state. `restart(node)` starts with empty volatile state and the previously persisted state.

This is only a simulation boundary. Raft will later wrap persistence operations in a stricter storage interface with explicit term/vote/log durability rules and crash injection at persistence boundaries.

## Raft election correctness

Raft term and vote state are persistent, while role and votes received are volatile. Election Safety is enforced as an executable cluster assertion: recording two different leaders for the same term raises `ElectionSafetyViolation`.

`RequestVote` carries the candidate's `last_log_index` and `last_log_term`. A voter grants a vote only when the candidate is at least as up to date as the voter's own persistent log, using Raft's lexicographic rule: compare the last log term first, then the last log index when terms are equal. A higher-term request still advances persistent term state even when the vote is rejected because the candidate's log is stale.

## Raft AppendEntries and Log Matching

`AppendEntries` now establishes follower-side consistency semantics before leader replication bookkeeping is introduced. A follower accepts an append only when `prev_log_index` exists and its term matches `prev_log_term`. Rejected probes do not mutate the persistent log.

On a successful append, an existing matching prefix is preserved. The first conflicting term truncates the follower suffix and the leader's remaining entries are appended persistently. A same-index/same-term entry with different contents is treated as an invariant failure rather than normal conflict repair because Raft's Log Matching property requires such entries to identify the same history.

`RaftCluster.assert_log_matching()` is an executable invariant: whenever two persisted logs contain an entry at the same index and term, their prefixes through that entry must be identical. Successful follower repairs invoke the assertion immediately.

The current leader API deliberately exposes only an explicit `send_append_entries()` probe. It does not yet implement `nextIndex`, `matchIndex`, retries, commit advancement, or state-machine application; those remain separate bounded milestones.

### Next layers

The intended sequence is now:

1. deterministic election timeout and timer reset semantics
2. leader `nextIndex` / `matchIndex` retry loop over AppendEntries responses
3. commit advancement and Leader Completeness / State Machine Safety assertions
4. persistence/recovery crash matrices
5. replicated state machine and linearizability histories
6. snapshots and membership changes

No additional consensus protocol should be introduced before the Raft safety harness is mature.
