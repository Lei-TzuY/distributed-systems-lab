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

### Next layers

Once this foundation is stable, the intended sequence is:

1. Raft term/vote state and election timers
2. RequestVote and Election Safety assertions
3. AppendEntries and Log Matching
4. persistence/recovery crash matrices
5. replicated state machine and linearizability histories
6. snapshots and membership changes

No additional consensus protocol should be introduced before the Raft safety harness is mature.
