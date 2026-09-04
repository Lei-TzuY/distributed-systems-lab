# Distributed Correctness Stability Checkpoint

This document defines the first stable maintenance boundary for `distributed-systems-lab`.

## Stable correctness chain

The checkpoint covers one deterministic end-to-end chain:

1. logical-time simulation and structured traces
2. explicit replayable message faults
3. Raft election and log replication
4. crash/restart persistence boundaries and recovery
5. committed state-machine application
6. replicated KV semantics and client request deduplication
7. operation-history capture and linearizability checking
8. seeded workload, fault, and lifecycle campaigns
9. versioned exact-replay failure artifacts
10. deterministic 1-minimal failure reduction

A maintenance change inside this boundary should preserve replay determinism, failure classification, Raft safety invariants, durable applied-prefix semantics, and artifact compatibility unless a versioned migration is explicitly intended.

## Integration responsibilities

| Layer | Owns | Must not silently own |
| --- | --- | --- |
| simulator | logical time, event ordering, fault application, traces | protocol semantics |
| Raft | term/vote/log/commit protocol state | application-specific KV semantics |
| state machine | ordered durable application | leader election or transport faults |
| replicated KV | deterministic command semantics and deduplication | linearizability search policy |
| history/checker | client-visible histories and linearizability decisions | simulator scheduling |
| campaign | seed compilation, replay, failure artifact publication | hidden host randomness |
| minimizers | deterministic deletion-based reduction | changing the failure oracle |

## Maintenance triggers

After this checkpoint, repository changes should normally be driven by one of:

- CI or deterministic regression failure
- a reproducible Raft/state-machine/linearizability correctness bug
- replay or artifact compatibility failure
- a real campaign exposing an infrastructure defect
- a clearly scoped Phase 2 milestone selected explicitly

Do not create commits merely because a scheduled run occurred.

## Phase 2, not maintenance

The following are intentionally outside this checkpoint and should start as explicit new phases rather than incremental filler work:

- snapshot/install-snapshot and log compaction
- membership changes / joint consensus
- additional consensus protocols such as Paxos or Zab
- gossip/SWIM-style membership systems
- production networking, RPC, storage engines, or deployment tooling
- performance tuning that weakens deterministic correctness instrumentation

## Validation gate

Before integrating maintenance changes:

- run Ruff
- run the complete pytest suite
- keep seeded/randomized failures exactly replayable
- add focused invariant/regression coverage for protocol changes
- verify the exact candidate head is green and the base has not drifted
