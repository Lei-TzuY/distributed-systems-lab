# Architecture

## Deterministic simulator foundation

The simulator is event-driven and single-threaded. Protocol code observes simulated logical time rather than wall-clock time.

Core invariants:

1. **Stable ordering** — events scheduled for the same logical time use a monotonic scheduling sequence.
2. **Explicit faults** — drop, delay, and duplicate behavior is encoded in `FaultPlan`.
3. **Replayable scenarios** — identical ordered scenario actions and persisted schedules produce identical traces when handlers are deterministic.
4. **Crash boundary** — crash clears volatile node state without mutating persistent state.
5. **No wall clock** — progress is driven by queued events and logical time.
6. **Structured trace** — scheduling, delivery, fault, crash, restart, and application decisions are recorded for regression assertions.

`ScenarioAction.run()` provides deterministic barriers within a replayable scenario. Message faults are explicit `FaultRule` values and seeded exploration compiles randomness into persisted schedules before execution; exact replay never regenerates randomness.

## Raft correctness boundary

Raft term, vote, and log state are persistent. Role, election bookkeeping, replication progress, and commit knowledge are volatile and reconstructed as required after restart.

The implementation covers:

- RequestVote with log up-to-date checks
- Election Safety
- AppendEntries prefix matching and conflict replacement
- leader `nextIndex` / `matchIndex` backtracking
- current-term majority commit advancement
- restart recovery and re-establishment of commit knowledge
- executable Log Matching, Leader Append-Only, Leader Completeness, and State Machine Safety checks

The safety harness is intentionally more important than protocol breadth. No second consensus protocol belongs in this checkpoint.

## State-machine application and durability

`StateMachineApplier` applies committed entries strictly in log-index order and persists the applied prefix. Recovery validates that durable applied history remains a prefix of the persistent Raft log and does not roll back already applied state simply because volatile `commitIndex` was lost.

## Replicated KV and client semantics

`ReplicatedKV` is the concrete deterministic state machine for this checkpoint. It supports key-value writes/deletes and durable reconstruction from the applied Raft prefix.

Client writes carry `(client_id, request_id)` identity. Duplicate requests are detected deterministically so retries do not apply a command twice, and conflicting reuse of a request identity is rejected. `KVClientHistory` records client-visible invocations and responses for linearizability analysis.

## Linearizability

`OperationHistory` captures ordered invocations/completions. `SingleKeyKVLinearizabilityChecker` evaluates whether a completed client-visible history admits a legal sequential execution that respects real-time constraints.

Non-linearizable histories can be deterministically reduced to a 1-minimal witness. Reduction is evidence tooling, not a separate correctness oracle: every candidate is evaluated by the same checker.

## Seeded campaigns and exact replay

Exploration is separated from reproduction:

- `SeededClientWorkloadGenerator` compiles a seed into explicit client actions.
- `SeededFaultGenerator` compiles message-fault opportunities into an explicit `SeededFaultSchedule`.
- `SeededLifecycleGenerator` compiles crash/restart choices into an explicit `SeededLifecycleSchedule`.
- `ReplicatedKVScenarioRunner` executes only explicit schedules.
- `SeededScenarioCampaign` runs bounded seeded scenarios and captures exact-replay failure artifacts.

`CampaignFailureArtifact` is versioned and stores the original schedules, minimized schedules/indices, trace evidence, and minimized history witness needed to reproduce a failure without consulting randomness.

## Failure minimization

The checkpoint has four deterministic reduction surfaces:

1. history operation deletion
2. client workload action deletion
3. message fault-rule deletion
4. lifecycle action deletion

The schedule reducers share one private deterministic deletion policy. Each public reducer keeps its own scenario semantics and failure oracle, but the scan/restart/1-minimal mechanics are centralized to prevent policy drift.

Invalid lifecycle projections are not treated as product failures; they are rejected as non-preserving candidates.

## Public package boundary

The package root exports the stable simulator/Raft/KV/history/campaign APIs plus the seeded lifecycle and schedule-reduction APIs needed to reproduce and reduce failures. Implementation helpers such as the generic deletion primitive remain private.

## Promoted Phase 2 architecture

The original deterministic correctness chain remains the replay/artifact compatibility boundary, but several bounded Phase 2 projects are now integrated:

1. snapshot/install-snapshot correctness and log compaction
2. membership changes with joint-consensus quorum rules
3. leadership transfer, pre-vote, CheckQuorum, and bounded heartbeat control
4. node-local leader runtime generations across election, crash/restart, and authority loss
5. generation-fenced client KV reads/writes
6. generation validation in deterministic scenario writes without changing their network-event schedule
7. commit-synchronous membership authority and generation-fenced transition recovery across leader handoff
8. membership-transition-aware client replication and KV projection across the shared Raft log

`LeaderGenerationGuard` is the shared authority primitive for client-facing service and scenario execution. It reads Raft's node-local active generation state and never schedules heartbeat/runtime work itself, keeping data-plane fencing separate from control-plane tick ownership.

Membership transitions now treat objective Raft commit as the live quorum boundary: durable membership watermark and `VotingConfiguration` advance together. Uncommitted membership commands can be reconstructed by the next leader from the durable log, while the old transition controller is fenced by its retired leader generation. Older-term recovered proposals use an explicit current-term recovery barrier before commit.

The next architectural frontier should therefore build on these integrated capabilities rather than reopening completed Phase 2 foundations. Autonomous runtime scheduling, higher-level operation orchestration, broader consensus protocols, and real-network deployment remain explicit future projects.

See `stability-checkpoint.md` for the replay and artifact compatibility contract.
