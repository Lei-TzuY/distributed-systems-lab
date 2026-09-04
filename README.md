# distributed-systems-lab

A deterministic distributed-systems correctness laboratory for replicated state machines, consensus, failure semantics, durability, and linearizability.

The repository now has a complete first correctness vertical slice rather than only the original simulator bootstrap. The stable checkpoint is:

```text
deterministic simulator
        -> Raft election/log replication
        -> persistence and crash/restart recovery
        -> replicated KV + client request deduplication
        -> client histories + linearizability checking
        -> seeded workload/fault/lifecycle campaigns
        -> exact replay + deterministic failure minimization
```

## Current capabilities

- deterministic logical-time event simulator with stable ordering and structured traces
- explicit drop/delay/duplicate message fault plans and persisted seeded schedules
- replayable crash/restart lifecycle schedules with persistent-vs-volatile state boundaries
- Raft election, RequestVote, AppendEntries, log backtracking, commit advancement, and restart recovery
- executable Election Safety, Leader Append-Only, Log Matching, Leader Completeness, and State Machine Safety checks
- deterministic replicated key-value state machine with client request deduplication and recovery
- client operation histories and a single-key KV linearizability checker
- seeded client workload, message-fault, and lifecycle campaigns
- versioned exact-replay failure artifacts
- deterministic 1-minimal reduction of failing histories, workloads, fault schedules, and lifecycle schedules

## Checkpoint scope

This is a correctness laboratory, not a production distributed database. The first checkpoint deliberately stops before new architectural phases such as snapshot/install-snapshot, log compaction, membership changes, additional consensus protocols, or real-network deployment.

Those are Phase 2 work and should only begin as explicit bounded projects; they are not automatic follow-ons merely to create repository activity.

See [`docs/architecture.md`](docs/architecture.md) for invariants and [`docs/stability-checkpoint.md`](docs/stability-checkpoint.md) for the maintenance boundary.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

A failing randomized/campaign case must remain reproducible from persisted schedules and artifacts. Flaky tests are treated as harness defects.
