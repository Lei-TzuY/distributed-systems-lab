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

This is a correctness laboratory, not a production distributed database. The original correctness checkpoint remains the compatibility boundary for deterministic replay, failure artifacts, and minimization, but the repository has now promoted beyond that checkpoint through bounded Phase 2 projects.

Integrated Phase 2 capabilities include snapshot/install-snapshot and compaction, membership changes with joint consensus, leadership transfer, pre-vote, CheckQuorum, bounded heartbeat control, per-node leader runtime generations, and generation-fenced KV authority. The deterministic campaign path now validates leader-generation authority without changing its replayable message schedule.

Phase 3 now includes crash-recoverable single-decree Paxos, a slot-indexed multi-decree Paxos log, stable-leader Multi-Paxos with Phase-1 amortization and takeover recovery, plus seeded fault/lifecycle exact-replay campaigns for the Paxos paths. Ordered Multi-Paxos state-machine application, real-network deployment, and production storage/networking remain future architectural phases rather than filler follow-ons.

See [`docs/architecture.md`](docs/architecture.md) for invariants and [`docs/stability-checkpoint.md`](docs/stability-checkpoint.md) for the compatibility boundary.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

A failing randomized/campaign case must remain reproducible from persisted schedules and artifacts. Flaky tests are treated as harness defects.
