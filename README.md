# distributed-systems-lab

A deterministic distributed-systems correctness laboratory for replicated state machines, consensus, failure semantics, durability, and linearizability.

The project starts with a deterministic simulation kernel before implementing Raft. Correctness, reproducibility, and executable invariants take priority over protocol breadth or networking realism.

## Initial architecture

The first milestone provides a deterministic event-driven simulator with:

- logical time
- stable event ordering
- explicit message envelopes
- deterministic drop, delay, and duplicate fault schedules
- crash/restart state hooks
- structured execution traces
- scenario replay suitable for regression tests

Seeded randomized fault generation is intentionally deferred until it can emit a persisted, replayable fault schedule.

Raft election and log replication will be layered on top only after the simulator has stable reproducibility tests.

## Scope discipline

This repository intentionally does not implement Paxos, Zab, SWIM, gossip, or a large real-network deployment while the Raft correctness harness is immature.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

See `docs/architecture.md` for design invariants once the simulator foundation lands.
