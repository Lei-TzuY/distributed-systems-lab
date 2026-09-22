# Single-decree Paxos

This Phase 3 slice adds a second executable consensus protocol without changing the
original Raft replay/checkpoint boundary.

## Scope

The implementation covers one Paxos decree over the deterministic simulator:

- globally unique proposal numbers `(round, proposer_id)`;
- durable monotonic proposer rounds so a crashed proposer cannot reuse an old number;
- durable acceptor promises and accepted values;
- Phase 1 prepare/promise quorum;
- Phase 2 accept/accepted quorum;
- value adoption from the highest accepted proposal observed in a promise quorum;
- learner notifications after objective majority acceptance;
- crash/restart recovery of acceptor and learner state;
- deterministic drop/delay/duplicate fault behavior through the existing simulator;
- a safety harness that independently reconstructs chosen values from durable accept
  histories instead of trusting proposer announcements.

## Safety boundary

A value is objectively chosen only after one proposal/value pair appears in the durable
accept histories of a majority of acceptors. `PaxosSafetyHarness` reconstructs those
majorities and rejects:

- two different values chosen by different quorums;
- one proposal number accepted with conflicting values;
- learned decisions without matching durable majority acceptance;
- divergence between objective chosen evidence and learner state.

Proposal state is intentionally volatile. Promises, accepted state, accept history,
learned decisions, and each proposer's last allocated round survive crash/restart.

## Deliberate limits

This is single-decree Paxos, not Multi-Paxos. It does not add leader leases, log slots,
configuration changes, batching, or production networking. Those require separate
vertical slices with their own safety evidence.
