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

## Seeded campaign and exact replay

Single-decree Paxos now participates in the repository's deterministic campaign
model. Proposal choices, message faults, and node lifecycle transitions are
compiled from seeds into explicit schedules before execution. The runner never
consults randomness.

`PaxosTrialArtifact` persists:

- the explicit proposal schedule;
- the exact message-fault schedule;
- the exact crash/restart lifecycle schedule;
- node membership and the final logical event budget;
- the classified outcome (`chosen`, `incomplete`, or `safety_violation`);
- objective chosen evidence when present;
- any safety violation text;
- the complete structured simulator trace.

Replay reconstructs the scenario only from persisted schedules and requires the
outcome, chosen evidence, violation classification, and entire trace to match.
`incomplete` is intentionally not a safety failure: faults may prevent a quorum
without violating Paxos safety. `SeededPaxosCampaign` stops early only on an
actual `PaxosSafetyViolation`.

## Multi-decree replicated-log foundation

`PaxosLogCluster` composes independent Paxos decrees into positive integer log slots.
Each acceptor persists promised proposal, accepted value, and accept history per slot;
learned decisions are also durable per slot.

A slot may become chosen independently of earlier slots, but externally consumable
`chosen_prefix()` / `learned_prefix()` stop at the first missing slot. This preserves
ordered log application while allowing consensus work for later slots to proceed.

`PaxosLogSafetyHarness` independently reconstructs majority evidence per slot and
rejects conflicting chosen values, forged learned decisions without quorum evidence,
or divergent learned prefixes. Crash/restart preserves slot promises, accepted state,
learned gaps, and the proposer's durable round allocator.

The slot-indexed implementation remains useful as the unamortized correctness
baseline. Stable-leader Multi-Paxos now builds on it with one Phase-1 quorum per
leader ballot and repeated Phase-2 proposals across slots.

## Stable-leader Multi-Paxos

`MultiPaxosCluster` provides durable global ballot promises, Phase-1 leader
authority, repeated Phase-2 proposals across slots, higher-ballot takeover with
accepted-value adoption, and durable chosen reconstruction. Crash/restart discards
volatile leader authority while retaining the ballot and accepted history needed for
safe takeover.

The Multi-Paxos path now also participates in seeded deterministic exploration.
`SeededMultiPaxosActionGenerator` compiles leadership-prepare and slot-proposal
actions before execution; message faults and lifecycle actions are persisted as
explicit schedules. `MultiPaxosTrialArtifact` stores the classified outcome,
sorted chosen-slot evidence, any safety violation, and the complete structured trace.

Exact replay requires all of those outputs to match. A run that loses quorum or
attempts a proposal without active Phase-1 authority is `incomplete`, not a safety
failure. Only `PaxosSafetyViolation` terminates a campaign as a correctness failure.

This layer still does not provide ordered state-machine application, client semantics,
membership changes, batching, or production networking. Those remain separate
cross-layer milestones.
