# Raft membership transitions

Membership changes operate over pre-provisioned transport nodes. The initial stable voter set may exclude learners.

`VotingConfiguration` is the quorum oracle. Stable configurations require one majority; joint configurations require independent majorities of both the old and new voter sets. The same oracle is used for elections and membership-aware commit advancement.

## Replicated stable-to-joint transition

`ReplicatedMembershipTransition` makes the first transition boundary durable instead of mutating quorum state immediately:

1. the current leader appends an immutable `JointConsensusCommand` as a normal current-term Raft log entry;
2. while that entry is uncommitted, the existing stable voter configuration remains authoritative;
3. `MembershipAwareLeaderReplicator` replicates and commits the entry using the current configuration;
4. only after the exact proposal index is committed does the cluster activate joint old/new quorum semantics.

A second proposal is rejected while one is pending, and a stale or replaced leader cannot activate its proposal. The transition records deterministic proposal and commit trace events so failures remain replayable.

This is intentionally not yet the complete membership-change protocol. Finalization into the new stable configuration is still an in-memory operation; a later slice must represent that final configuration as a replicated Raft entry and define restart/recovery of configuration state from durable history or snapshots.
