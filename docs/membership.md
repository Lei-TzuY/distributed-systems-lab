# Raft membership transitions

Membership changes operate over pre-provisioned transport nodes. The initial stable voter set may exclude learners.

`VotingConfiguration` is the quorum oracle. Stable configurations require one majority; joint configurations require independent majorities of both the old and new voter sets. The same oracle is used for elections and membership-aware commit advancement.

## Replicated stable-to-joint transition

`ReplicatedMembershipTransition` makes the first transition boundary durable instead of mutating quorum state immediately:

1. the current leader appends an immutable `JointConsensusCommand` as a normal current-term Raft log entry;
2. while that entry is uncommitted, the existing stable voter configuration remains authoritative;
3. `MembershipAwareLeaderReplicator` replicates and commits the entry using the current configuration;
4. only after the exact proposal index is committed does the cluster activate joint old/new quorum semantics.

## Replicated joint-to-stable finalization

Finalization follows the same commit-before-activation rule:

1. while joint consensus is active, the current leader appends `StableConsensusCommand` containing the exact new voter set;
2. the cluster remains joint while the finalization entry is uncommitted, so election and commit decisions still require independent old/new majorities;
3. the finalization entry itself must commit under that joint quorum;
4. only after the exact finalization index commits does the cluster install the new stable voter set and fence removed voters.

Only one membership command may be pending at a time. A stale or replaced leader cannot apply a pending command, and finalization is rejected unless the leader belongs to the new voter set. Proposal, commit, and finalization events are recorded in the deterministic trace.

This is still intentionally short of full restart-safe dynamic membership. The committed commands are durable Raft history, but a restarted `ReconfigurableRaftCluster` does not yet reconstruct its active voting configuration by replaying committed membership entries or a membership-bearing snapshot. That recovery rule is a separate correctness slice.
