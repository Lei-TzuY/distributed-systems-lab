# Raft membership transitions

Membership changes operate over pre-provisioned transport nodes. The initial stable voter set may exclude learners.

`VotingConfiguration` is the quorum oracle. Stable configurations require one majority; joint configurations require independent majorities of both the old and new voter sets. The same oracle is used for elections and membership-aware commit advancement.

## Replicated stable-to-joint transition

`ReplicatedMembershipTransition` makes the first transition boundary durable instead of mutating quorum state immediately:

1. the current leader appends an immutable `JointConsensusCommand` as a normal current-term Raft log entry;
2. while that entry is uncommitted, the existing stable voter configuration remains authoritative for elections and live membership state;
3. commit of the joint proposal, or of any later entry whose commit would cover it, requires independent majorities of the current and proposed new voter sets;
4. when that exact proposal index commits, the validated joint configuration becomes live synchronously at the same commit boundary; controller observation only clears proposal-local state.

## Replicated joint-to-stable finalization

Finalization follows the same commit-before-activation rule:

1. while joint consensus is active, the current leader appends `StableConsensusCommand` containing the exact new voter set;
2. the cluster remains joint while the finalization entry is uncommitted, so election and commit decisions still require independent old/new majorities;
3. the finalization entry itself must commit under that joint quorum;
4. when the exact finalization index commits, the new stable voter set is installed synchronously and removed voters are fenced before any later election can observe stale quorum state.

Only one membership command may be pending at a time. Commit advancement fails closed before changing `commit_index` or durable membership watermarks if a candidate covers multiple uncommitted membership commands, references unknown nodes, starts a second joint configuration, finalizes without active joint consensus, or finalizes to a voter set other than the active joint new-voter set. A stale or replaced leader cannot apply a pending command, and finalization is rejected unless the leader belongs to the new voter set. Proposal, commit, and finalization events are recorded in the deterministic trace.

## Leadership handoff

`ReplicatedMembershipTransition` is bound to the exact leader generation that created it. A retired generation cannot continue proposal replication or activation even if the same node later becomes leader again.

When leadership changes with one uncommitted membership command still present in the new leader's log, constructing a new transition controller reconstructs that pending index from durable membership watermarks plus the retained log. A second membership proposal is rejected while that recovered command is pending. If the recovered command was appended in an older term and no later current-term entry exists, the new leader appends a current-term commit-recovery barrier before replication so Raft can commit the older membership prefix without violating the current-term commit rule.

Objectively committed membership is not dependent on controller handoff: membership-aware commit advancement persists the watermark and installs the live stable/joint configuration synchronously. This closes the window where a leadership change or election could otherwise use a quorum older than the durable committed membership state.

## Restart and snapshot recovery

Committed membership activation persists a per-replica membership commit watermark. Cluster recreation replays only membership commands at or below the highest durable watermark, so later uncommitted proposals cannot become authoritative after restart.

Log compaction preserves that evidence in `KVSnapshot`: snapshots created by a `ReconfigurableRaftCluster` carry the stable or joint voting configuration effective at the snapshot boundary together with the membership commit index it represents. Recovery validates the snapshot against the compacted Raft index/term, uses its configuration as the prefix state, and replays only retained committed membership entries after the boundary. Missing, contradictory, or unknown-node membership metadata fails closed instead of guessing a quorum.

This keeps membership recovery aligned with the same durable snapshot boundary used by Raft and the replicated state machine while ordinary non-reconfigurable KV snapshots remain unchanged.
