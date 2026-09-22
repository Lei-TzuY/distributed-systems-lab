# Generation-fenced leader KV service

`LeaderKVService` binds client-visible KV work to one explicit
`LeaderRuntimeIdentity.generation`. It composes existing Raft replication,
membership-aware commit rules, linearizable read barriers, KV application, and
client history without taking ownership of heartbeat scheduling.

## Authority contract

A client operation validates the requested leader runtime generation before any
client-history or Raft mutation. The same generation is checked again after
network/replication work and immediately before publishing a client response.

This closes the gap between protocol-level stale-term checks and client-visible
authority:

- a retired leader generation cannot append a new client request;
- a write that commits but loses its generation before response remains pending
  instead of fabricating success;
- the exact pending `ClientRequest` can be retried on the next leader without
  creating a second client-visible invocation;
- stale reads are rejected before history mutation, and a read that loses
  authority before response remains incomplete evidence.

## Writes

A new write is recorded in `KVClientHistory`, appended to the current leader,
replicated under the active commit rule, applied locally only after commit, and
responded to only while the same runtime generation is still active.

Exact retries reuse the pending request identity. If that request already exists
in an older-term log after leadership transfer, the service appends a
`CommitRecoveryBarrier` in the new term and commits that barrier so the older
request becomes committed as part of the same prefix.

`ReconfigurableRaftCluster` automatically uses
`MembershipAwareLeaderReplicator`, so stable configurations ignore learners and
joint consensus requires independent old/new voter majorities.

## Linearizable reads

Reads first create a client-history invocation. If the leader has no committed
entry in its current term, the service commits a current-term recovery barrier
before constructing the normal `LinearizableKVReader` quorum barrier. The client
response is recorded only after the expected runtime generation is revalidated.

Failed reads remain pending and may be retried with the same operation id on a
later leader generation.

## Runtime separation

The service consumes only the public `LeaderRuntimeSupervisor.runtime_identity`
surface. It does not schedule heartbeat ticks or mutate runtime ownership, so
leader control-plane scheduling and client data-plane fencing remain separate
architectural responsibilities.

## Client-session orchestration

`KVClientSession` can now execute writes, exact retries, linearizable reads,
read retries, and recovery through `LeaderKVService` while retaining ownership
of client program order and request sequencing.

- the session checks single-flight and monotonic request-id rules before service
  execution;
- the service owns leader-generation, quorum, commit, application, and read
  barrier authority;
- failed writes remain session-pending only when the shared client history
  contains the matching unresolved request;
- successful service writes advance the session floor without emitting a
  duplicate client response;
- failed reads preserve the existing pending/abandonment model and can be
  retried against a later valid generation;
- generation-fenced recovery first confirms leader authority and quorum, then
  reconstructs the request-id floor and unresolved operation from durable
  leader state.

This keeps session sequencing and Raft authority as separate responsibilities
while providing one executable path that composes both layers.

## Membership-transition data-plane integration

Client traffic and membership changes share one Raft log. Reconfigurable leader
service fanout therefore targets all pre-provisioned transport peers, including
learners, while `MembershipAwareLeaderReplicator` remains the only source of
commit authority.

This distinction is required when a client entry follows an uncommitted joint
membership proposal: proposed new voters must receive that prefix before they
can contribute the new-configuration majority required to commit it. Merely
fanning out to the currently live stable voters would deadlock the transition
behind otherwise healthy learners.

Committed membership commands remain part of the durable applied Raft prefix and
therefore still participate in State Machine Safety and snapshot boundaries. For
the KV projection specifically, `JointConsensusCommand` and
`StableConsensusCommand` are deterministic no-ops: they alter voting authority,
not key/value or client-deduplication state.

Thus transport fanout, commit quorum, and KV projection remain separate concerns:

- fanout may include learners;
- commit requires the exact stable/joint voter quorum for the candidate prefix;
- membership commands are durably applied control-plane entries;
- KV state changes only for KV commands and commit-recovery barriers remain no-ops.
