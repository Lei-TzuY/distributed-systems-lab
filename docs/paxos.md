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

## Ordered Multi-Paxos application

`MultiPaxosCluster(..., learn_decisions=True)` adds the learner role: the leader
that observes a Phase-2 quorum durably records the chosen slot and broadcasts a
`LeaderDecision`. Each node keeps a durable, possibly gapped, learned-decision map.
Decision envelopes must come from the member that owns the decision ballot. A
conflicting decision for an already-learned slot raises `PaxosSafetyViolation`.
Learning is off by default, so existing Multi-Paxos campaign traces, event budgets,
and artifacts are byte-identical to the pre-application protocol.

`MultiPaxosStateMachine` applies learned decisions to a small deterministic KV
machine (`Put`/`Delete`, plus the non-idempotent `Append` so a double application
is visible in state; any other chosen value is an ordered no-op):

- `apply_ready(node)` applies slots `last_applied + 1, ...` while each is learned
  and stops at the first gap (`multipaxos-apply-gap`). Later learned slots wait.
- Each applied slot must match the node's durable learned decision and the cluster's
  objective chosen evidence, so an unchosen or forged decision is never applied.
- The applied prefix is durable and written in the same simulator step that applies
  the slot. KV state is volatile and is rebuilt by replaying the durable prefix
  after a crash or full reconstruction, so no slot is applied twice.
- Reconstruction rejects applied histories that repeat or skip a slot, contain a
  slot the node never durably learned, or apply a value without durable quorum
  evidence. It also rejects learned decisions without durable quorum evidence and
  in-memory state that diverges from the durable prefix.

Leader handoff needs no special application logic: a higher-ballot leader adopts the
chosen value through Phase 1, re-chooses it, and its decision fills gaps on followers.
Followers compare applied slots by value, since the same value can be learned
under different ballots.

## Automatic follower catch-up

With learning enabled, each normal `apply_ready(node)` pass applies the ready
prefix and starts recovery at `last_applied + 1`. A later learned slot exposes a
gap through `blocked_on(node)` as before. Probing the next window also discovers
a lost suffix when a partition hid *all* later decisions. The caller never needs
to name or re-propose a missing slot.

`DecisionRecoveryRequest(start_slot)` asks each peer, in sorted node order, for
its own durable learned records in `[start_slot, start_slot + 16)`.
`DecisionRecoveryResponse` contains at most 16 records in ascending slot order;
holes are omitted. The bound is on record count, not serialized bytes (values in
this laboratory are arbitrary Python objects). A peer must not fill a response
from the cluster's global chosen map or another peer's state.

Recovery accepts only member-peer envelopes, positive integer slots (not booleans),
valid member ballots, and tuples of unique, ordered records within that window.
The entire response is validated before any record is learned. As a correctness
harness check, each record must match durable acceptance by a distinct majority
at its stated ballot. A conflicting or unproven value raises
`PaxosSafetyViolation`; malformed messages are ignored. Recovery neither grants
leader authority nor changes acceptor promises or accepted values. The global
evidence check validates a value received over the transport; it never supplies
the value for a peer's response.

Recovered decisions enter the existing durable learner, and the next normal
`apply_ready` pass applies them using the same contiguous-prefix and chosen-value
checks. Duplicate, delayed, and old-leader responses remain safe without a matching
volatile request. In particular, an `Append` is applied once, and an already
applied record is never rewritten. Crash clears the active recovery episode and
retires its local timers; learned decisions and the applied prefix remain durable.
After restart or full reconstruction, the next application pass rebuilds KV from
that prefix and requests the next missing window.

### Bounded progress and liveness boundary

One application pass starts at most one recovery episode per node. Repeated
passes for the same active window coalesce. An episode sends at most three
requests per peer, four simulator ticks apart, using the existing transport and
its drop/delay/duplicate/partition rules. Local retry timers use the simulator's
existing deterministic scheduling mechanism; no wall-clock timer or service is
introduced. Once the first requested slot is learned, the episode stops. Later
application passes advance through further windows. After three unsuccessful
attempts, it emits `multipaxos-recovery-exhausted` with the starting slot, attempt
count, and whether a later learned decision proves a gap. It schedules no further
work. A subsequent application pass can retry, including after a long partition.

For any finite chosen prefix, convergence is guaranteed **provided** the normal
application driver keeps polling `apply_ready` on the relevant live nodes and
processing scheduled simulator work, communication is restored, some reachable
peer durably knows each required decision, and requests and their valid replies
are eventually delivered. Evidence may come from different peers. Responses
arriving after an episode expires are still accepted. Each successful missing
slot extends the learned prefix; application then monotonically extends the
durable applied prefix. There is no additional consensus round or re-proposal.

Calling `sim.run()` after just one application pass drains only a finite episode;
it does **not** promise recovery after arbitrary subsequent loss or partition.
Continued application polling is the retry trigger after exhaustion, and also
discovers new tail decisions. Permanent partitions, perpetual loss, missing
peer-local learned evidence, or genuinely unchosen holes remain blocked. A
globally reconstructed chosen value alone is insufficient to serve recovery.

`tests/test_multipaxos_catchup.py` drives a real three-node partition/heal scenario,
multiple windows, lost requests/replies, reordered and duplicate replies, takeover,
crash/restart and full reconstruction, malformed/conflicting claims, unavailable
evidence, and seeded fault schedules with exact trace/state replay. A deliberate
test mutant that trusts a peer claim is caught by the application safety invariant.
Learning remains disabled by default: no recovery work, durable state, or messages are
added to existing learner-disabled campaign traces or replay artifacts.

Applied-prefix compaction is not implemented, and the existing Multi-Paxos
campaign runner does not yet drive application. Recovery
fault campaigns live in the focused tests; its existing artifact format is unchanged.

Ordinary decision/application ballot provenance is not validated. Applied and learned
records are checked by slot and value only. Their `proposal` field is not compared
with the node's learned ballot
or with a ballot that formed the slot's durable quorum. A corrupted record that keeps
the chosen value under a fabricated member ballot is therefore accepted on recovery.
Applied state is unaffected, because application depends only on slot order and
value, but the recorded provenance cannot be trusted. Proposed follow-up: require an
applied record's ballot to equal the learned one, and a learned ballot to be a quorum
ballot for that slot and value, each with a focused corruption test. Recovery messages
do require the stated ballot to have durable quorum evidence; this does not repair
pre-existing ordinary learned or applied provenance.

## Multi-Paxos client-visible KV

`MultiPaxosClient` is a small simulator adapter over the existing protocol and
`MultiPaxosStateMachine`, with `learn_decisions=True`. It accepts the existing
`Put`/`Delete`/`Get` operation types:

```python
from distlab.kv import Put
from distlab.linearizability import Get
from distlab.multipaxos_client import ClientStatus, MultiPaxosClient
from distlab.multipaxos_state_machine import MultiPaxosRequest

client = MultiPaxosClient(machine)  # The cluster already has Phase-1 leadership.
write = MultiPaxosRequest("A", 1, "A-put-1", Put("k", "v"))
written = client.execute("n1", write)
read = MultiPaxosRequest("B", 1, "B-get-1", Get("k"))
answer = client.execute("n1", read)
# Check answer.status == ClientStatus.COMPLETED before using answer.value.
```

Every `(client_id, request_id)` represents one immutable request, including its
`operation_id`. Retrying the same request uses `submit` or `execute` again and
does not add another invocation to the client journal's `OperationHistory`.
Reusing the request identity with different contents, changing its history ID,
or reusing a history ID for another request is a conflict. Malformed request
IDs/operations raise validation errors before submission. This API does not
impose Raft's monotonic request sequencing or single-flight session policy.

### Durable deduplication and reads

The request envelope itself is a chosen log value. During ordered application,
the first occurrence records `(request, first_slot, result)` in the volatile
projection. Later identical occurrences consume their log slots but do not
execute the operation again. Conflicting occurrences raise
`ClientRequestConflict` before modifying KV or advancing the offending applied
slot. Existing unwrapped Put/Delete/Append/None behavior is preserved.

There is no separately trusted durable dedup table: the existing durable applied
prefix is the source of KV, identities, and results. Crash/restart or a new
Simulator/cluster/state machine deterministically replays that prefix. Get is a
no-write log command; it captures the value exactly when its first slot applies,
and replay recovers that same result. A duplicate Get after later writes returns
the original value, including an original missing-key `None` result.

`Append` may also be wrapped to exercise non-idempotent exactly-once effects.
It is intentionally excluded from OperationHistory, whose checker supports only
Put/Delete/Get. The tests also interleave writes before replaying a duplicate Put,
so idempotence cannot hide a missing deduplication check.

### Submission, acknowledgement, and takeover

`submit` only proposes/retransmits. `poll` observes the exact registered request;
by default it applies ready decisions and acknowledges a matching applied
record. `poll(apply=False)` exposes the learning/application gap, and
`poll(acknowledge=False)` models application with a lost acknowledgement.

| Status | Meaning |
| --- | --- |
| submitted | A proposal was issued; neither choice nor application is implied. |
| pending | Matching completion evidence is not yet available. |
| chosen_not_applied | The local learner knows this request, but a prefix gap may block it. |
| applied_not_acknowledged | Matching durable application exists; no history response was published. |
| completed | Matching application evidence and a client response exist. |
| conflict | An identity was reused with incompatible contents. |
| leadership_unavailable | The node is crashed, lacks Phase-1 quorum authority, has a newer local promise, or fails the caller's ballot fence. |
| quorum_unavailable | The finite execution budget expired without a matching learned/applied result; the request remains unresolved. |

Success never follows merely from `propose()` or from completion of a proposed
slot. The applied request must match all identity/content fields. If Phase 1
adopts B in the slot where a caller hinted A, A remains pending and a retry can
place A in another slot. `expected_ballot` on `poll` fences a stale submission
ticket. Current local Phase-1 authority is required before publishing a response;
a delayed old-leader call cannot bypass that gate using a slot number.
Results distinguish `ballot` (the applied decision's evidence, or the submission
ballot before application) from `authority_ballot` (the responder's current live
Phase-1 ballot). After takeover, a valid response can cite an older applied
decision while being authorized by the new leader's ballot.

New requests use a slot after the highest locally known accepted/learned slot,
including records transported in Phase-1 promises. This is essential for a new
leader's Get: a previously completed write may be absent from its own KV and log
but present in its promise quorum. The driver repairs earlier slots through
ordinary Phase 2, adopting highest accepted records (including accepted None).
Only a genuinely unaccepted hole receives a no-op. Existing chosen-gap recovery
continues to use peer-local durable decisions. No global chosen map supplies a
placement, authority decision, request payload, or read result; its existing use
inside application/quorum safety validation remains unchanged.

### Bounded execution and recovery contract

`execute` defaults to eight rounds, at most 64 queued events and 16 prefix repair
slots per round. It returns an observable status when this budget is exhausted;
`quorum_unavailable` does not prove failure or non-choice. Exact retry is required
when a delayed result is unknown. Larger gaps may need more rounds or later calls.
The adapter does not elect a leader, implement leases, or guarantee completion
under permanent partition/loss or continued leader changes.

The `ClientJournal` belongs to the client and retains request/history identities,
including incomplete reads. Retain it (or a deepcopy) when rebuilding the server
runtime to preserve the same history. Server dedup/results always reconstruct
from server applied decisions and never trust the journal as completion evidence.
Loss of the client's own journal does not undo server deduplication, but cannot
recreate client invocations that were never decided. Client state is not a
production persistence/RPC format.

The existing `check(history)` keeps its completed-only projection for campaign
compatibility. Opt in to `check_with_pending_extensions(history)` to check reads
that observe a pending lost-Ack write. Its witness may include that write without
adding a Completion or implying it was chosen. The client Ack gate is unchanged;
only a later actual retry can publish the missing acknowledgement. See
[`linearizability.md`](linearizability.md) for formal semantics, budgets, witness
replay, and the unsupported extension/minimizer combination.

Run `python -m distlab.multipaxos_demo` for concrete slot/ballot/request/status,
chosen/applied, replica KV/dedup, and linearizability evidence. It exercises two
clients, lost acknowledgement, leader crash/takeover, follower catch-up, a duplicate
request after an interleaved write, and full reconstruction. Tests additionally
replay three fixed-seed fault schedules, cover three-window prefix repair, and
deliberately mutate dedup/read-result handling to verify the safety detectors.

Membership changes, compaction, batching, and production networking remain outside
this client milestone.
