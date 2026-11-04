# Application-aware Multi-Paxos campaign (opt-in)

The original `MultiPaxosScenarioRunner` and v1 `MultiPaxosTrialArtifact`
remain **unchanged** (consensus-only, learner disabled). This feature adds a
separate, explicit deterministic runner:

- `MultiPaxosApplicationScenarioRunner` uses the same persisted
  `SeededMultiPaxosActionSchedule`, `SeededFaultSchedule`, and
  `SeededLifecycleSchedule`.
- `MultiPaxosApplicationArtifact` persists these complete inputs together
  with the output (actual ordered applied slots, reconstructed node KV,
  completed and pending client identities/results, oracle result, protocol
  and application trace). `replay()` rejects any byte-different output.
- Each `PROPOSE` action is an actual `MultiPaxosRequest`, with stable
  identity derived from its schedule ID and index. `PREPARE` remains an
  ordinary protocol Phase 1. No globally chosen payload is supplied to a
  client or a peer.
- Four explicit operation encodings are supported:
  `put:key:value`, `delete:key`, `get:key`,
  `append:key:suffix`. Values may be empty for Put/Append.
- The driver periodically pumps scheduled messages and calls
  `MultiPaxosStateMachine.apply_ready()` on live nodes. Thus it exercises
  normal peer-local recovery, ordered durable application and provenance
  checks. It polls the real `MultiPaxosClient` for client-visible results,
  using the normal authority and matching-applied-record gate.
- `withheld_acks` preserves selected invocations as pending even after
  application; no hypothetical completion is written to their observed
  history. The opt-in `MultiKeyKVLinearizabilityChecker` checks independent
  Put/Delete/Get operations across multiple keys, and is **not used** for
  Append, which is outside its sequential specification.
- Real no-quorum runs are **incomplete**, never counted as safety violations
  or claimed successful acknowledgements.
- Execution is bounded by per-action `events_after`,
  `final_event_budget`, `application_rounds` and
  `events_per_round`. These are simulator event budgets; no timer or
  hidden randomness. Bounded exhaustion provides no liveness guarantee.

## Example

```python
from distlab.multipaxos_application_campaign import (
    MultiPaxosApplicationScenarioRunner,
    MultiPaxosApplicationArtifact,
)
from distlab.multipaxos_campaign import (
    MultiPaxosAction, MultiPaxosActionKind, SeededMultiPaxosActionSchedule,
)
from distlab.randomized_faults import SeededFaultSchedule

actions = SeededMultiPaxosActionSchedule(7, (
    MultiPaxosAction("lead", MultiPaxosActionKind.PREPARE, "n1", events_after=20),
    MultiPaxosAction("a-put", MultiPaxosActionKind.PROPOSE, "n1",
                     slot=1, value="put:k:v", events_after=30),
    MultiPaxosAction("b-get", MultiPaxosActionKind.PROPOSE, "n1",
                     slot=2, value="get:k", events_after=30),
))
runner = MultiPaxosApplicationScenarioRunner(
    actions, SeededFaultSchedule(7, ()), withheld_acks=("a-put",)
)
result = runner.run()
artifact = MultiPaxosApplicationArtifact.capture(runner, result)
assert MultiPaxosApplicationArtifact.from_json(artifact.to_json()).replay() == result
assert result.completed == (("b-get", "v"),)
assert result.pending == ("a-put",)
assert result.linearizable is True
```

## Multi-key completion-extension oracle (opt-in)

Each Put, Delete, and Get touches exactly one key. Linearizability is local to
independent objects: the combined history admits a legal completion extension
if and only if every key's projected observed history admits one. The
`MultiKeyKVLinearizabilityChecker` returns deterministic, sorted per-key
`CompletionExtensionResult` witnesses, **not** a fabricated global order or
a claim that pending writes were durably chosen/acknowledged. Only actual
invocations and responses are projected, preserving within-key real-time
precedence and original pending/observed status. Original `OperationHistory`
is never mutated. A failure on any key produces an overall negative verdict;
resource exhaustion raises `LinearizabilitySearchExhausted`, never `False`.
Search limits bound total operations, key count, and states per key.

Example: `put:a:alpha` and `put:b:beta`, then `get:a` returns `alpha` and
`get:b` returns `beta`. A lost acknowledgement on one key may remain pending
while reads and writes on the other key complete. Legacy single-key checker
and completed-only `check()` behavior remain unchanged; existing application
artifact schema and single-key serialized bytes are unchanged.

**Not supported:** multi-key atomic transactions, cross-key invariant
specifications, multi-object command atomicity, and a global witness order.
Append remains outside the Put/Delete/Get linearizability oracle.

## Full Simulator reconstruction checkpoint (opt-in)

Set `reconstruction_boundary=N` on `MultiPaxosApplicationScenarioRunner`
to simulate one **entire-Simulator replacement** immediately before action
index `N` (zero-based). The allowed range is `0..len(actions.actions)`.
This differs from an ordinary node crash/restart: **all** old volatile leader
authority, transport queues, recovery timers, link partitions/delays and
message ordinals are discarded. Each fresh simulator receives the same
configured fault rules, with fresh per-epoch message counters.

Only a **deep copy of the old simulator's durable server state** transfers
to the new `Simulator`. A separate deep copy of the **client-owned
`ClientJournal`** keeps invocation identities, pending requests and already
observed completions. Neither invented responses nor volatile server
authority transfer across that boundary. Rebuilt replicas reconstruct their
applied log prefix, dedup table and KV from durable evidence. A subsequent
`PREPARE` action is required to establish fresh Phase-1 leader authority
before any new request can complete.

When enabled, the trace contains deterministic `phase=stop` and `phase=start`
`multipaxos-full-reconstruction` markers, with the boundary index, separating
the two simulator epochs. Trace times restart at zero **after** the marker;
they are epoch-local, not one continuous monotonic clock. All original
pre-reconstruction trace records are retained in order.

`MultiPaxosApplicationArtifact` stores the opt-in
`reconstruction_boundary` only when enabled, and its full canonical
`capture -> to_json -> from_json -> replay` round-trip verifies the
reconstruction, real post-restart client results, pending history and trace.
**With no checkpoint, the optional field is omitted**: all previously
captured v1 application artifacts and legacy consensus-only artifacts retain
their exact serialized bytes and replay meaning.

This is **not** a snapshot/compaction protocol, automatic server leader
election, client-journal durable storage implementation, or network service.
It models the external client retaining its journal while the server process
and volatile simulation are replaced. Pending requests may remain
unresolved until the schedule provides suitable leader authority and
transport. The runner supports one full reconstruction boundary per run.

## Nullable wildcard fault selectors

The opt-in application artifact decoder supports native `FaultRule` wildcard
selectors: `src`, `dst`, `ordinal`, and `payload_type` may be `null`.
A real dropped `LeaderDecision` with `ordinal: null` survives
`capture ¡÷ to_json ¡÷ from_json ¡÷ replay` with identical evidence bytes.
Invalid selectors, non-positive ordinals, and negative delays are rejected.
The legacy `SeededFaultSchedule.from_json()` decoder and
`MultiPaxosTrialArtifact` remain unchanged.

## Evidence and limits

`tests/test_multipaxos_application_campaign.py` covers live Put/Get,
lost acknowledgement, follower recovery after a dropped LeaderDecision,
crash/restart lifecycle, bounded no-quorum, deterministic seeded schedules,
negative input validation, Append's oracle exclusion and tamper-detecting
self-contained artifact replay. The unchanged legacy `MultiPaxosTrialArtifact`
has no new fields, schedules or trace events.

This is a deterministic laboratory, not production networking. The runner
does not automatically discover new leaders or retry a request on another
leader; schedules must express prepares and the service must have authority.
It does not include atomic multi-key transaction checking, automatic
client-journal recovery from server storage, or membership changes.
