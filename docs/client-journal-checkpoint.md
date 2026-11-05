# Portable ClientJournal checkpoints (v1)

`ClientJournal.to_json()` exports client-owned retry identity and observations as
canonical JSON. `ClientJournal.from_json(encoded)` returns a new journal, new
history, requests and operation instances. Malformed/inconsistent checkpoints
raise `ValueError`. Neither method reads server state or changes the original.

```python
encoded = client.journal.to_json()
restored_journal = ClientJournal.from_json(encoded)
restored_client = MultiPaxosClient(rebuilt_machine, restored_journal)
```

Rebuilt servers still require fresh Phase-1 authority and matching local durable
applied-request evidence before successful acknowledgement. Server dedup/results
come only from the durable applied prefix, never from this checkpoint.

## Exact schema

Every object has exactly the documented fields. Unknown/missing fields, duplicate
JSON member names, NaN/Infinity and unsupported versions fail closed. Object keys
use the existing compact, sorted `canonical_json` encoder. Input whitespace and
object-field ordering need not be canonical; exporting normalizes them.

```json
{
  "version": 1,
  "requests": [
    {"client_id":"A","request_id":1,"operation_id":"put-a",
     "operation":{"type":"put","key":"a","value":"alpha"}},
    {"client_id":"B","request_id":1,"operation_id":"get-a",
     "operation":{"type":"get","key":"a"}}
  ],
  "events": [
    {"sequence":0,"type":"invoke","operation_id":"put-a"},
    {"sequence":1,"type":"invoke","operation_id":"get-a"},
    {"sequence":2,"type":"complete","operation_id":"get-a","result":"alpha"}
  ],
  "abandoned": [],
  "client_request_ids": []
}
```

Put stays pending; only the actual observed Get completion is represented.

- `version` is integer `1`, excluding booleans/floats/strings.
- `requests` preserves registration order, including Append. Client/operation IDs
  are nonempty strings. Request IDs are nonnegative integers, excluding booleans.
  `(client_id, request_id)` and `operation_id` are unique; even identical duplicate
  entries are rejected. Exact retries reuse the existing entry/invocation.
- Operation objects contain exactly `type,key,value` for `put`, `type,key` for
  `delete`/`get`, or `type,key,suffix` for `append`. Keys are nonempty strings;
  values/suffixes are strings (empty allowed). Only these whitelisted operation
  types are constructed. There is no arbitrary Python-object decoder or pickle.
- `events` preserves global invocation/completion order. Sequence numbers start
  at zero and are contiguous integers, excluding booleans. Invoke events reference
  the registered request's client/operation; completion events add `result`.
  Put/Delete responses must be null; Get responses are strings or null. Every
  non-Append request has one invocation, in registration order after filtering
  Append. Completions require a preceding invocation and cannot repeat.
- Append appears only in `requests`. Its history invocation/completion is excluded
  from the Put/Delete/Get oracle as before. After reconstruction its outcome must
  be re-observed from server applied evidence.
- `abandoned` is a unique list of terminal pending Get IDs. Completed operations,
  writes and unknown IDs cannot be abandoned. No response is manufactured.
- `client_request_ids` entries contain exactly `operation_id,request_id`. They
  reference Put/Delete requests and agree with the request-table identity.
  Attachments are restored at invocation time, before completions, preserving
  pending metadata and a completed write's unretired attachment. Retired entries
  remain absent. Public history APIs rebuild the coupled active-identity map.

The exporter validates the in-memory journal and compares its reconstructed state
with the original. It rejects mismatched/unregistered history, sequence holes or
unsupported history metadata instead of silently discarding information.

## Observations and server evidence

Import validates representation/internal consistency. It cannot authenticate
whether an external observer actually saw a structurally valid claimed response;
there is no signature/identity infrastructure. Such client claims never establish
server commitment or application.

The existing Ack gate is unchanged. With an imported claimed Put completion and
an empty server, `poll(..., apply=False)` returns Pending even after valid Phase 1.
A contradictory imported Get result is rejected against the real applied result.
Without fresh leader authority, polling returns `leadership_unavailable`.

## Full Simulator integration

Set `journal_checkpoint=True` together with `reconstruction_boundary=N` on
`MultiPaxosApplicationScenarioRunner`. The client crosses the existing full-runtime
boundary through canonical JSON and a new decoded ClientJournal instance. Server
state still crosses only through durable evidence; old volatile authority and
queues are discarded. A following PREPARE establishes fresh leader authority.
Pending lost-Ack identities, multi-key operations and observed Get results survive.

The unchanged default is `journal_checkpoint=False`, using deepcopy. Only enabled
application artifacts include `"journal_checkpoint":true`; it must be a boolean
and requires a reconstruction boundary. Existing default-mode artifacts, including
previous deepcopy reconstruction checkpoints, retain their exact bytes and replay
behavior. Both paths produce identical simulated results/traces for a valid journal.
See [multipaxos-application-campaign.md](multipaxos-application-campaign.md) for the
underlying reconstruction model.

## Executed implementation evidence

Exact base: `3998f1d982765ea9f8b2c819005693e3d9195b54`.
Before edits: **817 passed**, Ruff passed. The baseline has neither portable API;
`json.dumps(ClientJournal(...))` raises `TypeError: Object of type ClientJournal is
not JSON serializable`. Retaining/deepcopying a Python object was required.

Before editing, eight application artifacts (normal, lost Ack, decision loss and
old deepcopy reconstruction at boundaries 0/1/2/end) plus four seeded consensus
artifacts were captured/replayed. All twelve canonical bytes, including complete
traces, match after implementation. Combined SHA-256:
`3663068e887caad03bab09fa516d88dfc01b8208e456608c021659531df7acec`.

The tests exercise completion order differing from invocation order, pending
requests, terminal metadata, Append exclusion, multi-key histories, fresh Phase-1
requirements, leader takeover, missed follower decisions/catch-up, full client/server
replacement, frozen Get results after later writes, identity conflicts, Delete/Get,
and exactly-once Append chosen in two slots. The JSON campaign path imports a new
journal and matches the default reconstruction's canonical result/trace. Its new
opt-in artifact round-trips and detects tampered replay results.

Disposable source mutations (never committed):

1. Remove duplicate/conflicting request identity rejection: **2 expected failures**
   with `DID NOT RAISE ValueError`.
2. Return success from client history before checking server applied evidence:
   **2 expected failures** (unproven Put wrongly completes; forged Get no longer
   raises a result mismatch).

Both source files were restored byte-for-byte, all source matched the corrected
checkout, and **69 checkpoint tests passed** again. Full test/Ruff results, exact
candidate SHA, CI and independent review evidence are recorded in the PR.

## Limits and Reviewer C handoff

This is a bounded laboratory representation, not filesystem durability, a database,
production RPC, cryptographic authenticity or a compaction protocol. Import builds
fresh state only after full validation; it does not merge checkpoints or invent
omitted invocations. Existing checker limits apply. Hostile-input resource limiting
is outside this format.

Independently validate the schema, observed event order, request identities and
terminal metadata. Rebuild fresh clients/servers and test forged claims against
empty/contradictory server evidence, fresh Phase 1, Append dedup and legacy bytes.
Repeat the disposable mutations and run full pytest/Ruff on the published HEAD.
