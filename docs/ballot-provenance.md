# Multi-Paxos ballot provenance integrity

## Invariant and scope

A learned `(slot, proposal, value)` requires matching durable acceptance records
from a distinct majority of member nodes. The slot and ballot round must be
positive integers, excluding booleans, and the ballot owner must be a member.
Repeated acceptance records on one node count once. Matching acceptance evidence
also passes the existing durable-history/promise checks.

`MultiPaxosCluster._require_decision_evidence` generalizes the existing recovery
validator. Ordinary learning (including duplicates), recovery responses on both
send and receive, durable safety checkpoints, reconstruction and state-machine
application use it. Recovery still reads **only the responding peer's own durable
learned map** to populate a response. The whole response is checked before any
learner mutation.

Every applied record must equal the same node's durable learned record in slot,
ballot and value, preserve a contiguous prefix, and retain objective chosen-value
evidence. Cached state/dedup results are checked before being exposed, including
`MultiPaxosClient.poll(..., apply=False)`. Restoring a crashed node's state checks
provenance before replaying the prefix into KV or dedup state. Invalid provenance
raises `PaxosSafetyViolation` before application or a successful acknowledgement.
No client Ack, retry, Get-result, consensus adoption or history logic changes.

A ballot need not equal the runtime chosen map's first ballot or reconstruction's
highest ballot. The same value may have several genuine quorum ballots; a node
retains its first learned record, and its applied record must retain that ballot.
The shared validator searches retained acceptance history, not just each acceptor's
latest accepted record. This preserves delayed evidence across leader takeover.

This is a simulator safety invariant rooted in durable acceptor histories. It is
not cryptographic authentication, Byzantine consensus, a production proof protocol,
or protection against fabrication of the evidence root itself. Histories are not
compacted. Rechecking evidence scans retained history; no cache or new artifact
format is introduced. Pure `MultiPaxosStateMachine.replay(...)` remains an offline
command-folding helper, not an evidence-validating reconstruction API; use a
cluster/state machine constructed from persistence for validated reconstruction.

## Independently constructed baseline reproductions

Exact unpatched base: `aafbc28f2fb67f69c0687fe47c9339ab3c82744a`.
Python 3.12.14, pytest 8.4.2, Ruff 0.17.3. Baseline full suite: **718 passed**;
Ruff passed. These are implementer reproductions, not an independent acceptance
review.

Each case constructs a fresh three-node cluster, performs real Phase 1 at
`(1, n1)`, proposes `Put(k, chosen)` at slot 1 and processes actual transport
messages. All three acceptors durably accept that exact tuple. `(99, n3)` is a
syntactically valid member ballot with **zero** matching acceptance records.
Only the indicated record's proposal field changes; slot and chosen value remain
unchanged. The chosen map is never edited or used as the peer's response source.

| Corruption | Unpatched safety checkpoint | Unpatched full reconstruction | Corrected behavior |
| --- | --- | --- | --- |
| Learned proposal becomes `(99, n3)` | Accepted | Accepted; KV `k=chosen` | `PaxosSafetyViolation`: lacks matching durable quorum evidence |
| Applied proposal becomes `(99, n3)`; learned remains `(1, n1)` | Accepted | Accepted; KV `k=chosen` | `PaxosSafetyViolation`: no matching durable learned decision |

The standalone reproduction also exercised an already-applied prefix. The focused
learned regression isolates that invariant with no applied state, so an applied
mismatch check cannot accidentally hide a missing learned-ballot check.

Run the identical four regression cases against unpatched source from an isolated
worktree, then against the corrected checkout. From the corrected checkout:

```sh
git worktree add --detach ../provenance-base aafbc28f2fb67f69c0687fe47c9339ab3c82744a
PYTHONPATH=../provenance-base/src python -m pytest -q tests/test_multipaxos_provenance.py -k 'correct_value_with_forged_learned or applied_ballot_different'
PYTHONPATH=src python -m pytest -q tests/test_multipaxos_provenance.py -k 'correct_value_with_forged_learned or applied_ballot_different'
```

Actual results: base **4 failed**, each `Failed: DID NOT RAISE
<class 'distlab.paxos.PaxosSafetyViolation'>`; corrected **4 passed**. Both the
checkpoint and reconstruction paths are tested independently for each invariant.

## Legitimate re-choice and integrations

The re-choice test chooses `Append(k, once)` in slot 1, with n3 missing the first
notification. n1 crashes; n3 obtains a real higher-ballot Phase-1 quorum, adopts
the accepted value instead of its proposed replacement, and re-chooses it.

| Evidence | Ballot `(1, n1)` | Ballot `(2, n3)` |
| --- | --- | --- |
| Durable acceptors | n1, n2 | n2, n3 |
| Nodes retaining this learned/applied ballot | n1, n2 | n3 |

Runtime objective evidence retains `(1, n1)`; full reconstruction selects
`(2, n3)`. All three nodes retain their local learned/applied ballots and KV
`k=once`. Both checkpoints pass. Delayed old-ballot recovery and newer duplicate
notifications do not rewrite either prefix or repeat Append. The same scenario
also passes for accepted `None`.

The dedicated client integration uses real API requests and transport:

- Slot 1: A/1 Put `k=v`; slot 2: B/1 Get returns `v`. This supported history passes
  the single-key linearizability checker.
- Slot 3: A/2 Append `!`, chosen at `(1, n1)` with its Ack withheld; n3 misses it.
  n1 crashes, n2 obtains Phase-1 authority, and A/2 retries successfully.
- Slot 4: the same request is chosen at `(2, n2)`; dedup retains first-effect slot
  3. Normal follower recovery repairs n3, without re-proposing its missing slot.
- Slot 5: B/2 Get returns `v!`, demonstrating one Append effect; slot 6: A/3
  Delete; slot 7: B/3 Get returns `None`.
- All nodes converge to an empty KV and identical dedup/results. Full runtime
  reconstruction and a new Phase 1 preserve A/2's first-effect slot and B/2's
  original `v!` result after deletion. Append is tested outside the Put/Delete/Get
  checker.

Other new cases cover forged duplicate decisions, malformed/non-member ballots,
one acceptor with duplicate records, a valid-first/forged-later recovery batch,
and corrupted cached client results with both `apply=True` and `apply=False`.
Existing crash/restart, catch-up, accepted-None, client linearizability and pending
completion-extension tests remain active.

## Disposable negative mutations

No mutation code is committed. Each probe ran in a separate source copy, then
restored the corrected bytes and checked identity.

| Temporary mutation | Dedicated regression result |
| --- | --- |
| Remove `accepted.proposal == decision.proposal` from the shared evidence matcher, retaining slot/value matching | **2 failed**: forged learned ballot checkpoint/reconstruction both `DID NOT RAISE` |
| Replace applied/learned record equality with the old value-only comparison | **3 failed**: both applied mismatch paths and the mismatch where both ballots independently have real quorums all `DID NOT RAISE` |

Restored source hashes (SHA-256):

- `multipaxos.py`: `accec56629ad874a9a3b980e0fe0a25908109516df619400ff34bac31d07de82`
- `multipaxos_state_machine.py`: `959cbf04a6b8a89e8a6b9c1d99a13a04c061cc9423f473d3de87ebd07253ed45`

All Python source bytes matched the corrected checkout after restoration; the
restored copy then passed all **28** dedicated provenance tests.

## Executed regressions and compatibility

```sh
PYTHONPATH=src python -m pytest -q
python -m ruff check .
PYTHONPATH=src python -m pytest -q tests/test_multipaxos*.py
PYTHONPATH=src python -m pytest -q tests/test_multipaxos_provenance.py
```

Corrected results: **746 passed** full suite; **194 passed** focused Multi-Paxos;
**28 passed** dedicated provenance; Ruff passed. Two existing forged-message tests
now assert rejection before learning rather than during later application. The
existing peer-trust negative test patches the renamed shared validator and still
verifies the separate chosen-value application guard. No test was removed.

The focused suite executes the existing eight catch-up fault seeds
`101, 202, 303, 404, 919, 1729, 2026, 65537`, three client seeds `101, 202, 303`,
and three pending-history seeds `101, 202, 303`, with serialized schedules and
exact replay. Fault effects are asserted, not assumed.

A separate base/candidate comparison generated and replayed 20 default
learner-disabled Multi-Paxos artifacts (seeds 0–19, 12 actions, leadership-change
rate .3, max-events-after 6, drop/delay/duplicate .04 each, crash .15/restart .25,
max ordinal 16, final event budget 128) and two existing Raft stale-read failure
artifacts (seed 2, with/without lifecycle). **22 artifacts and 22 replays per
version** were exercised. Canonical artifact bytes were identical, combined
SHA-256 `c85ef4caf1e710b85d32abcf2b16e704ef22f7817efab31a0b81aef9d58407e7`.

The existing client demo and all seven pending-history demo modes also produced
byte-identical canonical JSON on base/candidate: eight reports per version,
combined SHA-256 `4145c91a9f11efc3c73aeb584498fc749521e7f158ad8420a86681d4c6a0cf8b`.
Raft, single-decree/log Paxos, client API, both linearizability oracles, campaign
schemas and replay codecs are unchanged. CI status and the exact candidate SHA
are recorded in the PR, outside the source tree.

## Independent Reviewer C handoff

Verify the PR's exact HEAD and base, reproduce the four expected base failures,
then run the commands above. Inspect the shared quorum matcher, pre-learning
validation and the cached-machine guard. In particular, verify that both genuine
ballots in the re-choice case pass, while a locally mismatched applied ballot
fails even when it independently has a quorum. Repeat the two disposable mutations
and restore their bytes. Confirm all six CI matrix jobs on the reviewed HEAD.
This PR is an implementation handoff; independent acceptance remains outstanding.
