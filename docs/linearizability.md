# Observations and completion extensions

`SingleKeyKVLinearizabilityChecker.check(history)` retains its original
completed-only projection, verdict, and deterministic order. Historical Raft/Paxos
campaigns and minimizers continue to call that method. They are not silently
reinterpreted by this milestone.

The explicit opt-in API is:

```python
from distlab.linearizability import (
    LinearizabilitySearchExhausted,
    SingleKeyKVLinearizabilityChecker,
)

checker = SingleKeyKVLinearizabilityChecker()
try:
    witness = checker.check_with_pending_extensions(client.history)
except LinearizabilitySearchExhausted:
    # Resource budget exhausted: no linearizability verdict was established.
    raise
```

## Three different kinds of evidence

1. **Observed history:** the actual Invocation and Completion records. A lost-Ack
   write stays pending until a real client API response is recorded.
2. **Hypothetical completion extension:** the checker may select a pending write
   and place its effect in a legal sequential witness, as if its response were
   appended after the observed history. It adds no events to OperationHistory.
3. **Server evidence:** chosen decisions, applied prefixes, and request results
   remain the consensus/application layer's responsibility. This pure history
   oracle neither reads server state nor proves any pending request was chosen.

For example, an observed pending Put("k", "v") followed by a completed Get
returning "v" admits the witness Put -> Get. The writer's invocation overlaps
the read because it has no observed response. The original completed-only oracle
rejects this projection; the opt-in oracle can include the pending Put without
claiming that its acknowledgement was received.

## Formal semantics and result

Starting from an absent key, the search requires every completed Put/Delete/Get
exactly once. It may include or omit each pending Put/Delete. Pending Get,
including an abandoned read, is always omitted: it has no observed result or
effect that could explain another operation. No observed Get result is invented.

For every **actual** completion A that precedes a selected invocation B, A must
precede B in the witness. This includes edges into selected pending writes. A
pending operation has no observed response, so the checker never constructs an
early-response edge out of it. Two pending writes can be ordered either way when
their observations impose no precedence. A write invoked after a completed read
cannot retroactively explain that read's result.

The returned immutable `CompletionExtensionResult` has:

- `linearizable`: whether a valid extension exists within the completed search;
- `order`: selected operation IDs in legal sequential order;
- `included_pending`: selected pending write IDs, in witness order;
- `omitted_pending`: all other pending IDs, in original invocation order.

All completed IDs occur in a positive witness; each selected ID occurs once.
Put sets the key, Delete removes it, and each completed Get must match the value
at that point. A positive witness can therefore be replayed independently against
the original invocations/responses and their real-time precedence.

A negative result has `linearizable=False` and three empty tuples: it has no
legal witness or pending assignment. Pending-only histories may be accepted with
an empty order and every pending ID omitted. The deterministic search considers
completed candidates first, but does not promise the smallest pending subset.

Neither successful nor unsuccessful checking mutates invocations, completions,
sequence counters, abandonment state, or client request-ID metadata. Retry/Ack
gates in MultiPaxosClient and the Raft client APIs are unchanged.

## Supported inputs and bounded search

The opt-in method validates one non-empty string key across **all** invocations,
including pending operations that would be omitted. Put values must be strings.
Only Put/Delete/Get are supported; Append remains outside this checker.

The DFS memoizes failed `(selected-operation mask, current value)` states. Order
is deterministic and no wall-clock or random choices are involved. Budgets are:

- `max_operations=32`, configurable from 1 through 64. All invocations count,
  including omitted pending reads. The upper bound also bounds recursion depth.
- `max_states=100_000`, a positive integer limiting expanded non-terminal states.
  Memoized failures and terminal successes do not consume another expansion.

Exceeding either budget raises `LinearizabilitySearchExhausted`, distinct from a
negative verdict. Invalid budgets raise ValueError; unsupported operations/values
raise TypeError; key-contract violations raise InvalidHistory. Large difficult
histories may exhaust the budget rather than be classified.

## Compatibility and minimization boundary

The legacy `NonLinearizableHistoryMinimizer` remains a completed-only tool: it
uses `check()` and projects away pending invocations. It is **not supported for
completion-extension verdicts**. Do not override its oracle to forward to the new
method and then trust its projection: deleting pending evidence may change the
classification. This milestone adds no extension-aware minimizer or artifact
schema, and historical failure artifacts keep their original interpretation.

## Executable protocol evidence

Run from an installed checkout or with `PYTHONPATH=src`:

```bash
python -m distlab.pending_history_demo
python -m distlab.pending_history_demo takeover
```

Other scenarios are `restart`, `catchup`, `reconstruct`, `retry`, and `unchosen`.
Each uses actual MultiPaxosClient calls: A's Put is chosen/applied with its Ack
withheld, then B's log-based Get returns "v". The report separately prints the
observed history, hypothetical witness, and actual slot/ballot/chosen/applied
evidence. Before an actual retry, only B has an observed Completion; A stays
pending despite the oracle's positive result. The unchosen scenario also leaves
a minority-accepted C write without quorum/application and omits it from the
witness. The retry scenario shows a later **real** completion without a second
invocation. Fixed-seed message-fault schedules and lifecycle scenarios replay the
same histories, witness, and trace digest in dedicated tests.
