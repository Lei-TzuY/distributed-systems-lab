from __future__ import annotations

from typing import Any

from .paxos import PaxosSafetyViolation, ProposalNumber
from .paxos_log import PaxosLogCluster, SlotLearnedDecision
from .simulator import Simulator


class RecoveringPaxosLogCluster(PaxosLogCluster):
    """Paxos log runtime that reconstructs chosen decisions from durable accept history.

    ``PaxosLogCluster`` records chosen decisions when the live proposer observes an
    accept quorum.  That observation is volatile.  A reconstructed runtime must
    therefore derive the same objective decision from acceptor histories instead of
    treating an empty in-memory chosen map as evidence that nothing was chosen.
    """

    def __init__(self, sim: Simulator, node_ids: tuple[str, ...]) -> None:
        super().__init__(sim, node_ids)
        self._recover_chosen_from_durable_history()
        self.assert_safety()

    def _recover_chosen_from_durable_history(self) -> None:
        evidence: list[tuple[int, ProposalNumber, Any, set[str]]] = []
        proposal_values: list[tuple[int, ProposalNumber, Any]] = []

        for node_id in self.node_ids:
            for accepted in self.node(node_id).accept_history:
                matching = [
                    value
                    for slot, proposal, value in proposal_values
                    if slot == accepted.slot and proposal == accepted.proposal
                ]
                if matching and matching[0] != accepted.value:
                    raise PaxosSafetyViolation(
                        f"slot {accepted.slot} proposal {accepted.proposal!r} "
                        "has conflicting durable accepted values"
                    )
                if not matching:
                    proposal_values.append(
                        (accepted.slot, accepted.proposal, accepted.value)
                    )

                for item in evidence:
                    slot, proposal, value, acceptors = item
                    if (
                        slot == accepted.slot
                        and proposal == accepted.proposal
                        and value == accepted.value
                    ):
                        acceptors.add(node_id)
                        break
                else:
                    evidence.append(
                        (
                            accepted.slot,
                            accepted.proposal,
                            accepted.value,
                            {node_id},
                        )
                    )

        chosen_by_slot: dict[int, list[tuple[ProposalNumber, Any, set[str]]]] = {}
        for slot, proposal, value, acceptors in evidence:
            if len(acceptors) >= self.quorum_size:
                chosen_by_slot.setdefault(slot, []).append(
                    (proposal, value, acceptors)
                )

        for slot, candidates in sorted(chosen_by_slot.items()):
            value = candidates[0][1]
            if any(candidate_value != value for _, candidate_value, _ in candidates[1:]):
                raise PaxosSafetyViolation(
                    f"durable quorum evidence chose multiple values for slot {slot}"
                )

            proposal, _, acceptors = min(candidates, key=lambda item: item[0])
            self._chosen[slot] = SlotLearnedDecision(slot, proposal, value)
            self._chosen_acceptors[slot] = tuple(sorted(acceptors))
            self.sim._record(
                "paxos-log-chosen-recovered",
                slot=slot,
                proposal=proposal,
                value=value,
                acceptors=self._chosen_acceptors[slot],
            )
