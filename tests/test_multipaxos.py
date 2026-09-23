import pytest

from distlab.multipaxos import LeaderAccept, MultiPaxosCluster
from distlab.paxos import PaxosError, ProposalNumber
from distlab.simulator import Simulator


def _cluster() -> tuple[Simulator, MultiPaxosCluster]:
    sim = Simulator()
    return sim, MultiPaxosCluster(sim, ("n1", "n2", "n3"))


def test_stable_leader_amortizes_phase1_across_slots() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")

    ballot = leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority

    leader.propose(1, "alpha")
    leader.propose(2, "beta")
    leader.propose(3, "gamma")
    sim.run()

    assert cluster.chosen(1).value == "alpha"
    assert cluster.chosen(2).value == "beta"
    assert cluster.chosen(3).value == "gamma"
    assert sum(record.kind == "multipaxos-phase1-start" for record in sim.trace) == 1
    assert sum(record.kind == "multipaxos-phase1-complete" for record in sim.trace) == 1
    assert sum(record.kind == "multipaxos-phase2-start" for record in sim.trace) == 3
    assert all(cluster.node(node).promised_ballot == ballot for node in cluster.node_ids)
    cluster.assert_safety()


def test_new_leader_phase1_adopts_highest_previously_accepted_value() -> None:
    sim, cluster = _cluster()
    first = cluster.node("n1")
    first_ballot = first.prepare_leadership(round_number=1)
    sim.run()

    # Leave an accepted value below quorum: partition before Phase 2 so the
    # leader accepts locally while its peer messages are deterministically dropped.
    sim.partition(("n1",), ("n2", "n3"))
    first.propose(1, "old")
    sim.heal_partition(("n1",), ("n2", "n3"))

    # A higher ballot must discover and preserve that accepted value through Phase 1.
    second = cluster.node("n2")
    second.prepare_leadership(round_number=2)
    sim.run()
    assert second.has_leader_authority

    second.propose(1, "new")
    sim.run()
    assert cluster.chosen(1).value == "old"
    assert cluster.chosen(1).proposal > first_ballot
    cluster.assert_safety()


def test_higher_global_promise_fences_lower_ballot_for_every_slot() -> None:
    sim, cluster = _cluster()
    first = cluster.node("n1")
    first.prepare_leadership(round_number=1)
    sim.run()

    second = cluster.node("n2")
    higher = second.prepare_leadership(round_number=2)
    sim.run()
    assert second.has_leader_authority

    stale = LeaderAccept(99, ProposalNumber(1, "n1"), "stale")
    for node_id in cluster.node_ids:
        cluster.node(node_id)._receive_accept(stale)
        assert cluster.node(node_id).promised_ballot == higher
        assert all(item.slot != 99 for item in cluster.node(node_id).accept_history)

    cluster.assert_safety()


def test_crash_discards_volatile_authority_but_keeps_durable_global_promise() -> None:
    sim, cluster = _cluster()
    leader = cluster.node("n1")
    ballot = leader.prepare_leadership()
    sim.run()
    assert leader.has_leader_authority

    sim.crash("n1")
    sim.restart("n1")
    assert not leader.has_leader_authority
    assert leader.promised_ballot == ballot
    with pytest.raises(PaxosError, match="Phase-1 quorum"):
        leader.propose(1, "unsafe")

    new_ballot = leader.prepare_leadership()
    sim.run()
    assert new_ballot > ballot
    leader.propose(1, "safe")
    sim.run()
    assert cluster.chosen(1).value == "safe"
    cluster.assert_safety()
