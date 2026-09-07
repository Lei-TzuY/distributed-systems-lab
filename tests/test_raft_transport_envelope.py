import pytest

from distlab.raft import (
    AppendEntries,
    AppendEntriesResponse,
    LogEntry,
    RaftCluster,
    RequestVote,
    RequestVoteResponse,
)
from distlab.raft_invariants import RaftSafetyHarness
from distlab.simulator import Message, Simulator


@pytest.mark.parametrize(
    "payload",
    [
        RequestVote(term=8, candidate_id="n1"),
        RequestVoteResponse(term=8, voter_id="n1", vote_granted=True),
        AppendEntries(
            term=8,
            leader_id="n1",
            entries=(LogEntry(term=8, command="forged"),),
        ),
        AppendEntriesResponse(term=8, follower_id="n1", success=True, match_index=0),
    ],
)
def test_forged_payload_source_cannot_mutate_raft_state(payload: object) -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    follower = cluster.node("n2")

    sim.send("n3", "n2", payload)
    sim.run()

    assert follower.current_term == 0
    assert follower.voted_for is None
    assert follower.log == ()
    rejected = [record for record in sim.trace if record.kind == "raft-envelope-rejected"]
    assert len(rejected) == 1
    assert rejected[0].details["node"] == "n2"
    assert rejected[0].details["src"] == "n3"
    assert rejected[0].details["claimed_src"] == "n1"
    assert rejected[0].details["reason"] == "source-identity-mismatch"
    RaftSafetyHarness(cluster).checkpoint()


def test_delivery_destination_mismatch_is_rejected_before_term_advance() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    follower = cluster.node("n2")
    forged = Message(
        src="n1",
        dst="n3",
        payload=AppendEntries(term=9, leader_id="n1"),
        ordinal=1,
    )

    follower.handle_message(sim, forged)

    assert follower.current_term == 0
    rejected = [record for record in sim.trace if record.kind == "raft-envelope-rejected"]
    assert len(rejected) == 1
    assert rejected[0].details["node"] == "n2"
    assert rejected[0].details["dst"] == "n3"
    assert rejected[0].details["reason"] == "destination-identity-mismatch"
    RaftSafetyHarness(cluster).checkpoint()
