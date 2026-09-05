import pytest

from distlab.raft import AppendEntriesResponse, LogEntry, RaftCluster, RaftRole
from distlab.replication import LeaderReplicator, ReplicationResponseMissing
from distlab.simulator import Simulator


def test_stale_success_response_cannot_satisfy_new_append_probe() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))
    leader = cluster.node("n1")
    leader.start_election()
    sim.run()
    assert leader.role is RaftRole.LEADER

    leader._persist_log(
        (
            LogEntry(term=leader.current_term, command="a"),
            LogEntry(term=leader.current_term, command="b"),
        )
    )
    replicator = LeaderReplicator(leader)

    sim.send(
        "n2",
        "n1",
        AppendEntriesResponse(
            term=leader.current_term,
            follower_id="n2",
            success=True,
            match_index=1,
        ),
        delay=10,
    )
    sim.partition(("n1",), ("n2",))

    with pytest.raises(ReplicationResponseMissing, match="no AppendEntries response"):
        replicator.replicate("n2", max_attempts=1)

    progress = replicator.progress("n2")
    assert progress.match_index == 0
    assert progress.next_index == 3

    stale = [
        record
        for record in sim.trace
        if record.kind == "raft-append-response"
        and record.details["follower"] == "n2"
        and record.details["success"] is True
        and record.details["match_index"] == 1
    ]
    assert stale

    sim.heal_partition(("n1",), ("n2",))
    assert replicator.replicate("n2", max_attempts=1) is True
    assert replicator.progress("n2").match_index == 2
    assert replicator.progress("n2").next_index == 3

    cluster.assert_log_matching()
