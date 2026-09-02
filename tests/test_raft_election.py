from distlab.raft import LogEntry, RaftCluster, RaftRole, RequestVote
from distlab.simulator import FaultAction, FaultPlan, FaultRule, Simulator


def test_three_node_election_reaches_single_leader() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    sim.run()

    assert cluster.node("n1").role is RaftRole.LEADER
    assert cluster.leaders_by_term == {1: "n1"}
    assert len(cluster.node("n1").votes_received) >= 2


def test_simultaneous_candidates_cannot_both_win_same_term() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    cluster.node("n2").start_election()
    sim.run()

    assert list(cluster.leaders_by_term) == [1]
    assert cluster.leaders_by_term[1] in {"n1", "n2"}
    assert sum(node.role is RaftRole.LEADER for node in cluster.nodes.values()) == 1


def test_vote_persists_across_crash_and_restart() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    sim.send("n1", "n3", RequestVote(term=1, candidate_id="n1"))
    sim.run(max_events=1)
    assert cluster.node("n3").voted_for == "n1"

    sim.crash("n3")
    sim.restart("n3")
    sim.send("n2", "n3", RequestVote(term=1, candidate_id="n2"))
    sim.run()

    assert cluster.node("n3").voted_for == "n1"
    votes = [
        record
        for record in sim.trace
        if record.kind == "raft-vote" and record.details["voter"] == "n3"
    ]
    assert [(record.details["candidate"], record.details["granted"]) for record in votes] == [
        ("n1", True),
        ("n2", False),
    ]


def test_higher_term_request_resets_old_vote_and_steps_down() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()
    sim.run()
    assert cluster.node("n1").role is RaftRole.LEADER

    sim.send("n2", "n1", RequestVote(term=2, candidate_id="n2"))
    sim.run()

    assert cluster.node("n1").current_term == 2
    assert cluster.node("n1").voted_for == "n2"
    assert cluster.node("n1").role is RaftRole.FOLLOWER


def test_duplicate_vote_responses_do_not_inflate_majority() -> None:
    sim = Simulator(
        fault_plan=FaultPlan(
            (
                FaultRule(
                    FaultAction.DUPLICATE,
                    src="n2",
                    dst="n1",
                    ordinal=1,
                    extra_delay=0,
                ),
                FaultRule(FaultAction.DROP, src="n1", dst="n3", ordinal=1),
            )
        )
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3", "n4", "n5"))

    cluster.node("n1").start_election()
    sim.crash("n4")
    sim.crash("n5")
    sim.run()

    assert cluster.node("n1").role is RaftRole.CANDIDATE
    assert cluster.node("n1").votes_received == frozenset({"n1", "n2"})
    assert cluster.leaders_by_term == {}


def test_crashed_node_cannot_start_election() -> None:
    sim = Simulator()
    cluster = RaftCluster(sim, ("n1",))
    sim.crash("n1")

    try:
        cluster.node("n1").start_election()
    except RuntimeError as exc:
        assert "cannot start an election" in str(exc)
    else:
        raise AssertionError("crashed nodes must not start elections")


def test_request_vote_rejects_candidate_with_older_last_log_term() -> None:
    sim = Simulator()
    sim.persistent_state["n2"]["log"] = (LogEntry(term=1), LogEntry(term=3))
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        RequestVote(
            term=4,
            candidate_id="n1",
            last_log_index=5,
            last_log_term=2,
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").current_term == 4
    assert cluster.node("n2").voted_for is None
    vote = next(record for record in sim.trace if record.kind == "raft-vote")
    assert vote.details["granted"] is False
    assert vote.details["log_up_to_date"] is False


def test_request_vote_uses_index_when_last_log_terms_match() -> None:
    sim = Simulator()
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1),
        LogEntry(term=3),
        LogEntry(term=3),
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        RequestVote(
            term=4,
            candidate_id="n1",
            last_log_index=2,
            last_log_term=3,
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").voted_for is None
    vote = next(record for record in sim.trace if record.kind == "raft-vote")
    assert vote.details["granted"] is False
    assert vote.details["candidate_last_log_index"] == 2
    assert vote.details["voter_last_log_index"] == 3


def test_request_vote_accepts_candidate_with_newer_log_term() -> None:
    sim = Simulator()
    sim.persistent_state["n2"]["log"] = (
        LogEntry(term=1),
        LogEntry(term=2),
        LogEntry(term=2),
    )
    cluster = RaftCluster(sim, ("n1", "n2"))

    sim.send(
        "n1",
        "n2",
        RequestVote(
            term=4,
            candidate_id="n1",
            last_log_index=1,
            last_log_term=3,
        ),
    )
    sim.run(max_events=1)

    assert cluster.node("n2").voted_for == "n1"
    vote = next(record for record in sim.trace if record.kind == "raft-vote")
    assert vote.details["granted"] is True
    assert vote.details["log_up_to_date"] is True


def test_start_election_advertises_persistent_log_metadata() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (
        LogEntry(term=1),
        LogEntry(term=2),
        LogEntry(term=2),
    )
    cluster = RaftCluster(sim, ("n1", "n2", "n3"))

    cluster.node("n1").start_election()

    sends = [
        record
        for record in sim.trace
        if record.kind == "send" and record.details["src"] == "n1"
    ]
    requests = [record.details["payload"] for record in sends]
    assert requests
    assert all(request.last_log_index == 3 for request in requests)
    assert all(request.last_log_term == 2 for request in requests)


def test_persistent_log_survives_crash_restart_boundary() -> None:
    sim = Simulator()
    sim.persistent_state["n1"]["log"] = (LogEntry(term=1), LogEntry(term=2))
    cluster = RaftCluster(sim, ("n1",))

    sim.crash("n1")
    sim.restart("n1")

    assert cluster.node("n1").log == (LogEntry(term=1), LogEntry(term=2))
    assert cluster.node("n1").last_log_index == 2
    assert cluster.node("n1").last_log_term == 2
