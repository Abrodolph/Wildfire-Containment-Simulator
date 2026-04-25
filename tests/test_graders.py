from agents.heuristic_agent import HeuristicAgent
from graders.grader_easy import grade as grade_easy
from graders.grader_medium import grade as grade_medium
from graders.grader_hard import grade as grade_hard


def _check_details(details):
    assert isinstance(details, dict)
    assert "total_reward" in details
    assert "containment_pct" in details
    assert "pop_saved_pct" in details
    assert "steps" in details
    assert "crew_casualty" in details
    assert isinstance(details["total_reward"], float)
    assert 0.0 <= details["containment_pct"] <= 1.0
    assert 0.0 <= details["pop_saved_pct"] <= 1.0
    assert details["steps"] > 0
    assert isinstance(details["crew_casualty"], bool)


def test_each_grader_returns_float_and_details():
    agent = HeuristicAgent()
    for grade_fn in (grade_easy, grade_medium, grade_hard):
        result = grade_fn(agent, seed=42)
        assert isinstance(result, tuple) and len(result) == 2
        score, details = result
        assert isinstance(score, float)
        _check_details(details)


def test_grader_scores_are_in_expected_range():
    agent = HeuristicAgent()

    score_easy, _ = grade_easy(agent, seed=42)
    assert score_easy > 3.0, f"Easy heuristic score too low: {score_easy}"

    score_medium, _ = grade_medium(agent, seed=42)
    assert score_medium > -6.0, f"Medium heuristic score too low: {score_medium}"

    score_hard, _ = grade_hard(agent, seed=42)
    assert score_hard > -8.0, f"Hard heuristic score too low: {score_hard}"
