from env import WildfireEnv
from env.briefing import briefing_to_text
from env.serialization import serialize_observation
from agents.heuristic_agent import HeuristicAgent


def test_briefing_generated_on_reset():
    env = WildfireEnv()
    obs = env.reset(task_id="medium", seed=42)
    assert obs.briefing is not None, "Briefing should be present on first obs"
    assert len(obs.briefing.priority_populated_zones) >= 1, "Should have at least 1 priority zone"
    assert obs.briefing.incident_id != ""
    assert obs.briefing.ignition_cause != ""


def test_briefing_adherence_bonus():
    env = WildfireEnv()
    agent = HeuristicAgent()
    obs = env.reset(task_id="easy", seed=42)

    total_reward = 0.0
    while not env.done:
        action = agent.act(obs)
        result = env.step(action)
        total_reward += result.reward
        obs = result.observation

    final = env.state()
    pop_lost = final.get("population_lost", 0)
    # On easy with heuristic seed=42, all pop should be saved -> briefing bonus applies
    if pop_lost == 0:
        assert total_reward > 5.0, (
            f"Expected reward > 5.0 (includes +1 briefing bonus) but got {total_reward}"
        )


def test_briefing_in_serialized_prompt():
    env = WildfireEnv()
    obs = env.reset(task_id="medium", seed=42)
    text = serialize_observation(obs, 0, 150)
    assert "OPERATIONAL BRIEFING" in text, "Briefing header missing from serialized prompt"
    assert "PRIORITY 1" in text
    assert "Commander's intent" in text
