from env import WildfireEnv
from env.serialization import serialize_observation


def test_serialize_produces_all_sections(fresh_env):
    obs = fresh_env.reset(task_id="easy", seed=42)
    text = serialize_observation(obs, step_num=0, max_steps=80)
    for section in ["SITUATION:", "GRID SUMMARY", "RESOURCES:", "RECENT EVENTS:", "Available actions:"]:
        assert section in text, f"Missing section: {section}"


def test_serialize_handles_fog_of_war(fresh_env):
    obs = fresh_env.reset(task_id="hard", seed=42)
    text = serialize_observation(obs, step_num=0, max_steps=300)
    assert "[?]" in text, "Expected fog-of-war marker [?] in hard tier output"


def test_serialize_length_under_2048_tokens(fresh_env):
    for tier, max_steps in [("easy", 80), ("medium", 150), ("hard", 300)]:
        obs = fresh_env.reset(task_id=tier, seed=42)
        text = serialize_observation(obs, step_num=0, max_steps=max_steps)
        word_count = len(text.split())
        assert word_count < 1500, (
            f"Tier {tier}: serialized prompt too long ({word_count} words, limit 1500)"
        )
