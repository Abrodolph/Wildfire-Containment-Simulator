"""8 test cases for the 3-layer LLM action parser."""

import pytest
from env import WildfireEnv
from env.action_parser import parse_action
from env.models import ActionType


@pytest.fixture
def obs():
    env = WildfireEnv()
    return env.reset(task_id="easy", seed=42)  # 15x15 grid


def test_clean_json(obs):
    out = '{"action_type": "idle"}'
    action, status = parse_action(out, obs)
    assert status == "json_success"
    assert action.action_type == ActionType.IDLE


def test_json_in_fences(obs):
    out = '```json\n{"action_type": "recon_flight", "target_row": 3, "target_col": 4}\n```'
    action, status = parse_action(out, obs)
    assert status == "json_success"
    assert action.action_type == ActionType.RECON_FLIGHT


def test_json_with_surrounding_text(obs):
    out = 'I will deploy a crew. {"action_type": "deploy_crew", "crew_id": "crew_0", "target_row": 5, "target_col": 5} That is my plan.'
    action, status = parse_action(out, obs)
    assert status == "json_success"
    assert action.action_type == ActionType.DEPLOY_CREW


def test_malformed_json_regex_fallback(obs):
    # Missing quotes around value — JSON parse fails, regex should save it
    out = "action_type: deploy_crew, crew_id: crew_1, target_row: 6, target_col: 7"
    action, status = parse_action(out, obs)
    assert status == "regex_fallback"
    assert action.action_type == ActionType.DEPLOY_CREW


def test_garbage_output_safe_idle(obs):
    out = "I have no idea what to do here, just randomness @@##!!"
    action, status = parse_action(out, obs)
    assert status == "safe_idle"
    assert action.action_type == ActionType.IDLE


def test_out_of_bounds_coords_safe_idle(obs):
    # 15x15 grid — row 99 is out of bounds
    out = '{"action_type": "recon_flight", "target_row": 99, "target_col": 99}'
    action, status = parse_action(out, obs)
    assert action.action_type == ActionType.IDLE


def test_hallucinated_action_type_safe_idle(obs):
    out = '{"action_type": "nuke_fire", "target_row": 5, "target_col": 5}'
    action, status = parse_action(out, obs)
    assert status == "safe_idle"
    assert action.action_type == ActionType.IDLE


def test_empty_string_safe_idle(obs):
    action, status = parse_action("", obs)
    assert status == "safe_idle"
    assert action.action_type == ActionType.IDLE
