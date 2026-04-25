import os
import tempfile
from env import WildfireEnv
from env.models import Action, ActionType
from env.rendering import render_frame, render_episode_gif
from agents.random_agent import RandomAgent


def test_render_frame_produces_rgb(fresh_env):
    fresh_env.reset(task_id="easy", seed=42)
    state = fresh_env.state()
    frame = render_frame(state, step=0)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype.name == "uint8"
    assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_gif_creation(fresh_env):
    agent = RandomAgent()
    obs = fresh_env.reset(task_id="easy", seed=42)
    frames = [render_frame(fresh_env.state(), step=0)]
    for i in range(1, 21):
        action = agent.act(obs)
        result = fresh_env.step(action)
        obs = result.observation
        frames.append(render_frame(fresh_env.state(), step=i))
        if result.done:
            break

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        path = f.name
    try:
        render_episode_gif(frames, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 10_000, "GIF too small — likely empty"
    finally:
        os.unlink(path)
