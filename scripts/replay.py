"""
Replay script — renders a full episode as an animated GIF.

Usage:
    python scripts/replay.py --tier medium --seed 42 --agent heuristic --output demos/out.gif
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import WildfireEnv
from env.rendering import render_frame, render_episode_gif
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


AGENT_REGISTRY = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", choices=list(AGENT_REGISTRY), default="heuristic")
    parser.add_argument("--output", default="demos/replay.gif")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    env = WildfireEnv()
    agent = AGENT_REGISTRY[args.agent]()
    obs = env.reset(task_id=args.tier, seed=args.seed)

    frames = []
    step = 0

    # Capture initial frame
    s = env.state()
    frames.append(render_frame(s, step))

    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        obs = result.observation
        done = result.done
        step += 1
        s = env.state()
        frames.append(render_frame(s, step))

    print(f"Episode finished at step {step}. Rendering {len(frames)} frames...")

    render_episode_gif(frames, args.output)
    print(f"GIF saved → {args.output}")

    # Save final-frame PNG
    png_path = os.path.splitext(args.output)[0] + ".png"
    import imageio.v3 as iio
    iio.imwrite(png_path, frames[-1], extension=".png")
    print(f"Final frame PNG → {png_path}")

    # Print final stats
    final_state = env.state()
    pop_lost = final_state.get("population_lost", 0)
    total_pop = final_state.get("total_population", 0)
    cells_burned = final_state.get("cells_burned", 0)
    print(f"\nFinal stats: step={step}, pop_lost={pop_lost}/{total_pop}, cells_burned={cells_burned}")


if __name__ == "__main__":
    main()
