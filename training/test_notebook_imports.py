"""
Validates that all modules used in the training notebook can be imported
and the environment can be instantiated. Run this before opening Colab.
No GPU or model weights required.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Checking core env imports...")
from env import WildfireEnv
from env.serialization import serialize_observation
from env.action_parser import parse_action
from env.curriculum import CurriculumController
from env.models import TIER_EASY, TIER_MEDIUM, TIER_HARD
print("  OK")

print("Checking agent imports...")
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
print("  OK")

print("Checking stdlib / data science imports...")
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
print("  OK")

print("Checking imageio...")
import imageio.v3
print("  OK")

print("Instantiating environment and running one reset...")
env = WildfireEnv()
obs = env.reset(task_id="easy", seed=0)
text = serialize_observation(obs, 0, 80)
assert "SITUATION" in text
assert len(text) > 50
print("  OK")

print("Checking CurriculumController...")
ctrl = CurriculumController(start_tier="easy")
ctrl.after_episode(5.0)
assert ctrl.get_tier() in ("easy", "medium", "hard")
print("  OK")

print("Checking optional heavy deps (unsloth, trl, datasets)...")
_missing = []
for pkg in ("unsloth", "trl", "datasets"):
    try:
        __import__(pkg)
        print(f"  {pkg}: found")
    except ImportError:
        print(f"  {pkg}: NOT installed (expected in Colab only)")
        _missing.append(pkg)

print()
if _missing:
    print(f"Optional packages missing (install in Colab): {', '.join(_missing)}")
else:
    print("All packages found.")

print("\nAll import checks passed.")
