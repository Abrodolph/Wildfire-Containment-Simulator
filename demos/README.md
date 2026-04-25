# Demo Assets

## Regenerating demo assets

```bash
# Find the best demo seed (scans seeds 0-499, takes ~5 min)
python scripts/find_demo_seed.py

# Run demo with default seed (DEMO_SEED = 7)
python scripts/run_demo.py

# Run with a specific seed
python scripts/run_demo.py --seed 42

# Run trained LLM comparison (requires TRAINED_MODEL_PATH env var)
python scripts/run_demo.py --agent trained_llm
```

## Output files

| File | Description |
|------|-------------|
| `heuristic_demo.gif` | Animated replay — heuristic agent on demo seed |
| `heuristic_demo.png` | Final frame PNG |
| `trained_demo.gif` | Animated replay — trained LLM agent (post-training) |
| `candidate_seeds.json` | Top 5 seeds from the seed finder scan |

## Demo seed criteria

The chosen seed (`DEMO_SEED = 7`) was selected because:
- Wind shift fires between step 60-90, creating a mid-episode pivot moment
- Heuristic loses at least one populated cell (shows room for improvement)
- Total reward in the "flawed but not catastrophic" range (-4 to +2)
