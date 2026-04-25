# GRPO Training — Wildfire Containment Simulator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abrodolph/Wildfire-Containment-Simulator/blob/main/training/grpo_colab.ipynb)

## How to run

Open `grpo_colab.ipynb` in Colab (T4 GPU runtime) and run cells in order:

| Section | Cell(s) | What it does |
|---------|---------|--------------|
| 1 — Setup | 1–3 | Installs deps, clones repo, loads Qwen-2.5-1.5B with LoRA |
| 2 — Rollout | 4–5 | Defines `collect_rollout()` using env + serializer + parser |
| 3 — Training | 6–8 | Builds GRPO dataset, trains 50 steps with curriculum |
| 4 — Checkpointing | 9 | Saves final adapter, verifies reload |
| 5 — Plot | 10 | Plots reward curve with tier-promotion markers |

**Resume from checkpoint:** The first cell of Section 3 auto-detects the latest `checkpoints/step_*` folder and loads it. Re-run from that cell to continue training.

## Expected runtime on T4

~45 minutes for 50 GRPO steps (depends on episode length per tier).

## Downloading the trained adapter

After training completes, run in a Colab cell:

```python
from google.colab import files
import shutil
shutil.make_archive('wildfire_adapter', 'zip', 'checkpoints/final')
files.download('wildfire_adapter.zip')
```

## Local validation (no GPU needed)

```bash
python training/test_notebook_imports.py
```

This checks all imports and runs a quick env smoke test without loading model weights.
