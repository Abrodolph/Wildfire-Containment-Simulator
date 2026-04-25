import json
import os
import subprocess
import sys
import tempfile


def test_quick_mode_runs(tmp_path):
    output = tmp_path / "eval_results.json"
    result = subprocess.run(
        [sys.executable, "scripts/eval_compare.py", "--quick", "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert output.exists(), "Output JSON not created"

    with open(output) as f:
        data = json.load(f)

    assert "random" in data, "random agent missing from results"
    assert "heuristic" in data, "heuristic agent missing from results"

    for agent in ("random", "heuristic"):
        tier_data = data[agent].get("easy")
        assert tier_data is not None, f"{agent} easy tier is None"
        assert tier_data["total_reward"] is not None
