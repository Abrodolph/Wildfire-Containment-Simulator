import os
import subprocess
import sys


def test_synthetic_dashboard(tmp_path):
    output = tmp_path / "training_dashboard.png"
    result = subprocess.run(
        [sys.executable, "scripts/plot_dashboard.py", "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert output.exists(), "Dashboard PNG not created"
    assert output.stat().st_size > 50_000, f"PNG too small: {output.stat().st_size} bytes"
