"""Run frozen experiment checks in separate processes."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("v1_expert_control", "test_moe_control.py"),
    ("v2_feedback_control", "test_experiment.py"),
    ("v3_branching", "test_branching.py"),
    ("v4_state_aggregation", "test_*.py"),
    ("v5_calibration", "test_*.py"),
    ("v6_learning_diagnostics", "test_*.py"),
    ("v7_sequence_feedback", "test_*.py"),
    ("v8_multilayer_retention", "test_*.py"),
)


def main():
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    for directory, pattern in SUITES:
        print(f"\nTesting {directory}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", pattern, "-v"],
            cwd=ROOT / "experiments" / directory, env=env, check=True,
        )


if __name__ == "__main__":
    main()
