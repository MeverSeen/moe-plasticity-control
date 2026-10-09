"""Run frozen experiment checks in separate processes."""
from pathlib import Path
import argparse
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
    ("v9_learning_pressure", "test_*.py"),
    ("v11_rl_diagnostics", "test_*.py"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-v10', action='store_true', help='Also test native-MoE code; install its separate requirements first.')
    args = parser.parse_args()
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    suites = SUITES + (("v10_native_moe_feedback", "test_*.py"),) if args.include_v10 else SUITES
    for directory, pattern in suites:
        print(f"\nTesting {directory}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", pattern, "-v"],
            cwd=ROOT / "experiments" / directory, env=env, check=True,
        )


if __name__ == "__main__":
    main()
