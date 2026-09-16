"""Run an archived experiment into a fresh directory."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"v1": "v1_expert_control", "v2": "v2_feedback_control", "v3": "v3_branching"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=VERSIONS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true", help="Small v3 preflight (100 epochs).")
    args = parser.parse_args()
    if args.pilot and args.version != "v3":
        parser.error("--pilot is available only for v3")
    output = (args.out or ROOT / "runs" / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    if (ROOT / "experiments").resolve() in output.parents:
        parser.error("Use an output directory outside the frozen experiments archive")
    folder = ROOT / "experiments" / VERSIONS[args.version]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    if args.version == "v3":
        commands = [["branching.py", "--out", str(output)]]
        if args.pilot:
            commands[0] += ["--pilot", "--epochs", "100"]
        commands.append(["analyze.py", "--run", str(output), "--out", str(output / "analysis")])
    else:
        commands = [["run_all.py", "--out", str(output)]]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=folder, env=env, check=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
