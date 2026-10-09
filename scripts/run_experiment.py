"""Run an archived experiment into a fresh directory."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"v1": "v1_expert_control", "v2": "v2_feedback_control", "v3": "v3_branching",
            "v4": "v4_state_aggregation", "v5": "v5_calibration", "v6": "v6_learning_diagnostics", "v7": "v7_sequence_feedback",
            "v8": "v8_multilayer_retention", "v9": "v9_learning_pressure", "v10": "v10_native_moe_feedback",
            "v11": "v11_rl_diagnostics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=VERSIONS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true", help="Small v3–v9 preflight.")
    parser.add_argument("--download", action="store_true", help="Allow v8/v9 data or v10 pinned model download.")
    parser.add_argument("--mode", choices=['inspect', 'smoke', 'preflight', 'baselines', 'branches'], help='v10 only; defaults to inspect.')
    parser.add_argument("--data", type=Path, help='v10 prepared A/B JSONL corpus.')
    parser.add_argument("--device", help='v10 CUDA device, such as cuda:0.')
    args = parser.parse_args()
    if args.pilot and args.version not in {"v3", "v4", "v5", "v6", "v7", "v8", "v9"}:
        parser.error("--pilot is available only for v3–v9")
    if args.download and args.version not in {"v8", "v9", "v10"}:
        parser.error("--download is available only for v8/v9/v10")
    if args.version != "v10" and (args.mode or args.data or args.device):
        parser.error('--mode/--data/--device are available only for v10')
    output = (args.out or ROOT / "runs" / f"{args.version}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")
    if (ROOT / "experiments").resolve() in output.parents:
        parser.error("Use an output directory outside the frozen experiments archive")
    folder = ROOT / "experiments" / VERSIONS[args.version]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    if args.version == 'v11':
        commands = [['run_diagnostics.py', '--out', str(output)],
                    ['analyze.py', '--run', str(output), '--out', str(output/'analysis')],
                    [str(ROOT/'scripts/audit_v11_run.py'), '--run', str(output)]]
    elif args.version == 'v10':
        command = ['run_native.py', '--mode', args.mode or 'inspect', '--out', str(output)]
        if args.data:
            command += ['--data', str(args.data.resolve())]
        if args.device:
            command += ['--device', args.device]
        if args.download:
            command += ['--download']
        commands = [command]
    elif args.version in {"v3", "v4", "v5", "v6", "v7", "v8", "v9"}:
        entry = {"v3": "branching.py", "v4": "dagger.py", "v5": "calibrate.py", "v6": "diagnose.py", "v7": "run_sequence.py", "v8": "run_retention.py", "v9": "run_pressure.py"}[args.version]
        commands = [[entry, "--out", str(output)]]
        if args.pilot:
            commands[0] += ["--pilot"]
            if args.version in {"v3", "v4"}:
                commands[0] += ["--epochs", "100"]
        if args.download:
            commands[0] += ["--download"]
        analyzer = "analyze_pressure.py" if args.version == "v9" else "analyze.py"
        commands.append([analyzer, "--run", str(output), "--out", str(output / "analysis")])
        if args.version == "v7":
            commands.append([str(ROOT / "scripts" / "analyze_sequence_diagnostics.py"), "--run", str(output)])
    else:
        commands = [["run_all.py", "--out", str(output)]]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=folder, env=env, check=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
