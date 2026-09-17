"""Verify imported experiment artifacts without loading model checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = json.loads((ROOT / "docs" / "import_manifest.json").read_text())["files"]
    for version in ("v4_state_aggregation", "v5_calibration", "v6_learning_diagnostics"):
        folder = ROOT / "experiments" / version
        for line in (folder / "manifest.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            files[(folder / relative).relative_to(ROOT).as_posix()] = expected
    errors = []
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Changed: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(files)} archived experiment files.")


if __name__ == "__main__":
    main()
