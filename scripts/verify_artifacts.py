"""Verify imported experiment artifacts without loading model checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = json.loads((ROOT / "docs" / "import_manifest.json").read_text())["files"]
    errors = []
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Changed: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(files)} imported experiment files.")


if __name__ == "__main__":
    main()
