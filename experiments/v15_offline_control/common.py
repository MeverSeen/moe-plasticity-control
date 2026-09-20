"""Reuse the frozen v14 interface without modifying its archive."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V14 = HERE.parent / "v14_methods_20260920_0814"
sys.path.insert(0, str(V14))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
