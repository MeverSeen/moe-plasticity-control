"""Build auditable A/B documents without downloading or extracting inputs.

The v10 calibration deliberately protects one old domain (WikiText articles)
while adapting selected MoE experts to CPython source.  This command only
prepares document-level records; it never tokenizes, executes, or downloads
the supplied text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


WIKITEXT_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-103-raw-v1/train-00000-of-00002.parquet"
)
CPYTHON_URL = "https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz"
CPYTHON_VERSION = "3.11.9"
CPYTHON_PREFIX = f"Python-{CPYTHON_VERSION}/Lib/"
_TITLE = re.compile(r"^= ([^=].*?) =$", re.UNICODE)
_EXCLUDED_CPYTHON_PARTS = frozenset({"test", "tests", "site-packages", "ensurepip", "idlelib", "tkinter"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(namespace: str, value: str) -> str:
    return f"{namespace}-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def normalize_title(title: str) -> str:
    """Return a stable WikiText article source key without changing its text."""
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def _raw_wikitext_lines(path: Path) -> Iterator[str]:
    if path.suffix.lower() != ".parquet":
        with path.open("r", encoding="utf-8") as handle:
            yield from (line.rstrip("\n\r") for line in handle)
        return

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Reading parquet WikiText requires pyarrow") from exc
    table = parquet.read_table(path)
    if "text" not in table.column_names:
        raise ValueError("WikiText parquet must contain a 'text' column")
    for item in table.column("text").to_pylist():
        if not isinstance(item, str):
            raise ValueError("WikiText parquet text values must be strings")
        yield from item.splitlines()


def wikitext_documents(path: Path, max_articles: int | None = None,
                       drop_trailing_article: bool = False) -> list[dict]:
    """Reassemble WikiText line rows using only single-level article headings."""
    if max_articles is not None and max_articles <= 0:
        raise ValueError("max_articles must be positive when supplied")

    articles: list[dict] = []
    current_title: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title is None:
            return
        # Preserve source rows verbatim (apart from line-ending normalization);
        # use stripped text only to decide whether an article is empty.
        text = "\n".join(current_lines)
        if text.strip():
            source_title = normalize_title(current_title)
            # WikiText can repeat a title with a different revision/body.  A
            # title is the split source, not a unique document identity.
            # Keeping the content digest in the ID retains distinct revisions
            # while an exact repeated document still de-duplicates downstream.
            content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            articles.append({
                "id": _stable_id("wiki", f"{source_title}\0{content_digest}"),
                "source": f"article:{source_title}",
                "domain": "A",
                "text": text,
            })
        current_title = None
        current_lines = []

    for line in _raw_wikitext_lines(path):
        # WikiText rows can have representation whitespace.  Match a stripped
        # marker while retaining the original line in the document text.
        match = _TITLE.fullmatch(line.strip())
        if match:
            flush()
            if max_articles is not None and len(articles) >= max_articles:
                break
            current_title = match.group(1)
            # Retaining the title makes each output document self-contained.
            current_lines = [line]
        elif current_title is not None:
            # ``== Section ==`` and deeper headings intentionally remain here.
            current_lines.append(line)
        elif line.strip():
            raise ValueError("WikiText content appeared before a single-level article heading")
    else:
        # A parquet shard can end in the middle of an article.  Its last open
        # article is conservatively omitted; development uses max_articles
        # before the shard boundary.
        if not drop_trailing_article:
            flush()

    if max_articles is not None:
        articles = articles[:max_articles]
    if not articles:
        raise ValueError("WikiText input produced no non-empty articles")
    return articles


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe tar member path: {name!r}")
    return path


def _cpython_source(relative: PurePosixPath) -> str:
    first = relative.parts[0]
    if len(relative.parts) == 1:
        return f"module:{relative.stem}"
    return f"package:{first}"


def cpython_documents(path: Path) -> list[dict]:
    """Read eligible CPython 3.11.9 source files directly from a tar archive."""
    try:
        archive = tarfile.open(path, "r:*")
    except tarfile.TarError as exc:
        raise ValueError(f"Cannot read CPython tar archive: {path}") from exc

    documents: list[dict] = []
    with archive:
        members = archive.getmembers()
        top_level: set[str] = set()
        for member in members:
            safe = _safe_member_path(member.name)
            if safe.parts:
                top_level.add(safe.parts[0])
        cpython_roots = {part for part in top_level if part.startswith("Python-")}
        if cpython_roots != {f"Python-{CPYTHON_VERSION}"}:
            raise ValueError(
                f"CPython archive must contain exactly Python-{CPYTHON_VERSION}/; found {sorted(cpython_roots)}"
            )

        for member in members:
            name = member.name
            if not name.startswith(CPYTHON_PREFIX) or not member.isfile() or not name.endswith(".py"):
                continue
            relative = _safe_member_path(name[len(CPYTHON_PREFIX):])
            if not relative.parts or any(part in _EXCLUDED_CPYTHON_PARTS for part in relative.parts):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read tar member {name}")
            try:
                text = stream.read().decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"CPython source is not UTF-8: {name}") from exc
            if not text.strip():
                continue
            documents.append({
                "id": _stable_id("cpython", name),
                "source": _cpython_source(relative),
                "domain": "B",
                "text": text,
            })

    if not documents:
        raise ValueError("CPython archive produced no eligible Lib/*.py files")
    return documents


def _check_document_ids(documents: Iterable[dict]) -> list[dict]:
    """Allow exact duplicates only; reject a stable ID with different content."""
    output: list[dict] = []
    seen: dict[str, dict] = {}
    for document in documents:
        previous = seen.get(document["id"])
        if previous is None:
            seen[document["id"]] = document
            output.append(document)
        elif previous != document:
            raise ValueError(f"Conflicting duplicate document ID: {document['id']}")
    return output


def build_corpus(wikitext_train: Path, cpython_tar: Path, out: Path,
                 max_articles: int | None = None) -> dict:
    """Build a new directory containing ``documents.jsonl`` and provenance."""
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    if not wikitext_train.is_file():
        raise FileNotFoundError(wikitext_train)
    if not cpython_tar.is_file():
        raise FileNotFoundError(cpython_tar)

    is_parquet = wikitext_train.suffix.lower() == ".parquet"
    documents = _check_document_ids(
        [*wikitext_documents(wikitext_train, max_articles=max_articles,
                             drop_trailing_article=is_parquet), *cpython_documents(cpython_tar)]
    )
    counts = Counter(document["domain"] for document in documents)
    provenance = {
        "schema_version": 1,
        "builder": "v10_native_moe_feedback.prepare_corpus",
        "inputs": {
            "wikitext_train": {
                "path": str(wikitext_train), "sha256": sha256_file(wikitext_train),
                "format": "parquet" if is_parquet else "utf8_raw",
                "upstream_url": WIKITEXT_URL,
                "authority": "the supplied input SHA-256 is the authority for this build",
                "license_note": "Preserve and consult the original WikiText distribution and its Wikipedia attribution/license notices.",
            },
            "cpython_tar": {
                "path": str(cpython_tar), "sha256": sha256_file(cpython_tar),
                "upstream_url": CPYTHON_URL,
                "version": CPYTHON_VERSION,
                "license_note": "CPython's original LICENSE remains in the supplied archive; this builder does not remove or rewrite it.",
            },
        },
        "selection": {
            "wikitext_article_heading": "single-level '= Title =' only",
            "max_articles": max_articles,
            "wikitext_trailing_article": (
                "dropped for parquet shards because an EOF article may be incomplete"
                if is_parquet else "retained for raw input"
            ),
            "cpython_prefix": CPYTHON_PREFIX,
            "cpython_excluded_path_parts": sorted(_EXCLUDED_CPYTHON_PARTS),
        },
        "counts": {"documents": len(documents), "by_domain": dict(sorted(counts.items()))},
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        with (temporary / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
        with (temporary / "provenance.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext-train", type=Path, required=True,
                        help="UTF-8 wiki.train.raw or parquet with a text column")
    parser.add_argument("--cpython-tar", type=Path, required=True,
                        help="CPython v3.11.9 source tar archive")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="optional first-N article development subset; default is all articles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = build_corpus(args.wikitext_train, args.cpython_tar, args.out, args.max_articles)
    print(json.dumps(provenance["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
