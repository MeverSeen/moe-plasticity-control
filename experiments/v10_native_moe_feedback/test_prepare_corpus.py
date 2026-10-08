"""Unit tests for the offline v10 WikiText/CPython corpus builder."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from lm_data import prepare_documents
from prepare_corpus import build_corpus, cpython_documents, wikitext_documents


PREFIX = "Python-3.11.9/Lib/"


def _tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class PrepareCorpusTest(unittest.TestCase):
    def test_wikitext_rebuilds_articles_and_retains_subheadings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "wiki.train.raw"
            raw.write_text(" = Alpha Title = \nfirst paragraph\n\n= = Gameplay = =\nsecond paragraph\n= Beta =\nlast\n", encoding="utf-8")
            documents = wikitext_documents(raw)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["source"], "article:alpha title")
        self.assertIn(" = Alpha Title = ", documents[0]["text"])
        self.assertIn("= = Gameplay = =\nsecond paragraph", documents[0]["text"])
        self.assertEqual(documents[1]["source"], "article:beta")

    def test_repeated_title_with_different_body_has_distinct_ids_and_same_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "wiki.train.raw"
            raw.write_text(
                "= Alpha =\none\n= Middle =\nseparator\n= Alpha =\ntwo\n= End =\nend\n",
                encoding="utf-8",
            )
            documents = wikitext_documents(raw)
        alpha = [item for item in documents if item["source"] == "article:alpha"]
        self.assertEqual(2, len(alpha))
        self.assertNotEqual(alpha[0]["id"], alpha[1]["id"])
        self.assertEqual(["= Alpha =\none", "= Alpha =\ntwo"], [item["text"] for item in alpha])

    def test_cpython_groups_all_package_files_and_excludes_test_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "cpython.tar.gz"
            _tar(archive, {
                PREFIX + "pkg/__init__.py": b"# Copyright package\n",
                PREFIX + "pkg/sub.py": b"x = 1\n",
                PREFIX + "os.py": b"# Copyright module\n",
                PREFIX + "test/no.py": b"bad = True\n",
                PREFIX + "ensurepip/no.py": b"bad = True\n",
            })
            documents = cpython_documents(archive)
        self.assertEqual({item["source"] for item in documents}, {"package:pkg", "module:os"})
        self.assertTrue(any("Copyright" in item["text"] for item in documents))

    def test_rejects_unsafe_tar_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.tar.gz"
            _tar(archive, {"Python-3.11.9/Lib/../escape.py": b"x = 1\n"})
            with self.assertRaisesRegex(ValueError, "Unsafe tar member"):
                cpython_documents(archive)

    def test_empty_wikitext_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "empty.raw"
            raw.write_text("\n\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no non-empty articles"):
                wikitext_documents(raw)

    def test_build_writes_provenance_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wiki = base / "wiki.train.raw"
            wiki.write_text("= Alpha =\ntext\n", encoding="utf-8")
            archive = base / "cpython.tar.gz"
            _tar(archive, {PREFIX + "os.py": b"x = 1\n"})
            out = base / "prepared"
            provenance = build_corpus(wiki, archive, out, max_articles=1)
            documents = [json.loads(line) for line in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(provenance["counts"]["documents"], 2)
            self.assertEqual({item["domain"] for item in documents}, {"A", "B"})
            self.assertTrue((out / "provenance.json").is_file())
            # The builder's actual downstream contract is lm_data's strict
            # A/B validation, not merely a hand-inspected JSONL record.
            splits = prepare_documents(out / "documents.jsonl", seed=7)
            self.assertEqual(set(splits), {"A", "B"})
            self.assertEqual(sum(len(rows) for rows in splits["A"].values()), 1)
            self.assertEqual(sum(len(rows) for rows in splits["B"].values()), 1)
            with self.assertRaises(FileExistsError):
                build_corpus(wiki, archive, out)


if __name__ == "__main__":
    unittest.main()
