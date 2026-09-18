"""``list_repos`` must not pay a full parse per call for an unsummarized index.

An index saved before the summary sidecar existed (jdoc#77) has no
``<name>.summary.json``, so ``list_repos`` falls back to parsing the whole
monolith. Only a save of that index writes the sidecar, so the fallback
ran on every call, for the life of the index. The fallback now writes the
sidecar it just paid for.

This is cost rather than wrongness, so it would not surface as a failing
assertion elsewhere. These tests assert the mechanism: the fallback heals,
and the healed sidecar produces the same row as a saved one.

The other ``list_repos`` cost, derived sidecars matching the ``*/*.json``
glob, is covered by ``test_jdoc_121_list_repos_sidecars.py``.
"""

import json
from pathlib import Path

from jdocmunch_mcp.storage import DocStore
from jdocmunch_mcp.tools.index_local import index_local


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n", encoding="utf-8")
    store_path = tmp_path / "store"
    result = index_local(
        str(root), name="corpus", storage_path=str(store_path),
        use_ai_summaries=False, use_embeddings=False,
    )
    assert result["success"] is True
    return root, store_path


def _index_path(store_path: Path) -> Path:
    return store_path / "local" / "corpus.json"


class TestSummarySidecarHeals:
    def test_a_missing_sidecar_is_written_by_the_fallback(self, tmp_path):
        _root, store_path = _corpus(tmp_path)
        summary = _index_path(store_path).with_name("corpus.summary.json")
        assert summary.exists()
        saved = json.loads(summary.read_text(encoding="utf-8"))

        summary.unlink()  # an index last written before the sidecar existed
        store = DocStore(base_path=str(store_path))
        rows = store.list_repos()
        assert {r["repo"] for r in rows} == {"local/corpus"}

        assert summary.exists(), "the fallback must heal what it paid for"
        healed = json.loads(summary.read_text(encoding="utf-8"))
        # Same field list, both directions: the save path reads a DocIndex and
        # the heal reads the monolith dict, through one payload builder.
        assert healed == saved

    def test_the_healed_sidecar_produces_the_same_row(self, tmp_path):
        _root, store_path = _corpus(tmp_path)
        store = DocStore(base_path=str(store_path))
        from_saved = store.list_repos()

        _index_path(store_path).with_name("corpus.summary.json").unlink()
        from_monolith = store.list_repos()   # full parse, then heals
        from_healed = store.list_repos()     # reads the healed sidecar

        assert from_monolith == from_saved
        assert from_healed == from_saved
