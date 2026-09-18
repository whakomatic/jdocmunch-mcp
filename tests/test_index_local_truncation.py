"""Regression tests for jdoc#15: silent truncation when corpus > max_files.

Pre-fix the walker broke early at the cap, returned `success: true`, only
hinted at truncation via a free-text `note`. Callers couldn't detect data
loss programmatically. v1.64.2:

- Default `max_files` raised 500 -> 10_000.
- Response surfaces `truncated`, `discovered`, `indexed` as top-level
  fields whenever the cap is hit.
- Structured warning added to the `warnings` array.
- Walker keeps counting past the cap (up to safety ceiling) so
  `discovered` reflects reality, not the cap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jdocmunch_mcp.tools.index_local import discover_doc_files, index_local


def _seed_md_files(root: Path, n: int) -> None:
    for i in range(n):
        (root / f"doc_{i:05d}.md").write_text(f"# Doc {i}\n\nBody.\n", encoding="utf-8")


def test_discover_returns_discovered_count_when_under_cap(tmp_path):
    _seed_md_files(tmp_path, 5)
    files, warnings, discovered, _ = discover_doc_files(tmp_path, max_files=100)
    assert len(files) == 5
    assert discovered == 5


def test_discover_returns_full_discovered_count_when_over_cap(tmp_path):
    _seed_md_files(tmp_path, 50)
    files, warnings, discovered, _ = discover_doc_files(tmp_path, max_files=10)
    assert len(files) == 10
    assert discovered == 50  # NOT clamped to 10 — jdoc#15 fix


def test_index_local_response_includes_truncated_false_when_under_cap(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    storage = tmp_path / "store"
    _seed_md_files(corpus, 5)
    result = index_local(
        path=str(corpus),
        storage_path=str(storage),
        incremental=False,
        use_ai_summaries=False,
        use_embeddings=False,
    )
    assert result["success"] is True
    assert result["truncated"] is False
    assert "discovered" not in result  # only surfaced when truncation happened
    assert result["file_count"] == 5


def test_index_local_response_surfaces_truncation_when_over_cap(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    storage = tmp_path / "store"
    _seed_md_files(corpus, 60)
    result = index_local(
        path=str(corpus),
        storage_path=str(storage),
        incremental=False,
        max_files=10,
        use_ai_summaries=False,
        use_embeddings=False,
    )
    assert result["success"] is True
    assert result["truncated"] is True
    assert result["discovered"] == 60
    assert result["indexed"] == 10
    # Legacy `note` still present for back-compat:
    assert "note" in result
    assert "60" in result["note"]
    # Structured warning entry alongside the existing warnings array:
    assert any("max_files cap hit" in w for w in result.get("warnings", []))


def test_default_max_files_is_10k(tmp_path):
    """The default raised from 500 to 10_000 (jdoc#15)."""
    # We don't seed 10k files; we just confirm the signature default.
    import inspect

    sig = inspect.signature(index_local)
    assert sig.parameters["max_files"].default == 10_000


def test_discover_hard_ceiling_caps_walks_across_directories(tmp_path):
    """A pathological tree shouldn't run forever. The ceiling is checked
    between directory entries (after each subdir's inner loop), so a single
    huge flat dir is still fully counted, but a 10k-subdir tree stops once
    the count crosses `max_files * 20`."""
    # Seed many subdirs so the ceiling check actually fires.
    for d in range(100):
        sub = tmp_path / f"sub_{d:03d}"
        sub.mkdir()
        _seed_md_files(sub, 10)  # 1000 files total
    _, _, discovered, _ = discover_doc_files(tmp_path, max_files=10)
    # Ceiling = 10 * 20 = 200. Allow one extra dir's worth of overshoot.
    assert discovered <= 200 + 10
    assert discovered >= 200
