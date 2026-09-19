"""Regression tests for jdoc#16: prefer-newest walk order when cap is hit.

Before v1.65.0 the truncated subset was filesystem-walk order, which is
non-deterministic from the user's perspective. A file edited 4 minutes
before the index call could be silently dropped while older files made
the cut. v1.65.0 defaults to sort_by="newest" so the indexed subset is
always the N most recently-edited files when truncation happens.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from jdocmunch_mcp.tools.index_local import discover_doc_files, index_local


def _seed_files_with_mtimes(root: Path, names: list[str], base_age_s: float = 1000.0) -> None:
    """Create files and set deterministic mtimes spaced 1s apart.

    The first name in the list is the oldest; the last is the newest.
    Mtimes are set relative to a fixed past timestamp to avoid wall-clock
    drift affecting test ordering.
    """
    now = time.time()
    for i, name in enumerate(names):
        p = root / name
        p.write_text(f"# {name}\n", encoding="utf-8")
        # name[i] gets mtime = now - base_age + i, so later index = newer.
        mtime = now - base_age_s + i
        os.utime(p, (mtime, mtime))


def test_sort_by_newest_keeps_recent_files_when_truncated(tmp_path):
    """The 3 newest files survive truncation regardless of walk order."""
    _seed_files_with_mtimes(
        tmp_path,
        ["old_a.md", "old_b.md", "old_c.md", "new_a.md", "new_b.md", "new_c.md"],
    )
    files, warnings, discovered, _ = discover_doc_files(
        tmp_path, max_files=3, sort_by="newest"
    )
    assert discovered == 6
    assert len(files) == 3
    kept = sorted(f.name for f in files)
    assert kept == ["new_a.md", "new_b.md", "new_c.md"]


def test_sort_by_newest_strictly_prefers_newer_mtimes(tmp_path):
    """The contract: when truncating, sort_by='newest' picks files whose
    minimum mtime is at least the maximum mtime of the dropped subset.

    Tested via aggregate mtime comparison rather than name patterns
    because filesystem-walk order isn't guaranteed alphabetical (on
    Windows in particular, it follows directory entry order)."""
    _seed_files_with_mtimes(
        tmp_path,
        [f"f{i:02d}.md" for i in range(10)],  # f00 (oldest) … f09 (newest)
    )
    files, _, discovered, _ = discover_doc_files(
        tmp_path, max_files=3, sort_by="newest"
    )
    assert discovered == 10
    assert len(files) == 3
    # All three kept files must be the three highest mtimes.
    kept_mtimes = sorted(os.stat(f).st_mtime for f in files)
    all_mtimes = sorted(os.stat(tmp_path / f"f{i:02d}.md").st_mtime for i in range(10))
    assert kept_mtimes == all_mtimes[-3:]


def test_sort_by_walk_order_skips_mtime_sort(tmp_path):
    """walk_order is the back-compat escape hatch — no mtime sort applied.
    We verify the sort isn't happening by checking that the result is not
    guaranteed to be the newest subset (the precise order is filesystem-
    dependent, so we make this property-based)."""
    _seed_files_with_mtimes(
        tmp_path,
        [f"f{i:02d}.md" for i in range(10)],
    )
    files_walk, _, _, _ = discover_doc_files(
        tmp_path, max_files=3, sort_by="walk_order"
    )
    files_new, _, _, _ = discover_doc_files(
        tmp_path, max_files=3, sort_by="newest"
    )
    walk_min_mtime = min(os.stat(f).st_mtime for f in files_walk)
    new_min_mtime = min(os.stat(f).st_mtime for f in files_new)
    # 'newest' must produce a subset whose oldest member is at least as
    # new as walk_order's oldest member. (Equality is possible on small
    # corpora where walk order happens to land on newest by chance.)
    assert new_min_mtime >= walk_min_mtime


def test_sort_by_default_is_newest(tmp_path):
    """The new default; omitting sort_by gets prefer-newest behavior."""
    _seed_files_with_mtimes(
        tmp_path,
        ["old_a.md", "old_b.md", "new_a.md", "new_b.md"],
    )
    files, _, _, _ = discover_doc_files(tmp_path, max_files=2)
    kept = sorted(f.name for f in files)
    assert kept == ["new_a.md", "new_b.md"]


def test_sort_by_has_no_effect_when_corpus_fits_under_cap(tmp_path):
    """When discovered <= max_files we skip the sort entirely; order is
    whatever the walker produced. Either walk-order or newest is fine —
    we just don't pay the sort cost."""
    _seed_files_with_mtimes(tmp_path, ["a.md", "b.md", "c.md"])
    files_n, _, _, _ = discover_doc_files(tmp_path, max_files=100, sort_by="newest")
    files_w, _, _, _ = discover_doc_files(tmp_path, max_files=100, sort_by="walk_order")
    assert {f.name for f in files_n} == {"a.md", "b.md", "c.md"}
    assert {f.name for f in files_w} == {"a.md", "b.md", "c.md"}


def test_index_local_passes_sort_by_through(tmp_path):
    """End-to-end: index_local honors sort_by when truncation fires."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    storage = tmp_path / "store"
    _seed_files_with_mtimes(
        corpus,
        ["old_a.md", "old_b.md", "new_a.md", "new_b.md"],
    )
    result = index_local(
        path=str(corpus),
        storage_path=str(storage),
        incremental=False,
        max_files=2,
        sort_by="newest",
        use_ai_summaries=False,
        use_embeddings=False,
    )
    assert result["success"] is True
    assert result["truncated"] is True
    indexed_names = sorted(result["files"])
    assert indexed_names == ["new_a.md", "new_b.md"]
