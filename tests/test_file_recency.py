"""Tests for the file `changes` list in index_local's response."""

from __future__ import annotations

import os
from datetime import datetime


def _index(repo, storage, name, *, incremental, paths=None):
    from jdocmunch_mcp.tools.index_local import index_local
    return index_local(
        path=str(repo), name=name, use_ai_summaries=False, use_embeddings=False,
        storage_path=str(storage), incremental=incremental, paths=paths,
    )


def _write(path, text, mtime=None):
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_build_changes_list_sorting_and_mtimes():
    """Newest first, doc_path breaks ties, deleted entries and entries without
    an mtime last with a null mtime, ISO timestamps without a timezone, and an
    empty input gives an empty list."""
    from jdocmunch_mcp.tools.index_local import _build_changes_list

    out = _build_changes_list(
        new=["a.md", "b.md", "no_mtime.md"],
        changed=["c.md", "d.md"],
        deleted=["gone.md", "alpha.md"],
        mtimes_by_relpath={"a.md": 100.0, "b.md": 300.0, "c.md": 200.0, "d.md": 200.0},
    )
    assert [e["doc_path"] for e in out] == [
        "b.md", "c.md", "d.md", "a.md", "alpha.md", "gone.md", "no_mtime.md",
    ]
    status = {e["doc_path"]: e["status"] for e in out}
    assert status["b.md"] == "new" and status["c.md"] == "changed" and status["gone.md"] == "deleted"
    mtime = {e["doc_path"]: e["mtime"] for e in out}
    assert mtime["b.md"] == datetime.fromtimestamp(300.0).isoformat()
    assert mtime["alpha.md"] is None and mtime["gone.md"] is None and mtime["no_mtime.md"] is None
    assert _build_changes_list(new=[], changed=[], deleted=[], mtimes_by_relpath={}) == []


def test_discover_doc_files_returns_mtimes(tmp_path):
    from jdocmunch_mcp.tools.index_local import discover_doc_files

    _write(tmp_path / "a.md", "# A\n", 1_700_000_000.0)
    _write(tmp_path / "b.md", "# B\n", 1_700_000_100.0)
    files, _warnings, _discovered, mtimes = discover_doc_files(tmp_path, max_files=10)
    assert set(mtimes) == set(files)
    assert mtimes[tmp_path / "a.md"] == 1_700_000_000.0
    assert mtimes[tmp_path / "b.md"] == 1_700_000_100.0


def test_full_index_lists_every_file_as_new(tmp_path):
    repo = tmp_path / "docs"
    repo.mkdir()
    _write(repo / "a.md", "# A\n\nbody\n", 1_700_000_000.0)
    _write(repo / "b.md", "# B\n\nbody\n", 1_700_000_100.0)

    result = _index(repo, tmp_path, "full", incremental=False)

    assert [e["doc_path"] for e in result["changes"]] == ["b.md", "a.md"]
    assert all(e["status"] == "new" and e["mtime"] for e in result["changes"])
    for key in ("success", "repo", "folder_path", "indexed_at", "file_count",
                "section_count", "doc_types", "files", "semantic_search", "_meta"):
        assert key in result, f"existing key missing: {key}"


def test_explicit_paths_list_their_files(tmp_path):
    repo = tmp_path / "docs"
    repo.mkdir()
    _write(repo / "a.md", "# A\n", 1_700_000_000.0)
    _write(repo / "b.md", "# B\n", 1_700_000_100.0)

    result = _index(repo, tmp_path, "paths", incremental=False, paths=["a.md", "b.md"])

    assert [e["doc_path"] for e in result["changes"]] == ["b.md", "a.md"]
    assert all(e["mtime"] for e in result["changes"])


def test_incremental_lists_new_changed_and_deleted(tmp_path):
    repo = tmp_path / "docs"
    repo.mkdir()
    for name in ("keep.md", "edit.md", "drop.md"):
        _write(repo / name, f"# {name}\n\nbody\n", 1_700_000_000.0)
    _index(repo, tmp_path, "inc", incremental=False)

    _write(repo / "edit.md", "# edit.md\n\nupdated\n", 1_700_000_200.0)
    _write(repo / "fresh.md", "# fresh.md\n\nnew\n", 1_700_000_300.0)
    (repo / "drop.md").unlink()
    result = _index(repo, tmp_path, "inc", incremental=True)

    assert [(e["doc_path"], e["status"]) for e in result["changes"]] == [
        ("fresh.md", "new"), ("edit.md", "changed"), ("drop.md", "deleted"),
    ]
    assert result["changes"][2]["mtime"] is None
    for key in ("success", "repo", "folder_path", "incremental", "changed", "new",
                "deleted", "section_count", "indexed_at", "semantic_search", "_meta"):
        assert key in result, f"existing key missing: {key}"


def test_no_change_returns_empty_list(tmp_path):
    repo = tmp_path / "docs"
    repo.mkdir()
    _write(repo / "stable.md", "# Stable\n\nbody\n")
    _index(repo, tmp_path, "same", incremental=False)

    result = _index(repo, tmp_path, "same", incremental=True)

    assert result["message"] == "No changes detected"
    assert result["changes"] == []
    assert result["changes_total"] == 0
    assert result["changes_truncated"] is False
    for key in ("success", "message", "repo", "folder_path", "incremental",
                "changed", "new", "deleted", "_meta"):
        assert key in result, f"existing key missing: {key}"


def test_tool_description_names_the_field():
    import asyncio
    from jdocmunch_mcp import server as srv

    tool = next(t for t in asyncio.run(srv.list_tools()) if t.name == "index_local")
    assert "changes" in tool.description
    assert "changes_total" in tool.description and "changes_truncated" in tool.description


def test_changes_is_capped_with_disclosure(tmp_path):
    """More files than the cap: the list stops at CHANGES_CAP newest first,
    changes_total keeps the uncapped count, changes_truncated says so. Full
    and incremental paths both."""
    from jdocmunch_mcp.tools.index_local import CHANGES_CAP

    count = CHANGES_CAP + 10
    repo = tmp_path / "docs"
    repo.mkdir()
    for i in range(count):
        _write(repo / f"f{i:03d}.md", f"# F{i}\n\nbody\n", 1_700_000_000.0 + i)

    full = _index(repo, tmp_path, "cap", incremental=False)
    assert len(full["changes"]) == CHANGES_CAP
    assert full["changes_total"] == count
    assert full["changes_truncated"] is True
    assert full["changes"][0]["doc_path"] == f"f{count - 1:03d}.md"

    for i in range(count):
        _write(repo / f"f{i:03d}.md", f"# F{i}\n\nedited\n", 1_700_001_000.0 + i)
    inc = _index(repo, tmp_path, "cap", incremental=True)
    assert inc["changed"] == count
    assert len(inc["changes"]) == CHANGES_CAP
    assert inc["changes_total"] == count
    assert inc["changes_truncated"] is True
