"""Tests for index_local's `changes` list (v1.67.0 scope).

Covers the per-file change set added to the index_local response:
shape, sort order, status vocabulary, and mtime semantics.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


class TestBuildChangesList:
    def test_sort_order_mtime_desc_then_doc_path_asc(self):
        from jdocmunch_mcp.tools.index_local import _build_changes_list

        mtimes = {
            "a.md": 100.0,   # oldest
            "b.md": 300.0,   # newest
            "c.md": 200.0,
            "d.md": 200.0,   # ties with c.md - tie-break on doc_path asc
        }
        out = _build_changes_list(
            new=["a.md", "b.md"],
            changed=["c.md", "d.md"],
            deleted=[],
            mtimes_by_relpath=mtimes,
        )

        # Expected order: b (300), c (200), d (200), a (100).
        assert [e["doc_path"] for e in out] == ["b.md", "c.md", "d.md", "a.md"]
        # And the statuses came through correctly.
        statuses = {e["doc_path"]: e["status"] for e in out}
        assert statuses == {"a.md": "new", "b.md": "new",
                            "c.md": "changed", "d.md": "changed"}

    def test_deleted_entries_have_null_mtime_and_sort_last(self):
        from jdocmunch_mcp.tools.index_local import _build_changes_list

        out = _build_changes_list(
            new=[],
            changed=["live.md"],
            deleted=["gone.md", "alpha.md"],
            mtimes_by_relpath={"live.md": 500.0},
        )

        # live.md first (has mtime), then deleted entries in doc_path-asc order.
        assert [e["doc_path"] for e in out] == ["live.md", "alpha.md", "gone.md"]
        deleted_entries = [e for e in out if e["status"] == "deleted"]
        assert all(e["mtime"] is None for e in deleted_entries)

    def test_iso_format_is_naive_local(self):
        from datetime import datetime
        from jdocmunch_mcp.tools.index_local import _build_changes_list

        ts = 1_700_000_000.0
        out = _build_changes_list(
            new=["x.md"], changed=[], deleted=[],
            mtimes_by_relpath={"x.md": ts},
        )
        # Should match what datetime.fromtimestamp(ts).isoformat() produces -
        # naive local time, no timezone suffix (matches existing `indexed_at`).
        assert out[0]["mtime"] == datetime.fromtimestamp(ts).isoformat()
        assert "+" not in out[0]["mtime"] and "Z" not in out[0]["mtime"]

    def test_empty_inputs_returns_empty_list(self):
        from jdocmunch_mcp.tools.index_local import _build_changes_list
        assert _build_changes_list(
            new=[], changed=[], deleted=[], mtimes_by_relpath={},
        ) == []

    def test_missing_mtime_yields_null(self):
        # Defensive: if a caller passes a relpath but no mtime entry,
        # we don't crash - we just emit null mtime for that one and it
        # sorts to the bottom alongside deletes.
        from jdocmunch_mcp.tools.index_local import _build_changes_list
        out = _build_changes_list(
            new=["a.md"], changed=[], deleted=[],
            mtimes_by_relpath={},
        )
        assert out == [{"doc_path": "a.md", "status": "new", "mtime": None}]


class TestDiscoverDocFilesMtimes:
    def test_returns_mtimes_dict_keyed_by_path(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import discover_doc_files

        (tmp_path / "a.md").write_text("# A\n")
        (tmp_path / "b.md").write_text("# B\n")
        # Force deterministic mtimes so the assertion is exact.
        os.utime(tmp_path / "a.md", (1_700_000_000.0, 1_700_000_000.0))
        os.utime(tmp_path / "b.md", (1_700_000_100.0, 1_700_000_100.0))

        result = discover_doc_files(tmp_path, max_files=10)

        # Signature now returns 4-tuple: (files, warnings, discovered, mtimes)
        assert len(result) == 4
        files, warnings, discovered, mtimes = result
        assert isinstance(mtimes, dict)
        # Mtimes are keyed by absolute Path (matching `files` entries).
        for fp in files:
            assert fp in mtimes
        # And the values are floats (raw POSIX mtimes).
        for v in mtimes.values():
            assert isinstance(v, float)
        # And they match what we set.
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        assert mtimes[a] == 1_700_000_000.0
        assert mtimes[b] == 1_700_000_100.0


class TestIndexLocalFullIndexChanges:
    def test_first_index_returns_all_files_as_new(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "a.md").write_text("# A\n\nbody\n")
        (repo / "b.md").write_text("# B\n\nbody\n")
        os.utime(repo / "a.md", (1_700_000_000.0, 1_700_000_000.0))
        os.utime(repo / "b.md", (1_700_000_100.0, 1_700_000_100.0))

        result = index_local(
            path=str(repo), name="t3",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
        )

        assert result["success"] is True
        assert "changes" in result
        # Newest first: b.md (1_700_000_100) before a.md (1_700_000_000).
        assert [e["doc_path"] for e in result["changes"]] == ["b.md", "a.md"]
        for e in result["changes"]:
            assert e["status"] == "new"
            assert e["mtime"] is not None


class TestIndexLocalExplicitPathsChanges:
    def test_explicit_paths_populates_mtimes(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "a.md").write_text("# A\n")
        (repo / "b.md").write_text("# B\n")
        os.utime(repo / "a.md", (1_700_000_000.0, 1_700_000_000.0))
        os.utime(repo / "b.md", (1_700_000_100.0, 1_700_000_100.0))

        result = index_local(
            path=str(repo), name="t4",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
            paths=["a.md", "b.md"],
        )

        assert result["success"] is True
        # Both files appear in changes with non-null mtimes (not just empty).
        assert {e["doc_path"] for e in result["changes"]} == {"a.md", "b.md"}
        for e in result["changes"]:
            assert e["mtime"] is not None
        # And ordering is mtime-desc.
        assert [e["doc_path"] for e in result["changes"]] == ["b.md", "a.md"]


class TestIndexLocalIncrementalChanges:
    def _seed(self, tmp_path, storage):
        from jdocmunch_mcp.tools.index_local import index_local
        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "keep.md").write_text("# Keep\n\nbody\n")
        (repo / "edit.md").write_text("# Edit\n\noriginal\n")
        (repo / "drop.md").write_text("# Drop\n\nbye\n")
        os.utime(repo / "keep.md", (1_700_000_000.0, 1_700_000_000.0))
        os.utime(repo / "edit.md", (1_700_000_000.0, 1_700_000_000.0))
        os.utime(repo / "drop.md", (1_700_000_000.0, 1_700_000_000.0))
        index_local(
            path=str(repo), name="t5",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(storage), incremental=False,
        )
        return repo

    def test_modified_new_deleted_appear_in_changes(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = self._seed(tmp_path, tmp_path)

        # Mutate: edit one file, add one new, delete one.
        (repo / "edit.md").write_text("# Edit\n\nUPDATED\n")
        (repo / "fresh.md").write_text("# Fresh\n\nnew body\n")
        (repo / "drop.md").unlink()

        # Assign deterministic, distinguishable mtimes to surviving files.
        os.utime(repo / "edit.md", (1_700_000_200.0, 1_700_000_200.0))
        os.utime(repo / "fresh.md", (1_700_000_300.0, 1_700_000_300.0))

        result = index_local(
            path=str(repo), name="t5",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=True,
        )

        assert result["success"] is True
        assert "changes" in result

        by_path = {e["doc_path"]: e for e in result["changes"]}
        assert by_path["fresh.md"]["status"] == "new"
        assert by_path["edit.md"]["status"] == "changed"
        assert by_path["drop.md"]["status"] == "deleted"
        assert by_path["drop.md"]["mtime"] is None
        assert by_path["fresh.md"]["mtime"] is not None
        assert by_path["edit.md"]["mtime"] is not None

        # Sort order: fresh (300) > edit (200) > drop (None last).
        assert [e["doc_path"] for e in result["changes"]] == [
            "fresh.md", "edit.md", "drop.md",
        ]


class TestIndexLocalNoChangeShortCircuit:
    def test_no_change_returns_empty_changes_list(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "stable.md").write_text("# Stable\n\nbody\n")

        # First pass: full index.
        index_local(
            path=str(repo), name="t6",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
        )

        # Second pass: incremental, with nothing changed on disk.
        result = index_local(
            path=str(repo), name="t6",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=True,
        )

        assert result.get("message") == "No changes detected"
        assert "changes" in result
        assert result["changes"] == []


class TestDocumentationSurfaces:
    def test_index_local_docstring_mentions_changes(self):
        from jdocmunch_mcp.tools.index_local import index_local
        doc = index_local.__doc__ or ""
        assert "changes" in doc, "index_local docstring should describe the new `changes` field"

    def test_mcp_description_mentions_changes(self):
        import asyncio
        from jdocmunch_mcp import server as srv
        tools = asyncio.run(srv.list_tools())
        tool = next(t for t in tools if t.name == "index_local")
        assert "changes" in tool.description, (
            "MCP description for index_local should mention the new `changes` field"
        )


class TestIndexLocalResponseShapeUnchanged:
    """1.x compat audit: every prior top-level key still present."""

    def test_full_index_response_has_all_legacy_keys(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "a.md").write_text("# A\n\nbody\n")

        result = index_local(
            path=str(repo), name="t7full",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
        )

        # Every key that existed in the v1.66.x full-index response.
        for key in [
            "success", "repo", "folder_path", "indexed_at",
            "file_count", "section_count", "doc_types", "files",
            "semantic_search", "_meta",
        ]:
            assert key in result, f"Missing legacy key: {key}"
        # And the new field.
        assert "changes" in result

    def test_incremental_response_has_all_legacy_keys(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "a.md").write_text("# A\n\nbody\n")
        index_local(
            path=str(repo), name="t7inc",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
        )
        (repo / "b.md").write_text("# B\n\nbody\n")
        result = index_local(
            path=str(repo), name="t7inc",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=True,
        )

        for key in [
            "success", "repo", "folder_path", "incremental",
            "changed", "new", "deleted",
            "section_count", "indexed_at", "semantic_search", "_meta",
        ]:
            assert key in result, f"Missing legacy key: {key}"
        assert "changes" in result

    def test_no_change_response_has_all_legacy_keys(self, tmp_path):
        from jdocmunch_mcp.tools.index_local import index_local

        repo = tmp_path / "docs"
        repo.mkdir()
        (repo / "a.md").write_text("# A\n\nbody\n")
        index_local(
            path=str(repo), name="t7noc",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=False,
        )
        result = index_local(
            path=str(repo), name="t7noc",
            use_ai_summaries=False, use_embeddings=False,
            storage_path=str(tmp_path), incremental=True,
        )

        for key in [
            "success", "message", "repo", "folder_path", "incremental",
            "changed", "new", "deleted", "_meta",
        ]:
            assert key in result, f"Missing legacy key: {key}"
        assert "changes" in result
        assert result["changes"] == []
