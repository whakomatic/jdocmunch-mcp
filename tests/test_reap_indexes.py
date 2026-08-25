"""The reaper's predicate, which is the whole risk in the feature.

Nothing else ever removes an index, so dead ones accumulate forever: a test run
that indexes a fixture project under the OS temp directory leaves its index
behind, and 34 had built up on the machine this was written for.

The danger is the obvious implementation. "The root does not exist" is not the
predicate, because `Path.exists()` answers False both for a path that is
provably absent and for one it could not read. The corpora on that machine live
under OneDrive, where a placeholder or offline file raises rather than
answering, so a reaper built on `exists()` deletes live indexes the first time
the sync client is mid-flight. Every test here exists to pin one half of that
distinction.
"""

import os
from pathlib import Path


from jdocmunch_mcp.storage import DocStore
from jdocmunch_mcp.tools.index_local import index_local
from jdocmunch_mcp.tools.reap_indexes import (
    FINDING_NO_SOURCE_ROOT,
    FINDING_ROOT_GONE_OUTSIDE_TEMP,
    FINDING_ROOT_UNREADABLE,
    REASON_TEMP_ROOT_GONE,
    classify,
    reap_indexes,
)


class TestPredicate:
    def test_a_present_root_is_neither_reaped_nor_reported(self, tmp_path):
        assert classify(str(tmp_path)) == (None, None)

    def test_a_gone_root_under_temp_is_reaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        reason, finding = classify(str(tmp_path / "vanished"))
        assert reason == REASON_TEMP_ROOT_GONE and finding is None

    def test_a_gone_root_outside_temp_is_reported_not_reaped(self, tmp_path, monkeypatch):
        """"Gone" and "gone and never coming back" are different claims, and only
        the second justifies a delete. A vanished corpus somewhere a person keeps
        their work is far more likely an unmounted drive than a dead index.

        `tmp_path` is itself under the OS temp directory, so temp has to be
        pointed elsewhere for this case to exist at all.
        """
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "elsewhere"))
        reason, finding = classify(str(tmp_path / "vanished"))
        assert reason is None and finding == FINDING_ROOT_GONE_OUTSIDE_TEMP

    def test_an_unreadable_root_is_never_reaped(self, tmp_path, monkeypatch):
        """THE test. An OSError that is not ENOENT means the question could not
        be answered, and 'could not tell' must never resolve to 'delete it'.

        The root is under temp here, so every other condition for reaping is
        satisfied: only the three-way absence answer stops it.
        """
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        target = tmp_path / "unreadable"

        real_stat = os.stat

        def _raise(path, *args, **kwargs):
            if str(path) == str(target):
                raise OSError(5, "device I/O error")   # not ENOENT
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr("jdocmunch_mcp.tools.reap_indexes.os.stat", _raise)
        reason, finding = classify(str(target))
        assert reason is None, "an unreadable root must never be reaped"
        assert finding == FINDING_ROOT_UNREADABLE

    def test_an_index_with_no_recorded_root_is_never_reaped(self):
        reason, finding = classify("")
        assert reason is None and finding == FINDING_NO_SOURCE_ROOT


class TestReapPass:
    def _corpus(self, root: Path, store_path: Path, name: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "a.md").write_text("# A\n", encoding="utf-8")
        result = index_local(str(root), name=name, storage_path=str(store_path),
                             use_ai_summaries=False)
        assert result["success"] is True, result

    def test_dry_run_by_default_deletes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        store_path = tmp_path / "store"
        doomed = tmp_path / "doomed"
        self._corpus(doomed, store_path, "doomed")
        import shutil
        shutil.rmtree(doomed)

        result = reap_indexes(storage_path=str(store_path))

        assert result["applied"] is False
        assert [e["repo"] for e in result["reaped"]] == ["local/doomed"]
        assert DocStore(base_path=str(store_path)).load_index("local", "doomed") is not None

    def test_apply_removes_the_index_and_leaves_the_live_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        store_path = tmp_path / "store"
        doomed, kept = tmp_path / "doomed", tmp_path / "kept"
        self._corpus(doomed, store_path, "doomed")
        self._corpus(kept, store_path, "kept")
        import shutil
        shutil.rmtree(doomed)

        result = reap_indexes(apply=True, storage_path=str(store_path))

        assert result["applied"] is True
        assert [e["repo"] for e in result["reaped"]] == ["local/doomed"]
        store = DocStore(base_path=str(store_path))
        assert store.load_index("local", "doomed") is None
        assert store.load_index("local", "kept") is not None

    def test_the_delete_takes_the_sidecars_with_it(self, tmp_path, monkeypatch):
        """A reaper that removed only the monolith would leave behind the very
        mirror it exists to collect."""
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        store_path = tmp_path / "store"
        doomed = tmp_path / "doomed"
        self._corpus(doomed, store_path, "doomed")
        owner_dir = store_path / "local"
        assert list(owner_dir.glob("doomed*"))
        import shutil
        shutil.rmtree(doomed)

        reap_indexes(apply=True, storage_path=str(store_path))

        # The `.lock` file survives on purpose and is not a leftover: it is the
        # stable per-repo coordination object, and unlinking it mid-critical-
        # section is how two writers end up holding different inodes of the same
        # lock. Everything that holds CONTENT has to be gone.
        left = sorted(p.name for p in owner_dir.glob("doomed*")
                      if p.suffix != ".lock")
        assert left == [], left
