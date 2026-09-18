"""Ownership resolution for the per-file re-index hook (two rules).

RULE 1, containment: the owner is the index whose stored ``source_root``
contains the file, deepest root winning; a file under no root is refused; the
stored path is always relative to that root.

RULE 2, worktree refusal: a file inside a linked git worktree is owned only by
an index rooted at that worktree, never by the parent repository's, whose
``source_root`` does contain it.

The second is not implied by the first, which is why both are tested here:
a repo-rooted index's ``source_root`` genuinely contains its worktrees.
"""

from pathlib import Path

from jdocmunch_mcp.storage import DocStore
from jdocmunch_mcp.tools._git import is_linked_worktree, linked_worktree_between
from jdocmunch_mcp.tools.index_file import _find_owning_index, index_file
from jdocmunch_mcp.tools.index_local import discover_doc_files, index_local


def _fake_worktree(parent_root: Path, rel: str) -> Path:
    """A directory shaped like a linked worktree (no real git needed)."""
    wt = parent_root / rel
    wt.mkdir(parents=True)
    (wt / ".git").write_text(
        f"gitdir: {parent_root / '.git' / 'worktrees' / wt.name}\n",
        encoding="utf-8",
    )
    return wt


def _index(root: Path, store_path: Path, name: str) -> dict:
    return index_local(
        str(root), name=name, storage_path=str(store_path),
        use_ai_summaries=False, use_embeddings=False,
    )


class TestIsLinkedWorktree:
    def test_worktree_git_file_matches(self, tmp_path):
        assert is_linked_worktree(_fake_worktree(tmp_path, "wt")) is True

    def test_submodule_git_file_does_not_match(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / ".git").write_text("gitdir: ../.git/modules/sub\n", encoding="utf-8")
        assert is_linked_worktree(sub) is False

    def test_git_directory_does_not_match(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        assert is_linked_worktree(repo) is False


class TestLinkedWorktreeBetween:
    def test_worktree_between_root_and_file(self, tmp_path):
        wt = _fake_worktree(tmp_path, ".worktrees/pkg")
        assert linked_worktree_between(tmp_path, wt / "docs" / "a.md") == wt

    def test_plain_file_under_root(self, tmp_path):
        assert linked_worktree_between(tmp_path, tmp_path / "docs" / "a.md") is None

    def test_root_is_the_worktree(self, tmp_path):
        wt = _fake_worktree(tmp_path, "wt")
        assert linked_worktree_between(wt, wt / "a.md") is None

    def test_path_outside_root(self, tmp_path):
        assert linked_worktree_between(
            tmp_path / "root", tmp_path.parent / "elsewhere" / "a.md"
        ) is None


class TestRule1Containment:
    def test_file_outside_every_source_root_is_refused(self, tmp_path):
        """An edit outside every index root adds nothing to any index.

        An index named for a repo but rooted at its docs/ subdirectory, and a
        file elsewhere in the repo. Folder-name resolution matched the repo
        directory and admitted it; containment does not.
        """
        repo = tmp_path / "myrepo"
        (repo / "docs").mkdir(parents=True)
        (repo / "docs" / "design.md").write_text("# Design\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(repo / "docs", store_path, "myrepo")["success"] is True

        (repo / "notes").mkdir()
        note = repo / "notes" / "scratch.md"
        note.write_text("# Scratch\n", encoding="utf-8")

        assert _find_owning_index(note, DocStore(base_path=str(store_path))) is None

    def test_rel_path_is_relative_to_the_winning_root(self, tmp_path):
        """A file genuinely under the root still indexes, spelled once.

        The old resolver returned the file spelled from the REPO root while
        the walk stored it from the INDEX root, so one file acquired two
        entries and the fresh edit landed on the one nothing read.
        """
        repo = tmp_path / "myrepo"
        (repo / "docs").mkdir(parents=True)
        doc = repo / "docs" / "design.md"
        doc.write_text("# Design\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(repo / "docs", store_path, "myrepo")["success"] is True

        match = _find_owning_index(doc, DocStore(base_path=str(store_path)))
        assert match is not None
        owner, name, rel_path, root = match
        assert (owner, name) == ("local", "myrepo")
        assert rel_path == "design.md"  # NOT "docs/design.md"
        assert root == (repo / "docs").resolve()

    def test_deepest_source_root_wins(self, tmp_path):
        outer = tmp_path / "outer"
        inner = outer / "docs" / "guide"
        inner.mkdir(parents=True)
        (outer / "top.md").write_text("# Top\n", encoding="utf-8")
        doc = inner / "page.md"
        doc.write_text("# Page\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(outer, store_path, "outer")["success"] is True
        assert _index(inner, store_path, "guide")["success"] is True

        match = _find_owning_index(doc, DocStore(base_path=str(store_path)))
        assert match is not None
        assert match[1] == "guide"
        assert match[2] == "page.md"


class TestRule2WorktreeRefusal:
    def test_worktree_file_is_refused_by_the_parent_index(self, tmp_path):
        """An edit inside a linked worktree adds nothing to any index.

        The repo-rooted index's source_root genuinely CONTAINS this file, so
        Rule 1 alone admits it. Only the worktree test refuses it.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "readme.md").write_text("# Readme\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(repo, store_path, "repo")["success"] is True

        wt = _fake_worktree(repo, ".worktrees/pkg")
        wt_doc = wt / "readme.md"
        wt_doc.write_text("# Readme from the worktree\n", encoding="utf-8")

        store = DocStore(base_path=str(store_path))
        # Containment alone would say yes; assert the premise, then the refusal.
        assert wt_doc.resolve().is_relative_to(repo.resolve())
        assert _find_owning_index(wt_doc, store) is None

    def test_an_index_rooted_at_the_worktree_owns_its_own_files(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "readme.md").write_text("# Readme\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(repo, store_path, "repo")["success"] is True

        wt = _fake_worktree(repo, ".worktrees/pkg")
        wt_doc = wt / "readme.md"
        wt_doc.write_text("# Readme from the worktree\n", encoding="utf-8")
        assert _index(wt, store_path, "pkg")["success"] is True

        match = _find_owning_index(wt_doc, DocStore(base_path=str(store_path)))
        assert match is not None
        assert match[1] == "pkg"
        assert match[2] == "readme.md"

    def test_full_walk_indexes_each_file_once(self, tmp_path):
        """A repo-rooted reindex with a worktree present takes one copy.

        SKIP_PATTERNS matches path substrings and cannot express "this
        directory is a separate checkout", so the prune is a real test in the
        walk rather than another entry in that list. The worktree sits at a
        path with no leading dot on purpose: the walk already prunes dotted
        directories such as ``.worktrees/``, so a dotted fixture would pass
        without the worktree check.
        """
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "readme.md").write_text("# Readme\n", encoding="utf-8")
        wt = _fake_worktree(repo, "worktrees/pkg")
        (wt / "readme.md").write_text("# Readme from the worktree\n", encoding="utf-8")

        files, _warnings, _discovered, _mtimes = discover_doc_files(repo)
        rels = sorted(p.relative_to(repo).as_posix() for p in files)
        assert rels == ["readme.md"]


class TestIndexFileEndToEnd:
    """The hook's entry point, not only the resolver: what lands in the index."""

    def _docs_rooted(self, tmp_path):
        repo = tmp_path / "myrepo"
        (repo / "docs").mkdir(parents=True)
        (repo / "docs" / "guide.md").write_text("# Guide\n\nOld.\n", encoding="utf-8")
        (repo / "notes").mkdir()
        (repo / "notes" / "outside.md").write_text("# Outside\n", encoding="utf-8")
        store_path = tmp_path / "store"
        assert _index(repo / "docs", store_path, "myrepo")["success"] is True
        return repo, store_path

    def test_file_outside_the_root_is_not_added(self, tmp_path):
        repo, store_path = self._docs_rooted(tmp_path)
        res = index_file(str(repo / "notes" / "outside.md"),
                         storage_path=str(store_path), use_ai_summaries=False)
        assert res["success"] is False
        idx = DocStore(base_path=str(store_path)).load_index("local", "myrepo")
        assert sorted(idx.doc_paths) == ["guide.md"]

    def test_edit_under_the_root_keeps_one_spelling(self, tmp_path):
        repo, store_path = self._docs_rooted(tmp_path)
        doc = repo / "docs" / "guide.md"
        doc.write_text("# Guide\n\nNew.\n", encoding="utf-8")
        res = index_file(str(doc), storage_path=str(store_path), use_ai_summaries=False)
        assert res["success"] is True
        assert res["file"] == "guide.md"
        idx = DocStore(base_path=str(store_path)).load_index("local", "myrepo")
        assert sorted(idx.doc_paths) == ["guide.md"]
