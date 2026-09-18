"""Gitignored dot-directories must be pruned during discovery (jdoc#102).

`f"{dir_rel}/{d}/".lstrip("./")` ate the leading dot of the first path
component, because `str.lstrip` takes a character SET rather than a prefix. So
`.venv/`, `.tox/`, `.next/`, `.cache/` and `.worktrees/` were walked and indexed
while their undotted equivalents were pruned correctly -- which is precisely why
it read as working.

The undotted cases are kept here as controls: they passed before the fix and
must keep passing, so this file cannot go green by discovery breaking entirely.
"""

import pytest

from jdocmunch_mcp.tools.index_local import _walk_rel


class TestWalkRel:
    """The helper, in isolation. These are the assertions lstrip fails."""

    @pytest.mark.parametrize("dir_rel,name,expected", [
        (".", ".worktrees/", ".worktrees/"),
        (".", ".venv/", ".venv/"),
        (".", "keepdir/", "keepdir/"),
        (".", "README.md", "README.md"),
        ("", ".tox/", ".tox/"),
        (".worktrees/slug-a", "copy.md", ".worktrees/slug-a/copy.md"),
        ("keepdir", "kept.md", "keepdir/kept.md"),
        ("src/.hidden", "x.md", "src/.hidden/x.md"),
    ])
    def test_builds_the_path_git_would_match(self, dir_rel, name, expected):
        assert _walk_rel(dir_rel, name) == expected

    def test_does_not_use_character_set_stripping(self):
        """The exact corruption, pinned so it cannot come back."""
        assert f"./{'.worktrees/'}".lstrip("./") == "worktrees/"   # the old behavior
        assert _walk_rel(".", ".worktrees/") == ".worktrees/"      # the fixed one

    def test_no_leading_slash_at_root(self):
        """A leading slash would fail to match a relative gitignore pattern."""
        assert not _walk_rel(".", "x.md").startswith("/")
        assert not _walk_rel("", "x.md").startswith("/")


def _make_repo(tmp_path, ignored_dirname: str):
    """Reporter's fixture: one kept doc, one doc inside a gitignored directory."""
    root = tmp_path / "repo"
    (root / "keepdir").mkdir(parents=True)
    (root / ignored_dirname / "slug-a").mkdir(parents=True)
    (root / ".gitignore").write_text(f"{ignored_dirname}/\n", encoding="utf-8")
    (root / "keepdir" / "kept.md").write_text("# Kept\n\nalpha\n", encoding="utf-8")
    (root / ignored_dirname / "slug-a" / "copy.md").write_text(
        "# Copy\n\nbeta\n", encoding="utf-8"
    )
    return root


def _discovered(root):
    """Root-relative POSIX paths discovery returns. No skip/shape guards: an
    entry point that moves should fail this file loudly, not skip it."""
    from jdocmunch_mcp.tools.index_local import discover_doc_files

    files, _warnings, _count, _mtimes = discover_doc_files(root, max_files=10_000)
    return [str(f).replace("\\", "/") for f in files]


class TestDiscoveryPrunesIgnoredDirs:
    @pytest.mark.parametrize("ignored", [".worktrees", ".venv", ".tox", ".next", ".cache"])
    def test_dotted_ignored_dir_is_pruned(self, tmp_path, ignored):
        """The defect: every one of these leaked before the fix."""
        found = _discovered(_make_repo(tmp_path, ignored))
        assert any("kept.md" in f for f in found), "control document went missing"
        assert not any("copy.md" in f for f in found), (
            f"{ignored}/ was indexed despite being gitignored"
        )

    @pytest.mark.parametrize("ignored", ["ignoreddir", "build", "node_modules"])
    def test_undotted_ignored_dir_still_pruned(self, tmp_path, ignored):
        """Control: these worked before the fix and must keep working."""
        found = _discovered(_make_repo(tmp_path, ignored))
        assert any("kept.md" in f for f in found)
        assert not any("copy.md" in f for f in found)

    def test_nested_file_under_dotted_dir_does_not_escape_via_the_file_check(self, tmp_path):
        """The per-file fallback was corrupted the same way.

        Even if directory pruning were bypassed, `.worktrees/a/copy.md` became
        `worktrees/a/copy.md` and missed the pattern there too.
        """
        root = _make_repo(tmp_path, ".worktrees")
        deep = root / ".worktrees" / "slug-a" / "nested" / "deeper"
        deep.mkdir(parents=True)
        (deep / "buried.md").write_text("# Buried\n\ngamma\n", encoding="utf-8")
        found = _discovered(root)
        assert not any("buried.md" in f for f in found)

    def test_a_dot_directory_not_in_gitignore_is_still_indexed(self, tmp_path):
        """The fix must prune what git ignores, not everything starting with a dot."""
        root = tmp_path / "repo2"
        (root / ".github").mkdir(parents=True)
        (root / ".gitignore").write_text("build/\n", encoding="utf-8")
        (root / ".github" / "notes.md").write_text("# Notes\n\ndelta\n", encoding="utf-8")
        found = _discovered(root)
        assert any("notes.md" in f for f in found), (
            "a dot-directory git does NOT ignore was pruned; the fix overshot"
        )
