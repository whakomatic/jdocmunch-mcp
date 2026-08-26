"""A dot-directory must be excludable, and spelled correctly when it is indexed.

The walk built each candidate's relative path with ``.lstrip("./")``, which
strips a CHARACTER SET rather than a prefix, so ``./.claude/`` arrived at the
matchers as ``claude/``. Nothing could then exclude a root-level dot-directory:
``extra_ignore_patterns=[".claude/"]`` measured no effect in any spelling
(2026-08-25, 124 files admitted from ``E:/ModBuddy/AshesofXCom/.claude``), and
``SKIP_PATTERNS`` could not have pruned ``.git/`` there either. The same call
mis-spelled the stored path of any file inside a dot-directory that WAS indexed.
"""

from pathlib import Path

from jdocmunch_mcp.tools.index_local import discover_doc_files


def _corpus(tmp_path: Path, *rel: str) -> Path:
    for r in rel:
        p = tmp_path / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {p.stem}\n\nProse.\n", encoding="utf-8")
    return tmp_path


def _rels(tmp_path: Path, **kwargs) -> set[str]:
    files, _warnings, _count, _mtimes = discover_doc_files(tmp_path, **kwargs)
    return {Path(f).relative_to(tmp_path).as_posix() for f in files}


def test_extra_ignore_pattern_excludes_a_root_level_dot_directory(tmp_path):
    _corpus(tmp_path, "docs/guide.md", ".claude/settings-notes.md")
    assert _rels(tmp_path, extra_ignore_patterns=[".claude/"]) == {"docs/guide.md"}


def test_extra_ignore_pattern_still_excludes_a_plain_directory(tmp_path):
    """The half that already worked, kept so a fix cannot trade one for the other."""
    _corpus(tmp_path, "docs/guide.md", "current/card.md")
    assert _rels(tmp_path, extra_ignore_patterns=["current/"]) == {"docs/guide.md"}


def test_dot_claude_is_not_documentation_and_is_skipped_by_default(tmp_path):
    """Owner ruling 2026-08-25: `.claude/` content is configuration, not corpus.

    The per-file hook refuses it too (see test_index_file_ownership); this is the
    walk half, and the two must agree or a full reindex deletes what the next edit
    puts back.
    """
    _corpus(tmp_path, "docs/guide.md", ".claude/agents/reviewer/AGENT.md")
    assert _rels(tmp_path) == {"docs/guide.md"}


def test_a_file_inside_an_indexed_dot_directory_keeps_its_leading_dot(tmp_path):
    """`.github/` is not skipped, so its files must be stored spelled as they are."""
    _corpus(tmp_path, ".github/CONTRIBUTING.md")
    assert _rels(tmp_path) == {".github/CONTRIBUTING.md"}
