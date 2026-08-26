"""Shared constants for indexing tools."""

# ⚠ Left INTACT deliberately. The dot rule below supersedes its dotted members
# for the walk, but this list is matched as a path substring by other callers
# (`index_repo._should_skip`, per-file fallbacks), so removing an entry here is
# a regression in code this change does not touch.
SKIP_PATTERNS = [
    "node_modules/", "vendor/", "venv/", ".venv/", "__pycache__/",
    "dist/", "build/", ".git/", ".tox/", ".mypy_cache/",
    ".gradle/", "target/",
    # Owner ruling 2026-08-25: `.claude/` is agent and editor configuration, not
    # documentation, so it is not corpus. The per-file hook refuses it too
    # (tools/index_file.py); the two entry points have to agree, or a full
    # reindex deletes exactly what the next edit puts back.
    ".claude/",
]

# Dotted directories are skipped by RULE, not by enumeration (jdoc#113).
#
# ⚠⚠ SKIP_PATTERNS used to carry `.venv/` and `.git/` and was the only defence,
# which made it a denylist: it skipped exactly the dotted directories somebody
# thought of in advance and descended into every other one. A sibling tool wrote
# a projection into `.jmemorymunch/` inside an indexed corpus of 243 notes and
# the next index reported 486 documents, every note present twice -- the second
# copy a lossy condensation about a fifth the size. `search_sections` could then
# answer from the condensation with nothing marking it as a summary or as stale.
# `.claude/` is the same hazard in a code repo: agent instructions returned as
# project documentation.
#
# The rule inverts the failure mode. The NEXT tool to write a dotfile cache into
# an indexed tree needs no change here.
DOT_DIR_ALLOWLIST = frozenset({
    # Dotted and legitimately full of documentation: CONTRIBUTING.md, issue
    # templates, workflow docs. ⚠ Skipping this would trade one silent omission
    # for another, which is the defect this rule exists to close.
    ".github",
})


def is_skipped_dot_dir(name: str, include_dot_dirs=None) -> bool:
    """Should a directory named ``name`` be pruned for being dotted?

    ``name`` is a single path COMPONENT, never a path. ⚠ The root the caller
    pointed us at is not a component of any walk-relative path, so a corpus that
    itself lives under a dotted directory (``~/.claude/projects/.../memory``) is
    unaffected -- only directories BELOW the root are candidates. There is a test
    on exactly that, because getting it wrong empties such a corpus silently.

    ``include_dot_dirs`` is the caller's opt-back-in, unioned with the allowlist.
    """
    if not name.startswith(".") or name in (".", ".."):
        return False
    if name in DOT_DIR_ALLOWLIST:
        return False
    if include_dot_dirs and name in set(include_dot_dirs):
        return False
    return True

def should_skip(rel_path: str) -> bool:
    """True when a root-relative path lies under a directory that is not corpus.

    Lives here rather than in the walk because BOTH entry points have to answer
    this identically. The walk prunes on it; the per-file hook refuses on it
    (tools/index_file.py, Rule 3). When only the walk consulted it, a full
    reindex deleted the rows the next edit put straight back, which is how 13
    `.claude/**` rows outlived every reindex across four indexes.
    """
    normalized = "/" + rel_path.replace("\\", "/")
    return any(("/" + pat) in normalized for pat in SKIP_PATTERNS)
