"""Small Git helpers for local indexing metadata."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from ..storage.doc_store import normalize_commit_sha

DEFAULT_GIT_TIMEOUT = 10.0


def _git_timeout() -> Optional[float]:
    """Resolve the per-call git subprocess ceiling (seconds).

    Bounded by default so a blocked git (index.lock contention, credential
    prompt, LFS smudge) can never hang the synchronous tool path. Override with
    ``JDOCMUNCH_GIT_TIMEOUT``; a value <= 0 disables the ceiling entirely.
    """
    raw = os.environ.get("JDOCMUNCH_GIT_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_GIT_TIMEOUT
    try:
        val = float(raw)
    except ValueError:
        return DEFAULT_GIT_TIMEOUT
    return val if val > 0 else None


# Probe outcome kinds (jdoc#80 Part B): a failed git call is either a clean
# "git ran and this is not a usable repo" answer (a determination) or an
# "unavailable" result (git could not answer: timeout, missing binary, OS
# error). Only the latter is treated as failed-verification worth quarantining.
GIT_OK = "ok"
GIT_NOT_A_REPO = "not_a_repo"
GIT_UNAVAILABLE = "unavailable"


# Commit-relationship outcomes (jdoc#86 modern supersession). Every value is a
# PROOF class: "unproven" is the fail-closed answer whenever git could not make
# a determination (missing commit, unavailable binary, timeout, equal SHAs).
ANCESTRY_ANCESTOR = "ancestor"          # a is a strict ancestor of b
ANCESTRY_DESCENDANT = "descendant"      # b is a strict ancestor of a
ANCESTRY_UNRELATED = "unrelated"        # both directions provably false
ANCESTRY_UNPROVEN = "unproven"          # no determination — never destructive


def commit_ancestry(folder_path: Path, sha_a: str, sha_b: str) -> str:
    """Classify the Git relationship between two commits, fail-closed.

    Runs in the repository containing ``folder_path``. Both SHAs must be
    40-hex, distinct, and resolvable to commits in this repository; anything
    short of a definitive two-directional determination returns UNPROVEN so a
    caller can never treat uncertainty as an ordering.
    """
    a = (sha_a or "").strip().lower()
    b = (sha_b or "").strip().lower()
    hex40 = "0123456789abcdef"
    if len(a) != 40 or len(b) != 40 or a == b:
        return ANCESTRY_UNPROVEN
    if any(c not in hex40 for c in a) or any(c not in hex40 for c in b):
        return ANCESTRY_UNPROVEN
    for sha in (a, b):
        ok, _, kind = _git_probe(folder_path, ["cat-file", "-e", f"{sha}^{{commit}}"])
        if not ok:
            # NOT_A_REPO here means "commit absent" — still no proof basis.
            return ANCESTRY_UNPROVEN
    ok_ab, _, kind_ab = _git_probe(
        folder_path, ["merge-base", "--is-ancestor", a, b]
    )
    if not ok_ab and kind_ab == GIT_UNAVAILABLE:
        return ANCESTRY_UNPROVEN
    if ok_ab:
        return ANCESTRY_ANCESTOR
    ok_ba, _, kind_ba = _git_probe(
        folder_path, ["merge-base", "--is-ancestor", b, a]
    )
    if not ok_ba and kind_ba == GIT_UNAVAILABLE:
        return ANCESTRY_UNPROVEN
    if ok_ba:
        return ANCESTRY_DESCENDANT
    return ANCESTRY_UNRELATED


def _git_probe(cwd: Path, args: list[str]) -> tuple[bool, str, str]:
    """Like ``_git`` but classifies the failure mode.

    Returns ``(ok, stdout, kind)`` where ``kind`` is ``GIT_OK`` on success,
    ``GIT_NOT_A_REPO`` when git ran and made a determination (a non-zero
    ``fatal`` exit, e.g. 128 outside a repository — a definitive "no"), or
    ``GIT_UNAVAILABLE`` when git could not produce an answer at all (missing
    binary, timeout, or OS error). The distinction lets a caller tell a
    confirmed non-Git corpus from a transient verification failure without
    depending on git's (localizable) stderr text.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,  # see _git: prevents stdio-server deadlock
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            timeout=_git_timeout(),
            check=True,
        )
    except subprocess.CalledProcessError:
        # git ran and exited non-zero: a determination (not-a-repo / bad path).
        return False, "", GIT_NOT_A_REPO
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # git could not answer: binary missing, timed out, or OS error.
        return False, "", GIT_UNAVAILABLE
    except Exception:
        return False, "", GIT_UNAVAILABLE
    return True, proc.stdout.strip(), GIT_OK


def _git(cwd: Path, args: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            # stdin must be redirected: when the MCP server runs over stdio,
            # an un-redirected git child inherits the JSON-RPC pipe as its
            # stdin and Git for Windows blocks on it indefinitely (the
            # timeout's kill-then-drain also wedges on the inherited handle).
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            timeout=_git_timeout(),
            check=True,
        )
    except Exception:
        return False, ""
    return True, proc.stdout.strip()


def _git_bytes(cwd: Path, args: list[str]) -> tuple[bool, bytes]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,  # see _git: prevents stdio-server deadlock
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_git_timeout(),
            check=True,
        )
    except Exception:
        return False, b""
    return True, proc.stdout


def _git_root(folder_path: Path) -> Optional[Path]:
    ok, inside = _git(folder_path, ["rev-parse", "--is-inside-work-tree"])
    if not ok or inside != "true":
        return None
    ok, root = _git(folder_path, ["rev-parse", "--show-toplevel"])
    return Path(root).resolve() if ok and root else None


def local_git_head(folder_path: Path) -> Optional[str]:
    """Return the current HEAD SHA for a local Git worktree, if available."""
    folder_path = folder_path.resolve()
    if _git_root(folder_path) is None:
        return None
    ok, head = _git(folder_path, ["rev-parse", "HEAD"])
    return normalize_commit_sha(head) if ok else None


def local_git_state(folder_path: Path, scope_path: Optional[Path] = None) -> tuple[Optional[str], bool]:
    """Return (HEAD sha, dirty) for a local Git worktree.

    "Dirty" means tracked content in scope differs from HEAD. Untracked files
    are deliberately ignored (``--untracked-files=no``): they do not affect
    whether the *indexed corpus* is reproducible at HEAD, and that corpus is
    independently proven fully tracked by ``local_git_paths_tracked``. The
    explicit ``no`` mode also overrides any ``status.showUntrackedFiles`` repo
    config so the result is deterministic.

    Non-Git folders return ``(None, False)``. Once a worktree is detected,
    failure to prove clean status is treated as dirty so callers never emit an
    immutable repo@sha handle for an unknown state.
    """
    folder_path = folder_path.resolve()
    git_root = _git_root(folder_path)
    if git_root is None:
        return None, False

    head_sha = local_git_head(folder_path)
    if not head_sha:
        return None, False

    status_args = ["status", "--porcelain", "--untracked-files=no"]
    if scope_path is not None:
        try:
            rel = scope_path.resolve().relative_to(git_root).as_posix()
        except ValueError:
            rel = scope_path.resolve().as_posix()
        status_args.extend(["--", rel or "."])

    ok, status = _git(git_root, status_args)
    if not ok:
        return head_sha, True
    return head_sha, bool(status)


def _indexed_git_paths(folder_path: Path, indexed_paths: Iterable[str]) -> tuple[Optional[Path], Optional[set[str]]]:
    folder_path = folder_path.resolve()
    git_root = _git_root(folder_path)
    if git_root is None:
        return None, None

    wanted: set[str] = set()
    for rel_path in indexed_paths:
        if not isinstance(rel_path, str) or not rel_path:
            return git_root, None
        try:
            git_rel = (folder_path / rel_path).resolve().relative_to(git_root).as_posix()
        except ValueError:
            return git_root, None
        wanted.add(git_rel)
    return git_root, wanted


def local_git_paths_dirty(folder_path: Path, indexed_paths: Iterable[str]) -> bool:
    """Return True when tracked indexed paths differ from HEAD."""
    git_root, wanted = _indexed_git_paths(folder_path, indexed_paths)
    if git_root is None:
        return False
    if wanted is None:
        return True
    if not wanted:
        return False

    ordered = sorted(wanted)
    chunk_size = 200
    for i in range(0, len(ordered), chunk_size):
        ok, status = _git(
            git_root,
            ["status", "--porcelain", "--untracked-files=no", "--", *ordered[i:i + chunk_size]],
        )
        if not ok or status:
            return True
    return False


def local_git_paths_tracked(folder_path: Path, indexed_paths: Iterable[str]) -> bool:
    """Return True when every indexed path is tracked by Git."""
    git_root, wanted = _indexed_git_paths(folder_path, indexed_paths)
    if git_root is None or wanted is None:
        return False

    if not wanted:
        return True

    tracked: set[str] = set()
    ordered = sorted(wanted)
    chunk_size = 200
    for i in range(0, len(ordered), chunk_size):
        ok, output = _git_bytes(git_root, ["ls-files", "-z", "--", *ordered[i:i + chunk_size]])
        if not ok:
            return False
        tracked.update(
            p for p in output.decode("utf-8", errors="surrogateescape").split("\0") if p
        )
    return wanted <= tracked


def stable_local_git_state(
    before: tuple[Optional[str], bool],
    after: tuple[Optional[str], bool],
) -> tuple[Optional[str], bool]:
    """Combine pre/post read Git state; SHA movement makes the index dirty."""
    before_sha, before_dirty = before
    after_sha, after_dirty = after
    moved = before_sha != after_sha and bool(before_sha or after_sha)
    return after_sha or before_sha, bool(before_dirty or after_dirty or moved)


def is_linked_worktree(path: Path) -> bool:
    """True when ``path`` is the top of a linked ``git worktree`` checkout.

    A linked worktree marks its root with a ``.git`` FILE whose ``gitdir:``
    target lives under the main checkout's ``.git/worktrees/<name>``.
    Submodules also use a ``.git`` file but point at ``.git/modules/<name>``,
    so they deliberately do NOT match: a submodule's content IS indexed into
    the parent, and a worktree's is not.

    Detected by the ``.git`` marker, never by a ``.worktrees/`` path
    convention. No tool enforces that name, so a path test would both miss
    worktrees created elsewhere and refuse an ordinary directory that happens
    to be called that.

    No subprocess: this runs inside the per-file resolver on the re-index
    hook's path, which fires on every markdown write.

    Ported from jcodemunch's ``storage/git_root.py``, which carries the same
    rule for the code index.
    """
    dotgit = path / ".git"
    try:
        if not dotgit.is_file():
            return False
        text = dotgit.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return False
    if not text.startswith("gitdir:"):
        return False
    target = Path(text[len("gitdir:"):].strip())
    if not target.is_absolute():
        target = path / target
    return target.parent.name == "worktrees"


def linked_worktree_between(source_root: Path, path: Path) -> Optional[Path]:
    """Root of a linked worktree between an indexed root and a path, or None.

    ⚠⚠ Containment does NOT imply the worktree rule, and this is the whole
    reason the check exists as a separate question. ``<repo>/.worktrees/<x>/f.md``
    is genuinely inside ``<repo>``, so an index rooted at the repo contains it
    on every test an ownership resolver would ordinarily make. Only a
    worktree test that follows the containment match refuses it.

    ``source_root`` itself is never tested, so an index rooted AT the worktree
    owns its own files: worktree content belongs to the worktree's index or to
    none, and it enters the parent's when it merges.
    """
    try:
        if not path.is_relative_to(source_root):
            return None
        rel = path.relative_to(source_root)
    except (OSError, ValueError):
        return None
    ancestor = source_root
    for part in rel.parts[:-1]:  # exclude the filename itself
        ancestor = ancestor / part
        if is_linked_worktree(ancestor):
            return ancestor
    return None
