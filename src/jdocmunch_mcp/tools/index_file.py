"""Re-index a single file within an existing index."""

import time
from pathlib import Path
from typing import Optional

from ..parser import parse_file, preprocess_content, ALL_EXTENSIONS
from ..parser.office import (
    OFFICE_EXTENSIONS,
    office_available,
    convert_office,
    office_cache_dir,
)
from ..storage import DocStore
from ..storage.doc_store import normalize_commit_sha
from ..summarizer import summarize_sections
from ..embeddings import embed_sections
from ._embedding_coverage import attach_embedding_coverage as _attach_embedding_coverage
from ._git import (
    linked_worktree_between,
    local_git_head,
    local_git_paths_dirty,
    local_git_paths_tracked,
)


def _find_owning_index(
    file_path: Path,
    store: DocStore,
) -> Optional[tuple[str, str, str, Path]]:
    """Find which index owns a given file path, by CONTAINMENT.

    Two rules, and the second is not implied by the first.

    RULE 1, ownership by containment. The owner is the index whose stored
    ``source_root`` contains the file, deepest root winning when several do.
    A file under no index's root is refused rather than adopted, and the
    stored path is always relative to the winning root, so one file can never
    acquire two spellings.

    RULE 2, worktree refusal. A file inside a linked git worktree is owned
    only by an index rooted at that worktree, never by the parent repository's
    index, whose ``source_root`` may well contain it. Worktree content enters
    the parent's index when it merges.

    ⚠⚠ This used to resolve by FOLDER NAME: it walked up from the file loading
    ``local/<ancestor folder name>`` and, when the computed relative path was
    unknown, took ownership anyway on the sole condition that the index had any
    doc_paths at all. ``source_root`` was never consulted. So an index named
    ``myrepo`` rooted at ``myrepo/docs`` claimed every doc file anywhere in
    ``myrepo/``, linked worktrees included, and stored them repo-relative
    while the walk stored root-relative: an edit to ``docs/guide.md`` was
    stored as a second entry ``docs/guide.md`` beside ``guide.md``, and the
    fresh content went to the entry the walk never refreshes.

    ⚠ ``doc_paths`` is deliberately not consulted any more. Containment alone
    decides ownership, so a genuinely new file under the root is admitted
    because nothing asks whether it is already known. That was the old
    fallback's legitimate purpose; its defect was running before containment
    had ever been tested.

    Returns (owner, name, rel_path, source_root) or None.
    """
    file_path = file_path.resolve()

    # `list_repos` rows carry `source_root` and are read from the tiny summary
    # sidecars, so this is a metadata scan rather than one index load per
    # candidate. Do NOT replace this with a `_resolve_named_index` call per
    # candidate, which loads every index, on a path that fires on every
    # markdown write.
    best: Optional[tuple[str, str, str, Path]] = None
    best_depth = -1
    for row in store.list_repos():
        source_root = row.get("source_root") or ""
        if not source_root:
            continue  # legacy index with no recorded root: not resolvable here
        try:
            root = Path(source_root).expanduser().resolve()
            if not file_path.is_relative_to(root):
                continue
        except (OSError, ValueError):
            continue
        if linked_worktree_between(root, file_path) is not None:
            continue  # RULE 2: the parent's index does not own worktree content
        depth = len(root.parts)
        if depth > best_depth:
            owner, _, bare = str(row.get("repo", "")).partition("/")
            if not bare:
                continue
            best = (
                owner or "local",
                bare,
                file_path.relative_to(root).as_posix(),
                root,
            )
            best_depth = depth

    return best


def _resolve_named_index(
    file_path: Path,
    store: DocStore,
    name: str,
) -> Optional[tuple[str, str, str, Path]]:
    """Resolve an explicitly named index and the file's path within it (#38).

    Skips folder-name inference entirely: loads the index by its stored name
    and computes the file's relative path from the index's ``source_root``.
    This is the only path that works for a custom ``--name`` or a corpus whose
    root folder name is not path-safe (e.g. contains spaces).

    ``name`` may be ``owner/name`` or a bare name (owner defaults to ``local``).
    Returns (owner, name, rel_path, source_root) or None.
    """
    owner, _, bare = name.rpartition("/")
    owner = owner or "local"
    try:
        index = store.load_index(owner, bare)
    except ValueError:
        return None
    if index is None:
        return None
    root = getattr(index, "source_root", "")
    if not root:
        return None
    source_root = Path(root).expanduser().resolve()
    try:
        rel_path = file_path.relative_to(source_root).as_posix()
    except ValueError:
        return None
    return (owner, bare, rel_path, source_root)


def index_file(
    file_path: str,
    storage_path: Optional[str] = None,
    use_ai_summaries: bool = True,
    name: Optional[str] = None,
) -> dict:
    """Re-index a single file within an existing index.

    Finds which index owns the file (or uses ``name`` when given), re-parses
    it, and updates the index in place using incremental_save.

    Returns dict with results.
    """
    t0 = time.perf_counter()
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}", "exit_code": 2}
    if not path.is_file():
        return {"success": False, "error": f"Not a file: {file_path}", "exit_code": 2}

    ext = path.suffix.lower()
    if ext in OFFICE_EXTENSIONS:
        if not office_available():
            return {"success": False, "error": (
                f"Office document support requires the optional markitdown "
                f"extra (pip install jdocmunch-mcp[office]): {file_path}"
            ), "exit_code": 2}
    elif ext not in ALL_EXTENSIONS:
        return {"success": False, "error": f"Not a doc file ({ext}): {file_path}", "exit_code": 2}

    store = DocStore(base_path=storage_path)
    if name:
        match = _resolve_named_index(path, store, name)
        if match is None:
            return {"success": False, "error": (
                f"No index named {name!r} with a source_root containing "
                f"{file_path}"
            ), "exit_code": 1}
    else:
        match = _find_owning_index(path, store)
        if match is None:
            return {"success": False, "error": f"File not in any index: {file_path}", "exit_code": 1}

    owner, name, rel_path, detected_root = match
    repo_id = f"{owner}/{name}"

    # Read and parse the file
    try:
        if ext in OFFICE_EXTENSIONS:
            # Binary office document: convert to Markdown locally
            # (parser/office.py), matching index_local's read leg.
            content = convert_office(path, cache_dir=office_cache_dir(store.base_path))
        else:
            # newline="" preserves CRLF/CR so offsets/hashes address the real
            # on-disk bytes (#52); Path.read_text lacks newline before 3.13.
            with open(path, encoding="utf-8", errors="replace", newline="") as fh:
                content = fh.read()
        content = preprocess_content(content, rel_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to read: {e}", "exit_code": 2}

    try:
        new_sections = parse_file(content, rel_path, repo_id)
    except Exception as e:
        return {"success": False, "error": f"Failed to parse: {e}", "exit_code": 2}

    if not new_sections:
        new_sections = []

    new_sections = summarize_sections(new_sections, use_ai=use_ai_summaries)

    # Determine if this is a new or changed file
    index = store.load_index(owner, name)
    is_new = rel_path not in (index.file_hashes if index else {})
    source_root = detected_root
    if index is not None and getattr(index, "source_root", ""):
        candidate = Path(index.source_root).expanduser()
        if candidate.exists():
            source_root = candidate.resolve()
    previous_sha = normalize_commit_sha(index.head_sha if index else None)
    previous_certified = bool(index is not None and index.sha_certified)
    indexed_paths = set(index.doc_paths if index else [])
    indexed_paths.add(rel_path)
    paths_tracked = local_git_paths_tracked(source_root, indexed_paths)
    paths_dirty = local_git_paths_dirty(source_root, indexed_paths)
    current_sha = local_git_head(source_root)
    head_moved = bool(previous_sha and current_sha and previous_sha != current_sha)
    legacy_uncertified = bool(current_sha and not previous_sha)
    lost_git_context = bool(previous_sha and current_sha is None)
    head_sha = current_sha or previous_sha
    source_dirty = bool(
        paths_dirty
        or head_moved
        or legacy_uncertified
        or lost_git_context
        or (bool(head_sha) and not paths_tracked)
        or (index is not None and index.source_dirty)
    )
    sha_certified = bool(
        previous_certified
        and current_sha
        and previous_sha == current_sha
        and not source_dirty
        and paths_tracked
    )

    # Preserve embedding parity: if the existing index has embeddings, embed new sections too.
    if index is not None and index._has_embeddings():
        # jdoc#107: owner/name were missing, so the cache was disabled and the
        # vectors were persisted only by the save safety net — which used to
        # bail whenever a sidecar already existed, i.e. always. Every vector
        # this path produced was silently discarded, and this is the
        # PostToolUse auto-reindex path, so it ran on every doc edit. Merge,
        # never prune: one file is never authoritative for the corpus.
        new_sections = embed_sections(
            new_sections,
            owner=owner, name=name, storage_path=storage_path,
        )

    # Use incremental_save to update just this file
    updated = store.incremental_save(
        owner=owner,
        name=name,
        changed_files=[] if is_new else [rel_path],
        new_files=[rel_path] if is_new else [],
        deleted_files=[],
        new_sections=new_sections,
        raw_files={rel_path: content},
        doc_types={ext: 1},
        head_sha=head_sha,
        source_dirty=source_dirty,
        sha_certified=sha_certified,
        source_root=str(source_root),
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    result = {
        "success": True,
        "repo": repo_id,
        "file": rel_path,
        "is_new": is_new,
        "sections": len(new_sections),
        "total_sections": len(updated.sections) if updated else 0,
        "exit_code": 0,
        "_meta": {"latency_ms": latency_ms},
    }
    # jdoc#107: this path fires from the PostToolUse hook on every doc edit,
    # so it is where a coverage collapse would be noticed first.
    _attach_embedding_coverage(
        result, storage_path=storage_path, owner=owner, name=name, index=updated,
    )
    if updated and updated.head_sha:
        result["head_sha"] = updated.head_sha
    if updated:
        result["source_dirty"] = bool(updated.source_dirty)
        result["sha_certified"] = bool(updated.sha_certified)
    if updated and updated.repo_at_sha:
        result["repo_at_sha"] = updated.repo_at_sha
    return result


def index_file_cli(file_path: str, name: Optional[str] = None) -> dict:
    """CLI entry point for index-file subcommand."""
    return index_file(file_path, name=name)
