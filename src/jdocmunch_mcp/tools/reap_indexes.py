"""Remove indexes whose source corpus is provably gone.

Nothing else ever removes an index. A test run that indexes a fixture project
under the OS temp directory leaves its index behind forever, and those dead
handles are not free: ``list_repos`` reads every index in the directory, they
are the population a wrong ``resolve_repo`` answer can be drawn from, and each
one keeps its raw-content mirror on disk.

⚠⚠ THE PREDICATE IS THE WHOLE DESIGN, and "the root does not exist" is NOT it.

``Path.exists()`` answers False both for a path that is provably absent and for
one it could not read. On a machine where the corpora live under OneDrive, a
placeholder or offline file raises rather than answering, so a reaper built on
``exists()`` would mass-delete live indexes the first time the sync client was
mid-flight. Every absence test here goes through ``os.stat`` and treats ONLY
``FileNotFoundError`` / ``NotADirectoryError`` as proof; any other ``OSError``
means "cannot prove absent", which is not a reason to delete anything.

The second narrowing is the same instinct applied to scope: a provably-absent
root is reaped only when it sits under the OS temp directory, which no sync
client manages and which is where every dead index on the machine this was
written for came from. A root that is provably absent somewhere else is
REPORTED and not reaped, because "gone" and "gone and never coming back" are
different claims and only the second justifies a delete.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from ..storage import DocStore

# Reap reasons, so a caller can tell WHY each index went.
REASON_TEMP_ROOT_GONE = "temp_root_gone"

# Report-only findings: a dead index this pass deliberately does not delete.
FINDING_ROOT_GONE_OUTSIDE_TEMP = "root_gone_outside_temp"
FINDING_ROOT_UNREADABLE = "root_unreadable"
FINDING_NO_SOURCE_ROOT = "no_source_root"


def _absence(path: Path) -> str:
    """``"present"``, ``"absent"`` (proven), or ``"unknown"`` (could not tell).

    The three-way answer is the point. Collapsing ``unknown`` into ``absent`` is
    what turns a reaper into a data-loss bug on any synced filesystem.
    """
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    except OSError:
        return "unknown"
    return "present"


def _under_temp(path: Path) -> bool:
    """True when ``path`` is inside the OS temp directory.

    Compared through ``realpath`` on both sides: on Windows the temp directory
    is routinely reached by an 8.3 short name (``…\\AppData\\Local\\Temp`` vs
    ``…\\APPDA~1\\Local\\Temp``), and a plain prefix test misses that and
    silently declines to reap.
    """
    try:
        temp = Path(os.path.realpath(tempfile.gettempdir())).resolve()
        # The stored root no longer exists, so realpath cannot canonicalise it;
        # normalise lexically instead, which is all that is available.
        candidate = Path(os.path.normpath(os.path.abspath(str(path))))
    except (OSError, ValueError):
        return False
    try:
        return candidate.is_relative_to(temp)
    except (OSError, ValueError):
        return False


def classify(source_root: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(reap_reason, finding)``; at most one is set.

    A ``reap_reason`` means delete it. A ``finding`` means something is worth
    reporting and is NOT to be deleted by this pass.
    """
    if not source_root:
        # An index with no recorded root cannot be judged, and an unjudgeable
        # index is not a dead one.
        return None, FINDING_NO_SOURCE_ROOT
    root = Path(source_root).expanduser()
    state = _absence(root)
    if state == "present":
        return None, None
    if state == "unknown":
        return None, FINDING_ROOT_UNREADABLE
    if _under_temp(root):
        return REASON_TEMP_ROOT_GONE, None
    return None, FINDING_ROOT_GONE_OUTSIDE_TEMP


def reap_indexes(apply: bool = False, storage_path: Optional[str] = None) -> dict:
    """Find (and with ``apply``, delete) indexes whose corpus is provably gone.

    ⚠ Dry run by DEFAULT. A pass that deletes on its first invocation gives the
    operator no way to read the predicate's answer before trusting it, and this
    predicate is the entire risk.

    Deletion goes through ``DocStore.delete_index``, which already removes the
    summary sidecar, the embeddings sidecar, the glossary/related/boilerplate/
    duplicates sidecars and the raw-content directory. A reaper that removed
    only the monolith would leave the mirror it was written to collect.
    """
    store = DocStore(base_path=storage_path)
    reaped, findings, failed = [], [], []

    for row in store.list_repos():
        repo = row.get("repo", "")
        if "/" not in repo:
            continue
        owner, _, name = repo.partition("/")
        source_root = row.get("source_root") or ""
        reason, finding = classify(source_root)
        if finding:
            findings.append({"repo": repo, "source_root": source_root,
                             "finding": finding})
            continue
        if not reason:
            continue
        entry = {"repo": repo, "source_root": source_root, "reason": reason}
        if apply:
            try:
                # Refuses fast rather than waiting (jdoc#95 QA-25). A held lock
                # means something is actively working with this index, which is
                # evidence AGAINST it being dead; waiting for the handle would
                # make a destructive pass block on the one case where its own
                # premise is most in doubt. The index stays, the next pass sees
                # it again, and nothing is lost by declining.
                if not store.delete_index(owner, name, lock_wait=False):
                    failed.append({**entry, "error": "delete_index returned False"})
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad index must not stop the pass
                failed.append({**entry, "error": f"{type(exc).__name__}: {exc}"})
                continue
        reaped.append(entry)

    return {
        "success": True,
        "applied": apply,
        "reaped": reaped,
        "reaped_count": len(reaped),
        "findings": findings,
        "failed": failed,
    }
