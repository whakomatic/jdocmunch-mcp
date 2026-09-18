"""Embedding coverage disclosure for the indexing tools (jdoc#107).

#107's reporter lost a 5,316-vector sidecar three times and only found out by
writing their own external check — comparing sidecar rows against
`section_count` after every reindex and rebuilding below 50%. The index tools
emitted nothing: exit 0, no warning, and `search_sections` kept answering with
a semantic channel that ranked almost nothing.

⚠⚠ The write bug is fixed at its source; this exists because the SILENCE was
the other half of the defect. A coverage collapse from any future cause — a
provider timing out mid-batch, a half-written sidecar, an identity rotation —
is now visible in the response instead of needing a user to instrument us.

Counted off the sidecar's KEYS, never its vectors: a 26k-section corpus costs
no memory here.
"""

from __future__ import annotations

from typing import Optional

# Below this, the semantic channel is degraded enough that the caller should
# be told in words, not just handed a ratio. Matches the threshold #107's
# reporter chose independently for their external rebuild trigger.
COVERAGE_WARN_BELOW = 0.5


def saved_index_has_embeddings(store, owner: str, name: str) -> bool:
    """Whether ``search_sections`` will find vectors for this index.

    The indexing tools report ``semantic_search`` from this, not from the
    embedding configuration: a provider can be configured and still embed
    nothing. It reads the index the way ``search_sections`` does
    (``load_index`` then ``_has_embeddings``), because the object an
    incremental save returns has its vectors stripped and no sidecar pointer,
    so it reads as empty on a deletion-only refresh.
    """
    saved = store.load_index(owner, name)
    return bool(saved and saved._has_embeddings())


def attach_embedding_coverage(
    result: dict,
    *,
    storage_path: Optional[str],
    owner: str,
    name: str,
    index,
    warnings: Optional[list] = None,
) -> dict:
    """Add ``embedded_sections`` / ``embedding_coverage`` to a tool response.

    No-op when the index has no sections or no sidecar exists (a lexical-only
    index must not grow a 0.0 coverage field that reads like a regression).
    """
    if index is None:
        return result
    try:
        from ..embeddings import cache as _emb_cache

        stored = _emb_cache.stored_hashes(storage_path, owner, name)
        if not stored:
            return result
        sections = getattr(index, "sections", None) or []
        total = len(sections)
        if not total:
            return result
        covered = sum(1 for s in sections if s.get("content_hash") in stored)
        result["embedded_sections"] = covered
        result["embedding_coverage"] = round(covered / total, 4)
        if covered / total < COVERAGE_WARN_BELOW and warnings is not None:
            warnings.append(
                f"Embedding coverage is {covered}/{total} sections "
                f"({covered / total:.1%}). Semantic search will rank most of "
                f"this corpus at zero. Re-index with incremental=false to "
                f"rebuild the embeddings sidecar."
            )
    except Exception:
        pass  # disclosure is best-effort; never fail an index over it
    return result
