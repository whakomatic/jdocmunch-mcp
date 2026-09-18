"""jdoc#109 — rotating the embedding model must not leave an unqueryable index.

Reported by @pnm-jgb: change ``JDOCMUNCH_ST_MODEL``, re-index a corpus whose
files have not changed, and the run reports ``success: true`` while the sidecar
keeps its old 384-dim vectors. Every subsequent search then died with a raw
numpy error — ``size 768 is different from 384`` — and the only recovery was
``delete-index`` plus a full rebuild.

⚠ The report locates the bug at the ``embed_sections`` call on the incremental
path (``index_local.py`` ~2160), reasoning that ``entries`` ends up empty and
the ``if entries:`` guard skips the write. That mechanism is real, but it is
not the one the repro hits: with zero changed files ``index_local`` returns
from the **"No changes detected"** branch further up and never calls
``embed_sections`` at all. Fixing only the guard would have left the reported
repro broken. The detection therefore sits BEFORE the incremental branch.

Three distinct failures are covered here, because surviving the rotation on one
path is not the same as surviving it:

1. **Indexing** must notice the rotation and re-embed the whole corpus.
2. **Querying** must never raise on width-mismatched vectors, however they got
   there — including a sidecar holding two widths at once.
3. ⚠⚠ The pure-Python cosine fallback (no numpy) must not score mismatched
   vectors at all. ``cosine_similarity`` zips the two lists, so a 768-dim query
   against a 384-dim vector silently truncates and returns a *plausible*
   number. That is worse than the crash: the numpy path fails loudly, this one
   returns confident garbage and nothing anywhere records it.
"""

import pytest

from jdocmunch_mcp.embeddings import cache as emb_cache
from jdocmunch_mcp.embeddings.provider import cosine_similarity
from jdocmunch_mcp.storage.doc_store import DocIndex


# ---------------------------------------------------------------------------
# 1. cache.identity — telling "no sidecar" apart from "different model"
# ---------------------------------------------------------------------------

def _write_sidecar(tmp_path, provider, model, dim, n=3):
    emb_cache.write(
        str(tmp_path), "local", "corpus",
        provider=provider, model=model, dim=dim,
        entries=[(f"h{i}", [0.5] * dim) for i in range(n)],
    )


def test_identity_is_none_when_no_sidecar_exists(tmp_path):
    assert emb_cache.identity(str(tmp_path), "local", "corpus") is None


def test_identity_reports_the_stored_header(tmp_path):
    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384)
    assert emb_cache.identity(str(tmp_path), "local", "corpus") == {
        "provider": "sentence-transformers",
        "model": "all-MiniLM-L6-v2",
        "dim": 384,
        "embed_chars": 1000,   # jdoc#111 joined the identity
    }


def test_absent_and_rotated_are_distinguishable(tmp_path):
    """⚠⚠ The prerequisite for every other fix here.

    ``load()`` returns ``{}`` for both cases, which is precisely why nothing
    could act on a rotation: the incremental path could not tell a first index
    from a model change.
    """
    absent = emb_cache.identity(str(tmp_path), "local", "corpus")
    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384)
    rotated = emb_cache.identity(str(tmp_path), "local", "corpus")

    assert emb_cache.load(
        str(tmp_path), "local", "corpus",
        provider="sentence-transformers", model="BAAI/bge-base-en-v1.5", dim=768,
    ) == {}
    assert absent is None and rotated is not None


@pytest.mark.parametrize("provider,model,dim,expected", [
    ("sentence-transformers", "all-MiniLM-L6-v2", 384, True),
    ("sentence-transformers", "BAAI/bge-base-en-v1.5", 768, False),
    ("sentence-transformers", "all-MiniLM-L6-v2", 768, False),
    ("openai", "all-MiniLM-L6-v2", 384, False),
    ("sentence-transformers", "all-MiniLM-L6-v2", None, True),   # dim-tolerant, as load() is
])
def test_identity_matches_mirrors_the_load_header_check(tmp_path, provider, model, dim, expected):
    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384)
    stored = emb_cache.identity(str(tmp_path), "local", "corpus")
    assert emb_cache.identity_matches(stored, provider, model, dim) is expected


def test_identity_matches_is_false_for_a_missing_sidecar():
    assert emb_cache.identity_matches(None, "p", "m", 8) is False


# ---------------------------------------------------------------------------
# 2. Querying survives a width mismatch
# ---------------------------------------------------------------------------

def _index(dims):
    """A DocIndex whose sections carry embeddings of the given widths."""
    sections = [
        {"id": f"s{i}", "doc_path": "d.md", "title": f"T{i}", "content": "body",
         "embedding": [0.5] * d}
        for i, d in enumerate(dims)
    ]
    idx = DocIndex(
        owner="local", name="corpus", repo="local/corpus",
        indexed_at="2026-08-09T00:00:00", doc_paths=["d.md"], doc_types={".md": 1},
        sections=sections,
    )
    idx._rehydrate_embeddings = lambda: None   # vectors are already inline
    return idx


def test_query_of_a_different_width_does_not_raise(monkeypatch):
    """The reported crash: 768-dim query against 384-dim storage."""
    idx = _index([384] * 5)
    scored = idx._semantic_scored([0.1] * 768, None, None)
    assert scored == []


def test_matching_width_still_scores_normally():
    idx = _index([8] * 4)
    scored = idx._semantic_scored([1.0] * 8, None, None)
    assert len(scored) == 4
    assert all(0.99 <= s <= 1.01 for s, _ in scored)


def test_mixed_width_sidecar_scores_only_the_matching_rows():
    """⚠ A rotation that touched SOME files leaves both widths on disk.

    The old single-matrix build called ``np.asarray`` on ragged rows, which
    raises before a single query is scored — a second crash site the report
    does not mention.
    """
    idx = _index([8, 8, 16, 16, 16])
    assert len(idx._semantic_scored([1.0] * 8, None, None)) == 2
    assert len(idx._semantic_scored([1.0] * 16, None, None)) == 3


def test_embedding_dims_reports_the_spread():
    assert _index([8, 8, 16]).embedding_dims() == {8: 2, 16: 1}


def test_mismatch_is_recorded_for_disclosure():
    """Degrading to lexical silently would trade a loud failure for a quiet one."""
    idx = _index([384] * 3)
    idx._semantic_scored([0.1] * 768, None, None)
    assert idx._embedding_width_mismatch == {
        "query_dim": 768,
        "stored_dims": {384: 3},
    }


def test_no_mismatch_is_recorded_when_widths_agree():
    idx = _index([8] * 3)
    idx._semantic_scored([1.0] * 8, None, None)
    assert getattr(idx, "_embedding_width_mismatch", None) is None


def test_pure_python_fallback_also_skips_mismatched_vectors(monkeypatch):
    """⚠⚠ Without numpy there is no exception to catch — only wrong answers.

    ``cosine_similarity([1.0]*768, [0.5]*384)`` returns 0.707 — a perfectly
    ordinary-looking mid-range similarity — because ``zip`` stops at the
    shorter sequence and the norms are still taken over the full vectors. A
    numpy-free install would have seen degraded ranking and no error at all.
    """
    assert cosine_similarity([1.0] * 768, [0.5] * 384) == pytest.approx(0.7071, abs=1e-4)

    idx = _index([384] * 3)
    monkeypatch.setattr(DocIndex, "_semantic_matrices", lambda self: None)
    assert idx._semantic_scored([1.0] * 768, None, None) == []


def test_back_compat_matrix_shim_still_returns_np_mat_rows():
    """jdoc#63's entry point keeps its shape; scoring just no longer uses it."""
    idx = _index([8] * 3)
    built = idx._ensure_semantic_matrix()
    assert built is not None
    np_mod, mat, rows = built
    assert mat.shape == (3, 8) and len(rows) == 3
    assert idx._ensure_semantic_matrix() is built   # still cached (v1.83.0)


def test_shim_picks_the_most_covered_width():
    idx = _index([8, 16, 16, 16])
    _, mat, rows = idx._ensure_semantic_matrix()
    assert mat.shape == (3, 16) and len(rows) == 3


# ---------------------------------------------------------------------------
# 3. Indexing detects the rotation and escalates
# ---------------------------------------------------------------------------

class _FakeProvider:
    def __init__(self, dim):
        self.dim = dim
        self.embedded = 0

    def embed_texts(self, texts, task_type=None):
        self.embedded += len(texts)
        return [[0.25] * self.dim for _ in texts]


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        (src / f"doc{i}.md").write_text(
            f"# Doc {i}\n\n## Network resilience {i}\n\n"
            "When the uplink drops, queued work is retained.\n",
            encoding="utf-8",
        )
    return src


def _sidecar_header(store, owner="local", name="rot"):
    """The sidecar's identity line, read straight off disk."""
    import json as _json
    path = emb_cache._cache_path(str(store), owner, name)
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            return _json.loads(raw)
    raise AssertionError("empty sidecar")


def _use_model(monkeypatch, model, dim):
    """Point every identity surface at one (model, dim) pair."""
    from jdocmunch_mcp.embeddings import provider as prov

    p = _FakeProvider(dim)
    monkeypatch.setattr(prov, "_get_provider", lambda: p)
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity", lambda n: (model, dim))
    return p


def _run_index(monkeypatch, corpus, store, model, dim, **kwargs):
    from jdocmunch_mcp.tools.index_local import index_local
    prov = _use_model(monkeypatch, model, dim)
    result = index_local(
        path=str(corpus), name="rot", storage_path=str(store),
        use_ai_summaries=False, use_embeddings=True, **kwargs
    )
    return result, prov


def test_rotation_with_no_file_changes_re_embeds(monkeypatch, corpus, tmp_path):
    """⚠⚠ The reported repro, end to end.

    Not one file is touched between the two runs, so the old code returned
    "No changes detected" and left 384-dim vectors under a 768-dim encoder.
    """
    store = tmp_path / "idx"
    first, _ = _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    assert first["success"]
    # ⚠ Read the header off disk rather than through cache.identity(): this
    # test must fail on the BEHAVIOR when run against a build that predates
    # the fix, not on a missing helper.
    assert _sidecar_header(store)["dim"] == 384

    second, prov = _run_index(monkeypatch, corpus, store, "BAAI/bge-base-en-v1.5", 768)

    assert second["success"]
    assert prov.embedded > 0, "rotation re-embedded nothing — this is jdoc#109"
    header = _sidecar_header(store)
    assert header["model"] == "BAAI/bge-base-en-v1.5"
    assert header["dim"] == 768


def test_rotation_is_disclosed_not_silent(monkeypatch, corpus, tmp_path):
    """A full corpus re-embed is a cost — on a paid provider, a bill.

    The caller asked for an incremental refresh; it is owed the fact that it
    got something else, and why.
    """
    store = tmp_path / "idx"
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    second, _ = _run_index(monkeypatch, corpus, store, "BAAI/bge-base-en-v1.5", 768)

    rot = second.get("embedding_rotation")
    assert rot, "escalation happened with nothing in the payload to show it"
    assert rot["from"]["model"] == "all-MiniLM-L6-v2"
    assert rot["to"]["model"] == "BAAI/bge-base-en-v1.5"
    assert rot["to"]["dim"] == 768
    assert rot["action"] == "full_re_embed"


def test_no_rotation_still_takes_the_cheap_path(monkeypatch, corpus, tmp_path):
    """⚠ The fix must not turn every refresh into a full re-embed."""
    store = tmp_path / "idx"
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    second, prov = _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)

    assert "embedding_rotation" not in second
    assert second.get("incremental") is True
    assert prov.embedded == 0, "unchanged corpus was re-embedded anyway"


def test_unchanged_run_reports_embedding_status(monkeypatch, corpus, tmp_path):
    """jdoc#109 fix 3: absence of a field is not a status.

    The full-rebuild payload always carried `semantic_search`; the no-change
    payload carried nothing, so a caller could not distinguish "embeddings are
    healthy" from "embeddings were never looked at".
    """
    store = tmp_path / "idx"
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    second, _ = _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    assert second["message"] == "No changes detected"
    assert second["semantic_search"] is True


# ---------------------------------------------------------------------------
# 4. search_sections tells the user, instead of handing back a matmul error
# ---------------------------------------------------------------------------

def test_search_survives_rotation_and_names_the_fix(monkeypatch, corpus, tmp_path):
    """The reporter's step 3, which used to return a raw numpy error string.

    Two things have to be true at once: the query must succeed on the lexical
    lane, and the payload must SAY the semantic lane was dropped. Either alone
    is a different bug — a bare error, or a silent quality regression.
    """
    from jdocmunch_mcp.embeddings import provider as prov
    from jdocmunch_mcp.tools.search_sections import search_sections

    store = tmp_path / "idx"
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)

    # Rotate the QUERY encoder only — the on-disk vectors stay 384-dim, exactly
    # the state a user reaches by rotating the model without re-indexing.
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity", lambda n: ("BAAI/bge-base-en-v1.5", 768))
    monkeypatch.setattr(prov, "_get_provider", lambda: _FakeProvider(768))
    prov._query_cache().clear()

    result = search_sections(
        repo="local/rot", query="what happens when the uplink drops",
        storage_path=str(store),
    )

    assert "error" not in result, f"search still fails on rotated vectors: {result.get('error')}"
    stale = result["_meta"].get("embedding_stale")
    assert stale, "semantic lane was dropped with nothing in _meta to say so"
    assert stale["query_dim"] == 768
    assert list(stale["stored_dims"]) == [384]
    assert stale["stored_dims"][384] > 0
    assert stale["semantic_disabled"] is True
    assert "--rebuild" in stale["fix"]


def test_healthy_index_carries_no_stale_marker(monkeypatch, corpus, tmp_path):
    """⚠ A warning that fires on healthy indexes is noise, and gets ignored."""
    from jdocmunch_mcp.tools.search_sections import search_sections

    store = tmp_path / "idx"
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    result = search_sections(
        repo="local/rot", query="uplink drops", storage_path=str(store),
    )
    assert "embedding_stale" not in result["_meta"]


# ---------------------------------------------------------------------------
# 5. embed_sections purges a stale identity even with nothing to write
# ---------------------------------------------------------------------------

def test_empty_pass_under_a_rotated_identity_purges_the_sidecar(tmp_path, monkeypatch):
    """The mechanism the report DID identify, fixed on its own terms.

    ⚠ The comment in `embed_sections` claimed `cached` being {} on rotation
    made the write "collapse back to a clean rewrite, so rotation still
    purges". It only purges when at least one section arrives. Hand it none
    and the `if entries:` guard skipped the write entirely, leaving the old
    header and the old widths in place.

    index_local no longer reaches here during a rotation, but `index_file` and
    `index_repo` call `embed_sections` too, so it has to be correct alone.
    """
    from jdocmunch_mcp.embeddings import provider as prov

    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384, n=5)

    monkeypatch.setattr(prov, "_get_provider", lambda: _FakeProvider(768))
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity",
                        lambda n: ("BAAI/bge-base-en-v1.5", 768))

    prov.embed_sections([], owner="local", name="corpus", storage_path=str(tmp_path))

    header = emb_cache.identity(str(tmp_path), "local", "corpus")
    assert header["model"] == "BAAI/bge-base-en-v1.5"
    assert header["dim"] == 768
    assert emb_cache.stored_hashes(str(tmp_path), "local", "corpus") == set(), \
        "384-dim vectors survived a purge under a 768-dim identity"


def test_empty_pass_under_a_MATCHING_identity_leaves_the_sidecar_alone(tmp_path, monkeypatch):
    """⚠⚠ The jdoc#107 invariant this must not break.

    A no-op refresh under the SAME model must not rewrite — let alone empty —
    a sidecar that is the authoritative vector store. #107 was exactly that
    data loss (5,316 vectors to 21, exit 0).
    """
    from jdocmunch_mcp.embeddings import provider as prov

    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384, n=5)

    monkeypatch.setattr(prov, "_get_provider", lambda: _FakeProvider(384))
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity", lambda n: ("all-MiniLM-L6-v2", 384))

    prov.embed_sections([], owner="local", name="corpus", storage_path=str(tmp_path))

    assert len(emb_cache.stored_hashes(str(tmp_path), "local", "corpus")) == 5


# ---------------------------------------------------------------------------
# 6. jdoc#111 — the embed char cap is configurable AND part of the identity
# ---------------------------------------------------------------------------

def test_cap_defaults_to_1000(monkeypatch):
    from jdocmunch_mcp.embeddings import provider as prov
    monkeypatch.delenv("JDOCMUNCH_EMBED_CHARS", raising=False)
    assert prov._embed_chars() == 1000


@pytest.mark.parametrize("raw,expected", [
    ("4000", 4000),
    ("1", 1),
    ("", 1000),
    ("   ", 1000),
    ("0", 1000),          # non-positive falls back
    ("-500", 1000),       # not isdigit
    ("banana", 1000),
    ("2.5", 1000),
])
def test_cap_reads_the_env_and_ignores_nonsense(monkeypatch, raw, expected):
    """⚠ A typo'd env var must not fail the whole index — it embeds at 1000."""
    from jdocmunch_mcp.embeddings import provider as prov
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", raw)
    assert prov._embed_chars() == expected


class _Sec:
    def __init__(self, content, title="T", summary="", content_hash="h1"):
        self.title, self.summary, self.content = title, summary, content
        self.content_hash = content_hash
        self.embedding = None


def test_raising_the_cap_lets_more_prose_through(monkeypatch):
    from jdocmunch_mcp.embeddings import provider as prov
    body = "word " * 1000

    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "1000")
    short = prov._section_embed_text(_Sec(body))
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "4000")
    long = prov._section_embed_text(_Sec(body))

    assert len(long) > len(short)
    assert len(long) - len(short) == pytest.approx(3000, abs=5)


def test_cap_salts_the_cache_key(monkeypatch):
    """Patrick's stated minimum: the cap IS a derivation change."""
    from jdocmunch_mcp.embeddings import provider as prov
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "1000")
    at_1000 = prov._embed_cache_key(_Sec("x"))
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "4000")
    at_4000 = prov._embed_cache_key(_Sec("x"))
    assert at_1000 != at_4000


def test_salted_key_still_yields_the_bare_hash(monkeypatch, tmp_path):
    """⚠ `stored_hashes` recovers the content hash with rsplit('#', 1).

    Coverage reporting reads it, so a salt that lands before the last '#'
    would silently report zero coverage on a fully embedded corpus.
    """
    from jdocmunch_mcp.embeddings import provider as prov
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "4000")
    key = prov._embed_cache_key(_Sec("x", content_hash="abc123"))
    emb_cache.write(
        str(tmp_path), "local", "corpus",
        provider="p", model="m", dim=4, entries=[(key, [0.1] * 4)],
    )
    assert emb_cache.stored_hashes(str(tmp_path), "local", "corpus") == {"abc123"}


# --- the migration rule, which is the risky part ---------------------------

def test_a_pre_111_sidecar_reports_the_legacy_default(tmp_path):
    """⚠⚠ Absence of `embed_chars` means 1000, NOT unknown.

    Every sidecar written before this release lacks the field and was built at
    1000. Reading absence as a mismatch would escalate EVERY existing index to
    a full re-embed on its next run — a corpus-wide bill for users who changed
    nothing at all.
    """
    import json as _json
    path = emb_cache._cache_path(str(tmp_path), "local", "corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps({"_header": True, "provider": "sentence-transformers",
                     "model": "all-MiniLM-L6-v2", "dim": 384}) + "\n"
        + _json.dumps({"hash": "h0#pv1", "vector": [0.5] * 384}) + "\n",
        encoding="utf-8",
    )

    stored = emb_cache.identity(str(tmp_path), "local", "corpus")
    assert stored["embed_chars"] == 1000
    assert emb_cache.identity_matches(
        stored, "sentence-transformers", "all-MiniLM-L6-v2", 384, 1000
    ) is True
    assert emb_cache.load(
        str(tmp_path), "local", "corpus",
        provider="sentence-transformers", model="all-MiniLM-L6-v2",
        dim=384, embed_chars=1000,
    ) != {}


def test_a_raised_cap_does_invalidate_a_pre_111_sidecar(tmp_path):
    """The other half: 1000 -> 4000 IS a real change and must not be ignored."""
    import json as _json
    path = emb_cache._cache_path(str(tmp_path), "local", "corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps({"_header": True, "provider": "sentence-transformers",
                     "model": "all-MiniLM-L6-v2", "dim": 384}) + "\n",
        encoding="utf-8",
    )
    stored = emb_cache.identity(str(tmp_path), "local", "corpus")
    assert emb_cache.identity_matches(
        stored, "sentence-transformers", "all-MiniLM-L6-v2", 384, 4000
    ) is False
    assert emb_cache.load(
        str(tmp_path), "local", "corpus",
        provider="sentence-transformers", model="all-MiniLM-L6-v2",
        dim=384, embed_chars=4000,
    ) == {}


def test_cap_change_on_an_unchanged_corpus_re_embeds(monkeypatch, corpus, tmp_path):
    """jdoc#111 meets jdoc#109: same escalation, same disclosure."""
    store = tmp_path / "idx"
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "1000")
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    assert _sidecar_header(store)["embed_chars"] == 1000

    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "4000")
    second, prov = _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)

    assert prov.embedded > 0, "a cap change served the old short-text vectors"
    assert _sidecar_header(store)["embed_chars"] == 4000
    rot = second.get("embedding_rotation")
    assert rot and rot["to"]["embed_chars"] == 4000
    assert rot["from"]["embed_chars"] == 1000


def test_unchanged_cap_still_takes_the_cheap_path(monkeypatch, corpus, tmp_path):
    """⚠ The default must not turn every refresh into a full re-embed."""
    store = tmp_path / "idx"
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "1000")
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    second, prov = _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    assert "embedding_rotation" not in second
    assert prov.embedded == 0


def test_default_cap_leaves_existing_keys_byte_identical(monkeypatch):
    """⚠⚠ The upgrade trap in the report's own patch sketch.

    Salting unconditionally makes the new key `h#pv1-1000` miss against the
    `h#pv1` already on disk, so EVERY user on the default re-embeds their whole
    corpus on upgrade — paying, on a cloud provider, to arrive at identical
    vectors. The default must add no salt at all.
    """
    from jdocmunch_mcp.embeddings import provider as prov
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "1000")
    assert prov._embed_cache_key(_Sec("x", content_hash="abc")) == "abc#pv1"
    monkeypatch.delenv("JDOCMUNCH_EMBED_CHARS", raising=False)
    assert prov._embed_cache_key(_Sec("x", content_hash="abc")) == "abc#pv1"
    monkeypatch.setenv("JDOCMUNCH_EMBED_CHARS", "4000")
    assert prov._embed_cache_key(_Sec("x", content_hash="abc")) == "abc#pv1-4000"


def test_upgrading_an_existing_index_re_embeds_nothing(tmp_path, monkeypatch):
    """End to end: a pre-1.127.0 sidecar is still a full cache hit."""
    import json as _json
    from jdocmunch_mcp.embeddings import provider as prov

    # A sidecar exactly as 1.126.1 wrote it: no embed_chars, unsalted keys.
    path = emb_cache._cache_path(str(tmp_path), "local", "corpus")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_json.dumps({"_header": True, "provider": "fake",
                          "model": "fake-model", "dim": 8})]
    lines += [_json.dumps({"hash": f"h{i}#pv1", "vector": [0.5] * 8})
              for i in range(5)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fake = _FakeProvider(8)
    monkeypatch.delenv("JDOCMUNCH_EMBED_CHARS", raising=False)
    monkeypatch.setattr(prov, "_get_provider", lambda: fake)
    monkeypatch.setattr(prov, "get_provider_name", lambda: "fake")
    monkeypatch.setattr(prov, "_provider_identity", lambda n: ("fake-model", 8))

    sections = [_Sec("body", content_hash=f"h{i}") for i in range(5)]
    prov.embed_sections(sections, owner="local", name="corpus",
                        storage_path=str(tmp_path))

    assert fake.embedded == 0, (
        "upgrading to 1.127.0 re-embedded an unchanged corpus — every user "
        "on the default would pay for identical vectors"
    )
    assert all(s.embedding == [0.5] * 8 for s in sections)


# ---------------------------------------------------------------------------
# 7. A provider outage during a rotation must not destroy the vector store
# ---------------------------------------------------------------------------

class _BrokenProvider:
    def embed_texts(self, texts, task_type=None):
        raise RuntimeError("connection reset by peer")


def test_a_provider_outage_during_rotation_keeps_the_old_vectors(tmp_path, monkeypatch):
    """⚠⚠ The regression the jdoc#109 purge introduced, caught before release.

    Purging on an empty pass is right when the corpus produced no vectors and
    is DATA LOSS when the provider merely threw. Without the `embed_failed`
    guard, a transient outage mid-rotation empties the sidecar, writes the NEW
    header over it, and so convinces the next run that nothing is stale — the
    loss is permanent and silent. That is jdoc#107's exact shape.

    Keeping the stale vectors is the recoverable outcome: they are the wrong
    width, which the query side degrades and discloses.
    """
    from jdocmunch_mcp.embeddings import provider as prov

    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384, n=5)

    monkeypatch.setattr(prov, "_get_provider", lambda: _BrokenProvider())
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity",
                        lambda n: ("BAAI/bge-base-en-v1.5", 768))

    sections = [_Sec("body", content_hash=f"h{i}") for i in range(5)]
    prov.embed_sections(sections, owner="local", name="corpus",
                        storage_path=str(tmp_path), prune=True)

    surviving = emb_cache.stored_hashes(str(tmp_path), "local", "corpus")
    assert len(surviving) == 5, (
        f"a provider outage destroyed the vector store: {len(surviving)} left"
    )
    header = emb_cache.identity(str(tmp_path), "local", "corpus")
    assert header["model"] == "all-MiniLM-L6-v2", (
        "the new header was written over surviving old vectors — the next run "
        "will see a matching identity and never re-embed"
    )


def test_a_genuinely_empty_corpus_still_purges_a_stale_sidecar(tmp_path, monkeypatch):
    """The other side: no outage, nothing to embed, stale identity ⇒ purge.

    Guards against 'fixing' the outage case by never purging at all.
    """
    from jdocmunch_mcp.embeddings import provider as prov

    _write_sidecar(tmp_path, "sentence-transformers", "all-MiniLM-L6-v2", 384, n=5)

    monkeypatch.setattr(prov, "_get_provider", lambda: _FakeProvider(768))
    monkeypatch.setattr(prov, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(prov, "_provider_identity",
                        lambda n: ("BAAI/bge-base-en-v1.5", 768))

    prov.embed_sections([], owner="local", name="corpus", storage_path=str(tmp_path))

    assert emb_cache.stored_hashes(str(tmp_path), "local", "corpus") == set()
    assert emb_cache.identity(str(tmp_path), "local", "corpus")["dim"] == 768


# ---------------------------------------------------------------------------
# 8. A paid cloud provider is not auto-escalated
# ---------------------------------------------------------------------------

def _use_paid(monkeypatch, model, dim, provider="openai"):
    from jdocmunch_mcp.embeddings import provider as prov
    p = _FakeProvider(dim)
    monkeypatch.setattr(prov, "_get_provider", lambda: p)
    monkeypatch.setattr(prov, "get_provider_name", lambda: provider)
    monkeypatch.setattr(prov, "_provider_identity", lambda n: (model, dim))
    return p


def _index_paid(monkeypatch, corpus, store, model, dim, provider="openai"):
    from jdocmunch_mcp.tools.index_local import index_local
    p = _use_paid(monkeypatch, model, dim, provider)
    result = index_local(
        path=str(corpus), name="rot", storage_path=str(store),
        use_ai_summaries=False, use_embeddings=True,
    )
    return result, p


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_paid_rotation_does_not_auto_re_embed(monkeypatch, corpus, tmp_path, provider):
    """⚠⚠ `watch.py` calls this path from a background daemon on every save.

    An unattended service must not re-send a whole corpus to a billed third
    party, and the watcher prints only "re-indexed N file(s)" — so the
    disclosure would never reach a human anyway.
    """
    store = tmp_path / "idx"
    monkeypatch.delenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", raising=False)
    _index_paid(monkeypatch, corpus, store, "text-embedding-3-small", 1536, provider)

    second, prov = _index_paid(
        monkeypatch, corpus, store, "text-embedding-3-large", 3072, provider
    )

    assert prov.embedded == 0, "a paid provider was billed for an unrequested re-embed"
    rot = second["embedding_rotation"]
    assert rot["action"] == "rebuild_required"
    assert "--rebuild" in rot["fix"]
    assert "JDOCMUNCH_ALLOW_PAID_EMBEDDINGS" in rot["fix"]


def test_gated_rotation_is_disclosed_on_the_no_change_payload(monkeypatch, corpus, tmp_path):
    """⚠ The payload that most needs it: nothing changed AND vectors are stale.

    The gated path returns from the "No changes detected" branch, so attaching
    the disclosure only to the full-rebuild result would omit it exactly here.
    """
    store = tmp_path / "idx"
    monkeypatch.delenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", raising=False)
    _index_paid(monkeypatch, corpus, store, "text-embedding-3-small", 1536)
    second, _ = _index_paid(monkeypatch, corpus, store, "text-embedding-3-large", 3072)

    assert second["message"] == "No changes detected"
    assert second["embedding_rotation"]["action"] == "rebuild_required"


def test_paid_rotation_leaves_the_old_vectors_intact(monkeypatch, corpus, tmp_path):
    """Declining to re-embed must not also destroy what is there."""
    store = tmp_path / "idx"
    monkeypatch.delenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", raising=False)
    _index_paid(monkeypatch, corpus, store, "text-embedding-3-small", 1536)
    before = emb_cache.stored_hashes(str(store), "local", "rot")
    assert before

    _index_paid(monkeypatch, corpus, store, "text-embedding-3-large", 3072)

    assert emb_cache.stored_hashes(str(store), "local", "rot") == before
    assert _sidecar_header(store)["model"] == "text-embedding-3-small"


def test_opting_in_restores_auto_escalation(monkeypatch, corpus, tmp_path):
    """The existing consent signal is the switch — no second knob."""
    store = tmp_path / "idx"
    monkeypatch.delenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", raising=False)
    _index_paid(monkeypatch, corpus, store, "text-embedding-3-small", 1536)

    monkeypatch.setenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "1")
    second, prov = _index_paid(monkeypatch, corpus, store, "text-embedding-3-large", 3072)

    assert prov.embedded > 0
    assert second["embedding_rotation"]["action"] == "full_re_embed"
    assert _sidecar_header(store)["model"] == "text-embedding-3-large"


def test_a_local_provider_is_never_gated(monkeypatch, corpus, tmp_path):
    """⚠ The gate must not catch sentence-transformers, which costs nothing."""
    store = tmp_path / "idx"
    monkeypatch.delenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", raising=False)
    _run_index(monkeypatch, corpus, store, "all-MiniLM-L6-v2", 384)
    second, prov = _run_index(monkeypatch, corpus, store, "BAAI/bge-base-en-v1.5", 768)

    assert prov.embedded > 0
    assert second["embedding_rotation"]["action"] == "full_re_embed"
