"""Same-width model mismatch between stored vectors and the query encoder.

jdoc#109 made a WIDTH mismatch loud (384 stored, 768 query). Two models that
both emit 384 dims pass that check: sentence-transformers all-MiniLM-L6-v2 and
FastEmbed BAAI/bge-small-en-v1.5. With vectors from one and a query from the
other, cosine runs across two unrelated vector spaces and the semantic lane
returns noise that reads exactly like a working hybrid search.

Measured on one machine (2026-09-19): 11 of 59 sidecars were MiniLM while the
shared server queried with bge-small, because a second writer (a cli install
with sentence-transformers and no fastembed) kept rotating indexes back.

The guard compares the sidecar header with ``sidecar_identity`` of the active
provider, the same function the writer uses, so the jdoc#126 alias still
counts as a match. It fires only when BOTH identities are known: an
``__inline__`` placeholder, a header-less sidecar, or no active provider all
leave behaviour unchanged.
"""

import json
from unittest.mock import patch

import pytest

from jdocmunch_mcp.embeddings import provider as emb_provider
from jdocmunch_mcp.parser import parse_file
from jdocmunch_mcp.storage.doc_store import DocStore


SAMPLE_MD = """# Guide

## Authentication

To sign in users, configure OAuth 2.0 with your provider.

## Payments

We use Stripe for credit card processing.

## Notifications

Email alerts go through SendGrid.
"""

MINILM = ("sentence-transformers", "all-MiniLM-L6-v2", None)
BGE = ("fastembed", "BAAI/bge-small-en-v1.5", None)


def _make_index(tmp_path, header):
    store = DocStore(base_path=str(tmp_path))
    sections = parse_file(SAMPLE_MD, "README.md", "test/repo")
    vecs = {"Authentication": [1.0, 0.0, 0.0], "Payments": [0.0, 1.0, 0.0],
            "Notifications": [0.0, 0.0, 1.0]}
    for sec in sections:
        sec.embedding = vecs.get(sec.title, [0.33, 0.33, 0.33])
    store.save_index(owner="local", name="mm", sections=sections,
                     raw_files={"README.md": SAMPLE_MD}, doc_types={".md": 1})
    index = store.load_index("local", "mm")
    sidecar = index._embeddings_sidecar
    assert sidecar, "fixture bug: no sidecar written"
    with open(sidecar, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert json.loads(lines[0]).get("_header") is True
    if header is not None:
        provider, model, dim = header
        lines[0] = json.dumps({"_header": True, "provider": provider,
                               "model": model, "dim": dim})
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return store.load_index("local", "mm")


@pytest.fixture
def active(monkeypatch):
    """Set the active query provider's identity."""
    def _set(identity):
        if identity is None:
            monkeypatch.setattr(emb_provider, "get_provider_name", lambda: None)
            return
        monkeypatch.setattr(emb_provider, "get_provider_name", lambda: identity[0])
        monkeypatch.setattr(emb_provider, "sidecar_identity", lambda name: identity)
    return _set


def _search(tmp_path, **kw):
    from jdocmunch_mcp.tools.search_sections import search_sections
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as eq:
        eq.return_value = [0.0, 1.0, 0.0]
        result = search_sections(repo="local/mm", query="authentication",
                                 storage_path=str(tmp_path), **kw)
    return result, eq


def test_mismatched_model_disables_semantic_and_says_so(tmp_path, active):
    _make_index(tmp_path, header=MINILM)
    active(BGE)
    result, eq = _search(tmp_path)
    meta = result["_meta"]
    assert meta["search_mode"] == "lexical"
    stale = meta["embedding_stale"]
    assert stale["semantic_disabled"] is True
    assert stale["stored_model"] == {"provider": "sentence-transformers",
                                     "model": "all-MiniLM-L6-v2"}
    assert stale["query_model"] == {"provider": "fastembed",
                                    "model": "BAAI/bge-small-en-v1.5"}
    eq.assert_not_called()
    assert result["results"][0]["title"] == "Authentication"


def test_mismatch_semantic_only_returns_nothing_without_embedding(tmp_path, active):
    index = _make_index(tmp_path, header=MINILM)
    active(BGE)
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as eq:
        eq.return_value = [0.0, 1.0, 0.0]
        assert index.search("Stripe", semantic_only=True) == []
    eq.assert_not_called()


def test_matching_model_stays_hybrid(tmp_path, active):
    _make_index(tmp_path, header=BGE)
    active(BGE)
    result, eq = _search(tmp_path)
    assert result["_meta"]["search_mode"] == "hybrid"
    assert "embedding_stale" not in result["_meta"]
    eq.assert_called()


@pytest.mark.parametrize("header", [None, ("__inline__", "__inline__", None)])
def test_unknown_stored_identity_is_not_judged(tmp_path, active, header):
    # None keeps the save path's own header, which is the __inline__ placeholder.
    _make_index(tmp_path, header=header)
    active(BGE)
    result, _ = _search(tmp_path)
    assert result["_meta"]["search_mode"] == "hybrid"
    assert "embedding_stale" not in result["_meta"]


def test_no_active_provider_is_not_judged(tmp_path, active):
    _make_index(tmp_path, header=MINILM)
    active(None)
    result, _ = _search(tmp_path)
    assert result["_meta"]["search_mode"] == "hybrid"
    assert "embedding_stale" not in result["_meta"]


def test_dim_mismatch_counts_only_when_both_known(tmp_path, active):
    _make_index(tmp_path, header=("openai-compatible", "u::m", 768))
    active(("openai-compatible", "u::m", None))
    result, _ = _search(tmp_path)
    assert result["_meta"]["search_mode"] == "hybrid"


@pytest.mark.asyncio
async def test_warning_survives_the_default_meta_stripping(tmp_path, active, monkeypatch):
    """jdoc's default meta_fields strips `_meta` entirely, so a warning left
    inside it never reaches the client; the server re-attaches it after the
    filter, like the budget and ignored-arguments blocks."""
    from jdocmunch_mcp import config as c
    from jdocmunch_mcp.server import call_tool

    _make_index(tmp_path, header=MINILM)
    active(BGE)
    monkeypatch.setenv("DOC_INDEX_PATH", str(tmp_path))
    monkeypatch.setattr(c, "get_meta_fields", lambda: [])
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as eq:
        eq.return_value = [0.0, 1.0, 0.0]
        out = await call_tool("search_sections", {"repo": "local/mm", "query": "authentication"})
    body = json.loads(out[0].text)
    assert body["_meta"]["embedding_stale"]["semantic_disabled"] is True
