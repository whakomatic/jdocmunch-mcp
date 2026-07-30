"""Tests for BM25+semantic hybrid search and the use_embeddings='auto' flag."""

import os
from unittest.mock import patch

import pytest

from jdocmunch_mcp.parser import parse_file
from jdocmunch_mcp.storage.doc_store import DocStore, DocIndex
from jdocmunch_mcp.embeddings.provider import should_embed


SAMPLE_MD = """# Guide

## Authentication

To sign in users, configure OAuth 2.0 with your provider.
Tokens are returned as JWTs and stored in the session.

## Payments

We use Stripe for credit card processing. See the billing module
for invoice generation and subscription management.

## Notifications

Email alerts go through SendGrid. Push notifications use FCM.
"""


def _make_index(tmp_path, with_embeddings=False):
    """Build a real on-disk index. Optionally attach fake embeddings to each section."""
    store = DocStore(base_path=str(tmp_path))
    sections = parse_file(SAMPLE_MD, "README.md", "test/repo")
    if with_embeddings:
        # Fake deterministic 3-dim embeddings: each section gets a unit vector
        # aligned to a distinct axis so cosine ranking is predictable.
        vecs = {
            "Authentication": [1.0, 0.0, 0.0],
            "Payments": [0.0, 1.0, 0.0],
            "Notifications": [0.0, 0.0, 1.0],
        }
        for sec in sections:
            sec.embedding = vecs.get(sec.title, [0.33, 0.33, 0.33])
    store.save_index(
        owner="local",
        name="hybrid_test",
        sections=sections,
        raw_files={"README.md": SAMPLE_MD},
        doc_types={".md": 1},
    )
    return store.load_index("local", "hybrid_test")


# ---------------------------------------------------------------------------
# should_embed flag resolution
# ---------------------------------------------------------------------------

def test_should_embed_true():
    assert should_embed(True) is True


def test_should_embed_false():
    assert should_embed(False) is False


def test_should_embed_auto_with_provider(monkeypatch):
    # Package availability is mocked: a named provider only resolves when its
    # backing package is importable, and the dev env installs no cloud SDKs.
    from jdocmunch_mcp.embeddings import provider as emb_provider
    monkeypatch.setattr(emb_provider, "_provider_package_available", lambda name: True)
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    assert should_embed("auto") is True


def test_should_embed_auto_without_provider(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "none")
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert should_embed("auto") is False


def test_should_embed_auto_case_insensitive(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "none")
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert should_embed("AUTO") is False
    assert should_embed("Auto") is False


# Regression: jdoc#18 — non-empty string was truthy, so "false" enabled embeddings.
def test_should_embed_parses_string_false_as_false(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")
    assert should_embed("false") is False


def test_should_embed_parses_common_string_booleans(monkeypatch):
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")

    for truthy in ("true", "1", "yes", "on", "t", "y", "TRUE", "Yes", " true "):
        assert should_embed(truthy) is True, f"{truthy!r} should be True"

    for falsy in ("false", "0", "no", "off", "f", "n", "FALSE", "No", " false ", ""):
        assert should_embed(falsy) is False, f"{falsy!r} should be False"


def test_should_embed_unknown_string_preserves_legacy_truthy_behavior(monkeypatch):
    # 1.x compat: an unrecognised non-empty string remains truthy (matches
    # pre-1.66.1 behavior). Only the known false-y strings flip; typos like
    # "flase" still enable embeddings rather than silently disabling them.
    monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")
    assert should_embed("flase") is True
    assert should_embed("enabled") is True


# ---------------------------------------------------------------------------
# warmup() — jdoc#19: prevent stdio hang on sentence-transformers lazy-load
# ---------------------------------------------------------------------------

def test_warmup_skips_network_providers(monkeypatch):
    """gemini/openai/openai-compatible don't need warmup — they're network calls,
    not local model loads, so warming them just adds a startup round-trip."""
    from jdocmunch_mcp.embeddings import provider as emb_provider

    called = {"n": 0}

    def fake_embed_query(_q):
        called["n"] += 1
        return [0.0]

    monkeypatch.setattr(emb_provider, "embed_query", fake_embed_query)

    for name in ("gemini", "openai", "openai-compatible", "none"):
        monkeypatch.setattr(emb_provider, "get_provider_name", lambda n=name: n)
        assert emb_provider.warmup() == ""

    assert called["n"] == 0


def test_warmup_returns_no_provider_when_unconfigured(monkeypatch):
    from jdocmunch_mcp.embeddings import provider as emb_provider
    monkeypatch.setattr(emb_provider, "get_provider_name", lambda: None)
    assert emb_provider.warmup() == ""


def test_warmup_invokes_embed_query_for_sentence_transformers(monkeypatch):
    from jdocmunch_mcp.embeddings import provider as emb_provider

    queries = []

    def fake_embed_query(q):
        queries.append(q)
        return [0.1]

    monkeypatch.setattr(emb_provider, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(emb_provider, "embed_query", fake_embed_query)

    result = emb_provider.warmup()
    assert result == "sentence-transformers"
    assert len(queries) == 1


def test_warmup_swallows_embed_query_failure(monkeypatch):
    """A warmup failure must not crash server startup; the real call will retry."""
    from jdocmunch_mcp.embeddings import provider as emb_provider

    def boom(_q):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(emb_provider, "get_provider_name", lambda: "sentence-transformers")
    monkeypatch.setattr(emb_provider, "embed_query", boom)

    assert emb_provider.warmup() == ""


# ---------------------------------------------------------------------------
# Hybrid fusion: lexical-only path when no embeddings exist
# ---------------------------------------------------------------------------

def test_search_lexical_when_no_embeddings(tmp_path):
    index = _make_index(tmp_path, with_embeddings=False)
    results = index.search("authentication", max_results=5)
    assert results
    assert results[0]["title"] == "Authentication"


def test_search_semantic_false_forces_lexical(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    # Even with embeddings present, semantic=False must use lexical only.
    # Query "sign in" is paraphrased — lexical won't find Authentication.
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [1.0, 0.0, 0.0]
        results = index.search("sign in", max_results=5, semantic=False)
    # Lexical finds "sign in" as a substring in Authentication's content.
    # The key assertion: embed_query was NOT called (semantic fully skipped).
    mock_eq.assert_not_called()


# ---------------------------------------------------------------------------
# Hybrid fusion: combines lexical + semantic
# ---------------------------------------------------------------------------

def test_hybrid_combines_both_signals(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    # Query vector perfectly aligns with Payments axis; lexical misses entirely.
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [0.0, 1.0, 0.0]  # aligns with Payments
        results = index.search("Stripe processor", max_results=3, semantic_weight=0.7)
    assert results
    # Payments section has "Stripe" lexically AND perfect semantic alignment.
    assert results[0]["title"] == "Payments"


def test_semantic_only_ignores_lexical(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    # Query vec aligns purely with Notifications axis.
    # Lexically, "alerts" also matches Notifications content — so both point there.
    # Use a query that is lexically aligned with Payments to prove semantic dominates.
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [0.0, 0.0, 1.0]  # Notifications axis
        results = index.search("Stripe", max_results=3, semantic_only=True)
    assert results
    assert results[0]["title"] == "Notifications"


def test_semantic_weight_zero_is_lexical_only(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [0.0, 1.0, 0.0]
        results = index.search("authentication", max_results=3, semantic_weight=0.0)
    assert results[0]["title"] == "Authentication"
    # With weight=0, hybrid is skipped → embed_query never called.
    mock_eq.assert_not_called()


def test_hybrid_empty_query_vec_falls_back_to_lexical(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = None  # provider disabled at query time
        results = index.search("authentication", max_results=3, semantic=True)
    assert results
    assert results[0]["title"] == "Authentication"


def test_hybrid_filters_by_doc_path(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [1.0, 0.0, 0.0]
        results = index.search(
            "authentication", max_results=5, doc_path="nonexistent.md"
        )
    assert results == []


def test_results_strip_content_and_embedding(tmp_path):
    index = _make_index(tmp_path, with_embeddings=True)
    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [1.0, 0.0, 0.0]
        results = index.search("authentication", max_results=3)
    assert results
    for r in results:
        assert "content" not in r
        assert "embedding" not in r


# ---------------------------------------------------------------------------
# search_sections tool integration
# ---------------------------------------------------------------------------

def test_search_sections_reports_hybrid_mode(tmp_path):
    _make_index(tmp_path, with_embeddings=True)
    from jdocmunch_mcp.tools.search_sections import search_sections

    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [1.0, 0.0, 0.0]
        result = search_sections(
            repo="local/hybrid_test",
            query="authentication",
            storage_path=str(tmp_path),
        )
    assert result["_meta"]["search_mode"] == "hybrid"
    assert result["_meta"]["semantic_weight"] == 0.5


def test_search_sections_reports_lexical_without_embeddings(tmp_path):
    _make_index(tmp_path, with_embeddings=False)
    from jdocmunch_mcp.tools.search_sections import search_sections

    result = search_sections(
        repo="local/hybrid_test",
        query="authentication",
        storage_path=str(tmp_path),
    )
    assert result["_meta"]["search_mode"] == "lexical"
    assert "tip" in result["_meta"]


def test_search_sections_semantic_only_mode(tmp_path):
    _make_index(tmp_path, with_embeddings=True)
    from jdocmunch_mcp.tools.search_sections import search_sections

    with patch("jdocmunch_mcp.storage.doc_store.embed_query") as mock_eq:
        mock_eq.return_value = [0.0, 0.0, 1.0]
        result = search_sections(
            repo="local/hybrid_test",
            query="whatever",
            semantic_only=True,
            storage_path=str(tmp_path),
        )
    assert result["_meta"]["search_mode"] == "semantic_only"
    assert result["results"][0]["title"] == "Notifications"
