"""A provider name is a promise, and `semantic_search` is derived, never asserted.

Found in the field, as a matched pair of lies. An MCP host launched the server
with `JDOCMUNCH_EMBEDDING_PROVIDER=sentence-transformers` in an environment
where the package was not installed. `get_provider_name()` returned the name
unchecked, so `index_local`'s default `use_embeddings="auto"` resolved True; the
factory then raised ImportError, `_get_provider()` swallowed it into None, and
`embed_sections` silently returned every section unembedded. The response still
said `"semantic_search": true` — computed from intent (`use_embeddings and
get_provider_name() is not None`), not from the index — and every later search
fell back to lexical BM25 without a word. Semantic search was reported enabled
for weeks over an index holding zero vectors.

Two fixes, each with its own tests here:

1. **A name is only returned when it can embed.** An explicitly named provider
   whose backing package is not importable resolves to None, with a warning
   naming the missing package — matching how the openai-compatible branch has
   always validated its URL+model config before returning its name.

2. **The flag is DERIVED, never asserted** (test_ledger_flag_honesty.py,
   invariant 1 — the same rule, applied to the producer instead of the ledger).
   `index_local`/`index_repo` report `semantic_search` from the saved index's
   `_has_embeddings()`, the predicate `search_sections` gates hybrid retrieval
   on, so the flag cannot claim a channel the index has no data for.
"""

from __future__ import annotations

import pytest

import jdocmunch_mcp.embeddings.provider as provider


_EMBED_ENV = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "JDOCMUNCH_EMBEDDING_PROVIDER",
    "JDOCMUNCH_ALLOW_PAID_EMBEDDINGS",
    "JDOCMUNCH_OPENAI_COMPAT_URL",
    "JDOCMUNCH_OPENAI_COMPAT_MODEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No ambient embedding configuration, and no cross-test warn-once bleed."""
    for var in _EMBED_ENV:
        monkeypatch.delenv(var, raising=False)
    provider._WARNED_MISSING_PACKAGE.clear()
    yield monkeypatch
    provider._WARNED_MISSING_PACKAGE.clear()


# ---------------------------------------------------------------------------
# Fix 1: a named provider without its package resolves to None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["gemini", "openai", "sentence-transformers"])
def test_named_provider_without_package_resolves_to_none(clean_env, monkeypatch, name):
    """The defect, exactly. Fails on revert: returns the name, and every caller
    of `get_provider_name() is not None` reports semantic search as enabled
    while nothing can embed."""
    monkeypatch.setattr(provider, "_provider_package_available", lambda n: False)
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: False)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", name)
    assert provider.get_provider_name() is None
    assert provider.should_embed("auto") is False


@pytest.mark.parametrize("name", ["gemini", "openai", "sentence-transformers"])
def test_named_provider_with_package_is_honored(clean_env, monkeypatch, name):
    """Non-vacuity. A guard that always said None would pass the test above by
    breaking every working configuration."""
    monkeypatch.setattr(provider, "_provider_package_available", lambda n: True)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", name)
    assert provider.get_provider_name() == name


def test_missing_package_is_logged_not_silent(clean_env, monkeypatch, caplog):
    """A capability that vanishes without saying so is its own bug — the same
    rule the paid-cloud suppression warning enforces. The warning must name the
    installable package, or the user cannot act on it."""
    monkeypatch.setattr(provider, "_provider_package_available", lambda n: False)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")
    with caplog.at_level("WARNING"):
        provider.get_provider_name()
    assert any("sentence-transformers" in r.getMessage() for r in caplog.records), (
        f"expected a warning naming the missing package; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_missing_package_warns_once(clean_env, monkeypatch, caplog):
    """get_provider_name() runs on every index and every search; a per-call
    warning would drown the log it is trying to inform."""
    monkeypatch.setattr(provider, "_provider_package_available", lambda n: False)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "gemini")
    with caplog.at_level("WARNING"):
        provider.get_provider_name()
        provider.get_provider_name()
    hits = [r for r in caplog.records if "gemini" in r.getMessage()]
    assert len(hits) == 1


def test_module_injected_into_sys_modules_counts_as_available(clean_env, monkeypatch):
    """An entry in sys.modules IS importable, whatever find_spec thinks of it.
    Tests across the suite inject bare ModuleType fakes (no __spec__), on which
    find_spec raises ValueError instead of finding them; the guard must honor
    sys.modules first or every such fake reads as uninstalled."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    assert provider._provider_package_available("openai") is True


def test_auto_detect_skips_unavailable_and_falls_through(clean_env, monkeypatch):
    """Auto-detect must not stop dead at a keyed-but-uninstalled cloud provider
    when the offline fallback can still embed."""
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    clean_env.setenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "1")
    monkeypatch.setattr(
        provider, "_provider_package_available",
        lambda n: n == "sentence-transformers",
    )
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: True)
    assert provider.get_provider_name() == "sentence-transformers"


# ---------------------------------------------------------------------------
# Fix 2: index_local's semantic_search flag is derived from the saved index
# ---------------------------------------------------------------------------

def _write_corpus(tmp_path):
    docs = tmp_path / "corpus"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Calibrating the widget\n\nTurn the dial until the needle settles.\n"
        "\n## Draining the reservoir\n\nOpen the valve slowly.\n",
        encoding="utf-8",
    )
    return docs


def test_index_local_reports_false_when_nothing_embedded(clean_env, monkeypatch, tmp_path):
    """The response-side half of the defect. Provider named, package absent:
    embedding silently degrades to a no-op, and the report must say so rather
    than echo the configuration."""
    from jdocmunch_mcp.tools.index_local import index_local

    monkeypatch.setattr(provider, "_provider_package_available", lambda n: False)
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: False)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")

    docs = _write_corpus(tmp_path)
    result = index_local(str(docs), name="honesty-full", storage_path=str(tmp_path / "store"))
    assert result["success"] is True
    assert result["semantic_search"] is False

    # The incremental path builds its own response dict; it must not regress
    # independently (it did originally — both sites computed from intent).
    (docs / "guide.md").write_text(
        "# Calibrating the widget\n\nTurn the dial until the needle settles down.\n",
        encoding="utf-8",
    )
    result = index_local(str(docs), name="honesty-full", storage_path=str(tmp_path / "store"))
    assert result["success"] is True
    assert result.get("incremental") is True
    assert result["semantic_search"] is False


def test_index_local_reports_true_when_vectors_are_written(clean_env, monkeypatch, tmp_path):
    """Non-vacuity: a hardcoded False passes the test above. With a provider
    that actually embeds, the saved index has vectors and the flag follows."""
    from jdocmunch_mcp.tools.index_local import index_local

    class _FakeProvider:
        def embed_texts(self, texts, task_type="retrieval_document"):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(provider, "_get_provider", lambda: _FakeProvider())
    monkeypatch.setattr(provider, "_provider_package_available", lambda n: True)
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")

    docs = _write_corpus(tmp_path)
    result = index_local(str(docs), name="honesty-true", storage_path=str(tmp_path / "store"))
    assert result["success"] is True
    assert result["semantic_search"] is True
