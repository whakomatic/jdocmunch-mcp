"""A provider name means embedding can run, and `semantic_search` is derived.

Found in the field. An MCP host launched the server with
`JDOCMUNCH_EMBEDDING_PROVIDER=sentence-transformers` in an environment where
the package was not installed. `get_provider_name()` returned the name
unchecked, so `index_local`'s default `use_embeddings="auto"` resolved True;
the factory then raised ImportError, `_get_provider()` swallowed it into None,
and `embed_sections` returned every section unembedded. The response still
said `"semantic_search": true`, computed from configuration
(`use_embeddings and get_provider_name() is not None`) rather than from the
index, and every later search ran lexical BM25 with no warning.

Two fixes, each tested here:

1. A named provider whose backing package is not importable resolves to None,
   with one warning naming the package. The openai-compatible branch already
   validated its URL and model before returning its name; this applies the
   same rule to the package.

2. `index_local` and `index_repo` report `semantic_search` from the saved
   index's `_has_embeddings()`, the predicate `search_sections` gates hybrid
   retrieval on, so the flag cannot claim vectors the index does not hold.

The "absent" tests block a package through seams that exist with or without
the fix (package metadata, `sys.modules`, the provider factory), so on the
unfixed code they fail on their assertions rather than on a missing helper.
"""

from __future__ import annotations

import sys
import types

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

# Import name the provider's package is loaded under; see _PROVIDER_PACKAGES.
_CLOUD_IMPORTS = {
    "gemini": "google.generativeai",
    "openai": "openai",
    "openai-compatible": "openai",
}


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No ambient embedding configuration, no warn-once or cache bleed."""
    for var in _EMBED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DOC_INDEX_PATH", str(tmp_path / "doc-index"))
    getattr(provider, "_WARNED_MISSING_PACKAGE", set()).clear()
    provider._reset_provider_cache()
    yield monkeypatch
    getattr(provider, "_WARNED_MISSING_PACKAGE", set()).clear()
    provider._reset_provider_cache()


def _raise_import_error():
    raise ImportError("No module named 'sentence_transformers'")


def _block_package(monkeypatch, name):
    """Make ``name``'s backing package look uninstalled and unimportable."""
    if name in _CLOUD_IMPORTS:
        # A None entry in sys.modules makes `import x` raise ImportError.
        monkeypatch.setitem(sys.modules, _CLOUD_IMPORTS[name], None)
        return
    if name == "fastembed":
        monkeypatch.setattr(provider, "_fastembed_available", lambda: False)
        monkeypatch.setitem(sys.modules, "fastembed", None)
        return
    assert name == "sentence-transformers"
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    # The import itself happens in the factory (or a worker child). Replace
    # the factory so the dev env's real install cannot embed anyway, and skip
    # the subprocess import probe, which would find the real install.
    monkeypatch.setitem(
        provider._PROVIDER_FACTORIES, "sentence-transformers", _raise_import_error
    )
    monkeypatch.setattr(provider, "_embed_worker_enabled", lambda: True)


def _provide_package(monkeypatch, name):
    """Make ``name``'s backing package look installed, without importing it."""
    if name in _CLOUD_IMPORTS:
        import_name = _CLOUD_IMPORTS[name]
        monkeypatch.setitem(sys.modules, import_name, types.ModuleType(import_name))
    elif name == "fastembed":
        monkeypatch.setattr(provider, "_fastembed_available", lambda: True)
    else:
        monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: True)


def _name_provider(env, name):
    env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", name)
    if name == "openai-compatible":
        env.setenv("JDOCMUNCH_OPENAI_COMPAT_URL", "http://localhost:1234/v1")
        env.setenv("JDOCMUNCH_OPENAI_COMPAT_MODEL", "some-model")


_ALL_NAMED = ["gemini", "openai", "openai-compatible", "fastembed", "sentence-transformers"]


# ---------------------------------------------------------------------------
# Fix 1: a named provider without its package resolves to None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _ALL_NAMED)
def test_named_provider_without_package_resolves_to_none(clean_env, monkeypatch, name):
    """The defect. Without the fix the name comes back, and every caller of
    `get_provider_name() is not None` reports semantic search as enabled
    while nothing can embed."""
    _block_package(monkeypatch, name)
    _name_provider(clean_env, name)
    assert provider.get_provider_name() is None
    assert provider.should_embed("auto") is False


@pytest.mark.parametrize("name", _ALL_NAMED)
def test_named_provider_with_package_is_honored(clean_env, monkeypatch, name):
    """Non-vacuity: a check that always said None would pass the test above
    by breaking every working configuration."""
    _provide_package(monkeypatch, name)
    _name_provider(clean_env, name)
    assert provider.get_provider_name() == name


def test_missing_package_is_logged(clean_env, monkeypatch, caplog):
    """The warning must name the installable package, or the user cannot act
    on it."""
    _block_package(monkeypatch, "sentence-transformers")
    _name_provider(clean_env, "sentence-transformers")
    with caplog.at_level("WARNING", logger=provider.logger.name):
        provider.get_provider_name()
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "sentence-transformers" in m and "not importable" in m for m in messages
    ), f"expected a warning naming the missing package; got {messages}"


def test_missing_package_warns_once(clean_env, monkeypatch, caplog):
    """get_provider_name() runs on every index and every search; a warning per
    call would flood the log."""
    _block_package(monkeypatch, "gemini")
    _name_provider(clean_env, "gemini")
    with caplog.at_level("WARNING", logger=provider.logger.name):
        provider.get_provider_name()
        provider.get_provider_name()
    hits = [r for r in caplog.records if "not importable" in r.getMessage()]
    assert len(hits) == 1


def test_module_injected_into_sys_modules_counts_as_available(clean_env, monkeypatch):
    """Tests across the suite inject bare ModuleType fakes (no __spec__), on
    which find_spec raises ValueError. The check must read sys.modules first
    or every such fake reads as uninstalled."""
    monkeypatch.setitem(sys.modules, "openai", types.ModuleType("openai"))
    assert provider._provider_package_available("openai") is True


def test_auto_detect_skips_unavailable_and_falls_through(clean_env, monkeypatch):
    """Auto-detect must not stop at a keyed but uninstalled cloud provider when
    an offline provider can still embed."""
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    clean_env.setenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "1")
    _block_package(monkeypatch, "openai")
    monkeypatch.setattr(provider, "_fastembed_available", lambda: False)
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
    """Provider named, package absent: embedding degrades to a no-op, and each
    of the three response paths (full, incremental, no change) must say so
    rather than echo the configuration."""
    from jdocmunch_mcp.tools.index_local import index_local

    _block_package(monkeypatch, "sentence-transformers")
    _name_provider(clean_env, "sentence-transformers")
    docs = _write_corpus(tmp_path)
    store = str(tmp_path / "store")

    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result["success"] is True
    assert result["semantic_search"] is False

    (docs / "guide.md").write_text(
        "# Calibrating the widget\n\nTurn the dial until the needle settles down.\n",
        encoding="utf-8",
    )
    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result["success"] is True
    assert result.get("incremental") is True
    assert result["semantic_search"] is False

    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result.get("message") == "No changes detected"
    assert result["semantic_search"] is False


def test_index_local_reports_true_when_vectors_are_written(clean_env, monkeypatch, tmp_path):
    """Non-vacuity: a hardcoded False passes the test above. With a provider
    that embeds, the saved index has vectors on every path, including a
    deletion-only refresh, and the flag follows."""
    from jdocmunch_mcp.tools.index_local import index_local

    class _FakeProvider:
        def embed_texts(self, texts, task_type="retrieval_document"):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(provider, "_get_provider", lambda: _FakeProvider())
    _provide_package(monkeypatch, "sentence-transformers")
    _name_provider(clean_env, "sentence-transformers")
    docs = _write_corpus(tmp_path)
    (docs / "extra.md").write_text("# Spare parts\n\nKeep a spare gasket.\n",
                                   encoding="utf-8")
    store = str(tmp_path / "store")

    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result["success"] is True
    assert result["semantic_search"] is True

    (docs / "guide.md").write_text(
        "# Calibrating the widget\n\nTurn the dial until the needle settles down.\n",
        encoding="utf-8",
    )
    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result.get("incremental") is True
    assert result["semantic_search"] is True

    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result.get("message") == "No changes detected"
    assert result["semantic_search"] is True

    # A deletion-only refresh saves no new sections; the vectors that remain
    # live in the sidecar, and the flag must still see them.
    (docs / "extra.md").unlink()
    result = index_local(str(docs), name="honesty", storage_path=store,
                         use_ai_summaries=False)
    assert result.get("incremental") is True
    assert result.get("deleted") == 1
    assert result["semantic_search"] is True
