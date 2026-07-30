"""Embedding providers for semantic section search.

Supports Gemini (text-embedding-004), OpenAI (text-embedding-3-small),
OpenAI-compatible endpoints, and sentence-transformers (fully offline,
no API key required).

Auto-detection priority (first available wins):
    1. JDOCMUNCH_EMBEDDING_PROVIDER env var (gemini/openai/openai-compatible/sentence-transformers/none)
    2. GOOGLE_API_KEY → Gemini            (opt-in, see below)
    3. OPENAI_API_KEY → OpenAI            (opt-in, see below)
    4. sentence-transformers installed → local offline model

⚠ Steps 2 and 3 are PAID CLOUD providers and are SKIPPED by auto-detect unless
JDOCMUNCH_ALLOW_PAID_EMBEDDINGS is set. A bare key in the environment must not
silently bill, and must not silently send the indexed corpus to a third party.
Naming the provider in step 1 is always honored.

⚠ Honored, but not invented: a named provider whose package is not installed
resolves to no provider at all, with a warning. `get_provider_name()` returning
a name means embedding can actually run, which is what its callers assume when
they report semantic search as available.

Set JDOCMUNCH_EMBEDDING_PROVIDER=none to disable all embedding.
"""

import importlib.util
import logging
import math
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

# Bump when _section_embed_text's derivation changes, so the content_hash-keyed
# embedding cache re-embeds instead of serving vectors built from the old text.
_EMBED_TEXT_VERSION = "pv1"


def _section_embed_text(section) -> str:
    """Build the text to embed for a section.

    Prepends title so short-titled sections (e.g. "Emotional Consequences"
    followed by a bullet list) still get a semantically rich embedding. The
    content is reduced to its prose view (frontmatter + fences stripped, #58)
    BEFORE the cap, so the embed window holds prose rather than YAML/TOML keys
    or fenced code, matching the BM25 channel's text.
    """
    from ..retrieval.tokenize import prose_view
    parts = [section.title]
    if section.summary and section.summary != section.title:
        parts.append(section.summary)
    if section.content:
        parts.append(prose_view(section.content).strip()[:1000])
    return "\n".join(parts)


def _embed_cache_key(section) -> str:
    """Cache key for a section's embedding: content_hash salted with the embed
    text-derivation version, so a derivation change (#58) invalidates cleanly."""
    h = getattr(section, "content_hash", "") or ""
    return f"{h}#{_EMBED_TEXT_VERSION}" if h else ""


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------

def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _openai_compat_url() -> str:
    return os.environ.get("JDOCMUNCH_OPENAI_COMPAT_URL", "").strip()


def _openai_compat_model() -> str:
    return os.environ.get("JDOCMUNCH_OPENAI_COMPAT_MODEL", "").strip()


def _openai_compat_api_key() -> str:
    return os.environ.get("JDOCMUNCH_OPENAI_COMPAT_API_KEY") or "local"


def _openai_compat_batch_size(default: int = 32) -> int:
    value = os.environ.get("JDOCMUNCH_OPENAI_COMPAT_BATCH_SIZE", "").strip()
    if not value:
        return default
    try:
        batch_size = int(value)
    except ValueError:
        return default
    return batch_size if batch_size > 0 else default


def _sentence_transformers_available() -> bool:
    """Return True if sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# Embedding providers that bill a remote cloud account per call AND send the
# indexed text off the machine. A bare env key for any of these must NEVER
# auto-enable embedding: it silently spends money and, worse, ships the corpus
# to a third party. Discovered when `index_local` on a PRIVATE memory store
# auto-selected OpenAI from an ambient OPENAI_API_KEY and began embedding it.
#
# The summarizer path already had this guard (`batch_summarize._PAID_CLOUD_PROVIDERS`)
# with the same rationale. It was never ported here, so AI summaries were
# correctly suppressed while embeddings sailed through the identical hazard.
# Naming the provider (JDOCMUNCH_EMBEDDING_PROVIDER) is always honored.
_PAID_CLOUD_EMBEDDING_PROVIDERS = frozenset({"gemini", "openai"})
_WARNED_SUPPRESSED_PAID_EMBED: set = set()

# (env var, provider name) in auto-detect priority order.
_EMBED_AUTO_DETECT_ORDER = (
    ("GOOGLE_API_KEY", "gemini"),
    ("OPENAI_API_KEY", "openai"),
)


# Backing package per provider, as (import name, install name). A provider whose
# package is absent CANNOT embed: `_get_provider` swallows the ImportError its
# factory raises and returns None, `embed_sections` then returns its sections
# untouched, and callers that ask `get_provider_name() is not None` report
# semantic search as enabled over an index holding zero vectors. Naming a
# provider is a deliberate choice, but it cannot conjure the package, so the
# name is honored only when the package is actually importable.
_PROVIDER_PACKAGES = {
    "gemini": ("google.generativeai", "google-generativeai"),
    "openai": ("openai", "openai"),
    "openai-compatible": ("openai", "openai"),
    "sentence-transformers": ("sentence_transformers", "sentence-transformers"),
}
_WARNED_MISSING_PACKAGE: set = set()


def _provider_package_available(name: str) -> bool:
    """Whether the named provider's backing package is importable."""
    if name == "sentence-transformers":
        return _sentence_transformers_available()
    package = _PROVIDER_PACKAGES.get(name)
    if not package:
        return False
    # An entry in sys.modules IS importable, whatever find_spec thinks of it —
    # tests inject bare ModuleType fakes there, and find_spec raises ValueError
    # on a module whose __spec__ is None instead of finding it.
    if package[0] in sys.modules:
        return sys.modules[package[0]] is not None
    try:
        return importlib.util.find_spec(package[0]) is not None
    except (ImportError, ValueError):
        # A namespace-package parent that is itself absent raises rather than
        # returning None. Absent is absent either way.
        return False


def _usable_provider(name: str) -> Optional[str]:
    """Return ``name`` when it can actually embed, else None with one warning.

    Reporting no provider is the honest answer for a provider that cannot run:
    the alternative is a name that every caller reads as "semantic search is on"
    while nothing embeds. Indexing still succeeds, lexically.
    """
    if _provider_package_available(name):
        return name
    if name not in _WARNED_MISSING_PACKAGE:
        _WARNED_MISSING_PACKAGE.add(name)
        import_name, install_name = _PROVIDER_PACKAGES.get(name, (name, name))
        logger.warning(
            "Embedding provider %s was selected but its package (%s) is not "
            "importable here — treating embeddings as DISABLED rather than "
            "claiming a provider that cannot embed. Install %s to enable it, or "
            "set JDOCMUNCH_EMBEDDING_PROVIDER=none to silence this. Indexing "
            "continues with lexical BM25 search.",
            name, import_name, install_name,
        )
    return None


def _paid_embeddings_allowed() -> bool:
    """Whether the user explicitly opted in to paid-cloud auto-embedding.

    Off by default: an ambient cloud API key never bills, and never exports the
    corpus, on its own. Turn on with JDOCMUNCH_ALLOW_PAID_EMBEDDINGS=1. Naming a
    provider explicitly (JDOCMUNCH_EMBEDDING_PROVIDER) is always honored and does
    not need this.
    """
    return os.environ.get("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def get_provider_name() -> Optional[str]:
    """Return the active provider name, or None if embeddings are disabled.

    Auto-detect NEVER selects a paid cloud provider from a bare env key unless
    JDOCMUNCH_ALLOW_PAID_EMBEDDINGS is set. Naming the provider explicitly
    bypasses this, because that is a deliberate choice rather than an ambient one.

    Every returned name is one whose backing package is importable. A name is a
    promise that embedding can run, and callers spend it that way — reporting
    semantic search as enabled, skipping the lexical-only disclosure. Naming an
    uninstalled provider therefore yields None, not the name.
    """
    explicit = os.environ.get("JDOCMUNCH_EMBEDDING_PROVIDER", "").lower().strip()
    if explicit == "gemini":
        return _usable_provider("gemini")
    if explicit == "openai":
        return _usable_provider("openai")
    if explicit == "openai-compatible":
        # Not in the paid set: it requires an explicitly configured URL + model,
        # which is itself the opt-in, and the common target is a local runtime.
        if _openai_compat_url() and _openai_compat_model():
            return _usable_provider("openai-compatible")
        return None
    if explicit in ("sentence-transformers", "sentence_transformers", "local"):
        return _usable_provider("sentence-transformers")
    if explicit == "none":
        return None
    # Auto-detect: cloud providers first, then offline fallback.
    allow_paid = _paid_embeddings_allowed()
    for env_var, name in _EMBED_AUTO_DETECT_ORDER:
        if not os.environ.get(env_var):
            continue
        if not allow_paid and name in _PAID_CLOUD_EMBEDDING_PROVIDERS:
            if name not in _WARNED_SUPPRESSED_PAID_EMBED:
                _WARNED_SUPPRESSED_PAID_EMBED.add(name)
                logger.warning(
                    "%s is set but paid-cloud embeddings are opt-in — NOT billing "
                    "%s automatically, and NOT sending indexed text off this "
                    "machine. To enable, set JDOCMUNCH_EMBEDDING_PROVIDER=%s (or "
                    "JDOCMUNCH_ALLOW_PAID_EMBEDDINGS=1). Indexing continues with "
                    "lexical BM25 search.",
                    env_var, name, name,
                )
            continue
        usable = _usable_provider(name)
        if usable:
            return usable
        # Package absent: fall through to the next candidate rather than
        # abandoning auto-detect, so a second configured key still wins.
    if _sentence_transformers_available():
        return "sentence-transformers"
    return None


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class _GeminiProvider:
    """Embed via Google Gemini text-embedding-004 (768 dims)."""

    MODEL = "models/text-embedding-004"
    BATCH_SIZE = 50  # conservative to avoid rate limits

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._genai = genai

    def embed_texts(self, texts: list, task_type: str = "retrieval_document") -> list:
        embeddings = []
        for text in texts:
            try:
                result = self._genai.embed_content(
                    model=self.MODEL,
                    content=text,
                    task_type=task_type,
                )
                embeddings.append(result["embedding"])
            except Exception:
                embeddings.append([])
        return embeddings


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class _OpenAIProvider:
    """Embed via OpenAI text-embedding-3-small (1536 dims)."""

    MODEL = "text-embedding-3-small"
    BATCH_SIZE = 100

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed_texts(self, texts: list, task_type: str = "retrieval_document") -> list:
        # task_type is ignored for OpenAI — included for interface compatibility
        embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i:i + self.BATCH_SIZE]
            try:
                response = self._client.embeddings.create(model=self.MODEL, input=batch)
                embeddings.extend([e.embedding for e in response.data])
            except Exception:
                embeddings.extend([[] for _ in batch])
        return embeddings


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class _OpenAICompatibleProvider:
    """Embed via a caller-supplied OpenAI-compatible embeddings endpoint."""

    BATCH_SIZE = 32

    def __init__(self):
        base_url = _openai_compat_url()
        model = _openai_compat_model()
        if not base_url:
            raise ValueError("No JDOCMUNCH_OPENAI_COMPAT_URL")
        if not model:
            raise ValueError("No JDOCMUNCH_OPENAI_COMPAT_MODEL")

        from openai import OpenAI

        self.model = model
        self.batch_size = _openai_compat_batch_size(self.BATCH_SIZE)
        self._client = OpenAI(api_key=_openai_compat_api_key(), base_url=base_url)
        self.dim = self._probe_dim()

    def _probe_dim(self) -> Optional[int]:
        """Discover the endpoint's actual embedding dim with a one-token canary.

        Closes the silent-corruption window where a backing-model swap behind
        the same URL/model env vars (e.g. retagging an Ollama model) would mix
        vectors of different dims in the on-disk cache (jdoc#20).

        Failure is non-fatal: returns None and the cache layer falls back to
        its wildcard-dim behavior (pre-v1.66.3 semantics). Network outage,
        misbehaved endpoint, or any other probe error degrades gracefully.
        """
        try:
            response = self._client.embeddings.create(model=self.model, input=["."])
            vec = response.data[0].embedding
            n = len(vec)
            return n if n > 0 else None
        except Exception:
            return None

    def embed_texts(self, texts: list, task_type: str = "retrieval_document") -> list:
        # task_type is ignored for OpenAI-compatible endpoints.
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
                embeddings.extend([e.embedding for e in response.data])
            except Exception:
                embeddings.extend([[] for _ in batch])
        return embeddings


# ---------------------------------------------------------------------------
# sentence-transformers provider (fully offline)
# ---------------------------------------------------------------------------

class _SentenceTransformersProvider:
    """Embed via sentence-transformers (all-MiniLM-L6-v2 by default, 384 dims).

    Runs entirely offline — no API key required. Install with:
        pip install sentence-transformers
    Override the model with JDOCMUNCH_ST_MODEL env var.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    BATCH_SIZE = 64

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("JDOCMUNCH_ST_MODEL", self.DEFAULT_MODEL)
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list, task_type: str = "retrieval_document") -> list:
        # task_type is ignored — sentence-transformers handles asymmetric search
        # via separate query/passage models when needed; for MiniLM it's symmetric.
        try:
            embeddings = self._model.encode(texts, batch_size=self.BATCH_SIZE, show_progress_bar=False)
            return [emb.tolist() for emb in embeddings]
        except Exception:
            return [[] for _ in texts]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Provider cache (B7) — avoid re-instantiation on every search query.
#
# A factory map is exposed so tests can stub providers; production code reads
# only via _get_provider().
# ---------------------------------------------------------------------------

_PROVIDER_FACTORIES: dict = {
    "gemini": _GeminiProvider,
    "openai": _OpenAIProvider,
    "openai-compatible": _OpenAICompatibleProvider,
    "sentence-transformers": _SentenceTransformersProvider,
}

# Cache: {(provider_name, model_signature): provider_instance}
_PROVIDER_CACHE: dict = {}


def _provider_signature(name: str) -> tuple:
    """Compute a cache key that invalidates when env-driven model choice changes."""
    if name == "sentence-transformers":
        return (name, os.environ.get("JDOCMUNCH_ST_MODEL", _SentenceTransformersProvider.DEFAULT_MODEL))
    if name == "gemini":
        return (name, _GeminiProvider.MODEL, os.environ.get("GOOGLE_API_KEY", "")[:8])
    if name == "openai":
        return (name, _OpenAIProvider.MODEL, os.environ.get("OPENAI_API_KEY", "")[:8])
    if name == "openai-compatible":
        return (
            name,
            _openai_compat_url(),
            _openai_compat_model(),
            _openai_compat_api_key()[:8],
            _openai_compat_batch_size(_OpenAICompatibleProvider.BATCH_SIZE),
        )
    return (name,)


def _reset_provider_cache() -> None:
    """Test hook — clears the provider cache."""
    _PROVIDER_CACHE.clear()


def _get_provider():
    name = get_provider_name()
    if not name:
        return None
    factory = _PROVIDER_FACTORIES.get(name)
    if not factory:
        return None
    key = _provider_signature(name)
    cached = _PROVIDER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        instance = factory()
    except Exception:
        return None
    _PROVIDER_CACHE[key] = instance
    return instance


def _provider_identity(name: str) -> tuple[str, Optional[int]]:
    """Return ``(model_name, dim)`` for the active provider.

    Used by the embedding cache to validate the sidecar's identity header.
    Dim is best-effort: providers expose it as a class constant when known,
    otherwise None and the cache treats the dim slot as wildcard.
    """
    if name == "gemini":
        return (_GeminiProvider.MODEL, 768)
    if name == "openai":
        return (_OpenAIProvider.MODEL, 1536)
    if name == "openai-compatible":
        # Read dim from the cached provider instance (probed once at __init__).
        # Falls back to None when no instance is constructed yet — the cache
        # layer treats dim=None as a wildcard, preserving v1.66.0 behavior.
        inst = _PROVIDER_CACHE.get(_provider_signature(name))
        dim = getattr(inst, "dim", None) if inst is not None else None
        return (f"{_openai_compat_url()}::{_openai_compat_model()}", dim)
    if name == "sentence-transformers":
        return (
            os.environ.get("JDOCMUNCH_ST_MODEL", _SentenceTransformersProvider.DEFAULT_MODEL),
            None,
        )
    return (name, None)


def embed_sections(
    sections: list,
    *,
    owner: Optional[str] = None,
    name: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> list:
    """Generate and attach embeddings to sections in-place.

    When ``owner`` and ``name`` are supplied, looks up cached vectors keyed
    by ``content_hash`` from ``~/.doc-index/<owner>/<name>.embeddings.jsonl``.
    Only cache misses are sent to the provider — typical incremental
    re-indexes touch <10% of sections, so cache hit-rate dominates cost.

    Cache header records (provider, model, dim); a mismatch on load
    purges the file and forces a full re-embed.

    Silently degrades to no-embeddings when no provider is configured.
    Backward-compatible with the v1.0–v1.14 signature
    ``embed_sections(sections)`` — caching is opt-in via owner+name.
    """
    provider = _get_provider()
    if not provider:
        return sections

    provider_name = get_provider_name() or ""
    model, dim = _provider_identity(provider_name)

    cache_enabled = bool(owner and name)
    if cache_enabled:
        from . import cache as _cache  # local import to avoid circulars
        cached = _cache.load(
            storage_path, owner, name,
            provider=provider_name, model=model, dim=dim,
        )
    else:
        cached = {}

    # First pass: split sections into cache-hits and misses.
    misses: list = []
    miss_indices: list[int] = []
    for i, sec in enumerate(sections):
        k = _embed_cache_key(sec)
        vec = cached.get(k) if k else None
        if vec:
            sec.embedding = vec
        else:
            misses.append(sec)
            miss_indices.append(i)

    # Second pass: embed misses in one provider batch.
    if misses:
        texts = [_section_embed_text(s) for s in misses]
        try:
            embeddings = provider.embed_texts(texts, task_type="retrieval_document")
            for sec, emb in zip(misses, embeddings):
                if emb:
                    sec.embedding = emb
        except Exception:
            pass  # lexical search still works

    # Rewrite cache when enabled — gathers all current (hash, vector) pairs.
    if cache_enabled:
        from . import cache as _cache
        entries = []
        for sec in sections:
            k = _embed_cache_key(sec)
            vec = getattr(sec, "embedding", None)
            if k and vec:
                entries.append((k, list(vec)))
        if entries:
            try:
                _cache.write(
                    storage_path, owner, name,
                    provider=provider_name, model=model, dim=dim,
                    entries=entries,
                )
            except Exception:
                pass

    return sections


def warmup() -> str:
    """Force-load the active embedding provider so its first call is hot.

    Returns the provider name that was warmed, or empty string if nothing
    was warmed (no provider configured, warmup not needed for this provider,
    or warmup failed).

    Only providers with significant first-call latency get warmed.
    sentence-transformers lazy-loads a local model on first embed_query,
    which (a) can hang past the MCP client's tool-call timeout, and
    (b) can write progress chatter to stdout, corrupting MCP JSON-RPC
    framing if it happens after stdio_server takes over.

    Network providers (gemini, openai, openai-compatible) are first-call-fast
    enough that warmup is unnecessary; warming them would add an avoidable
    network round-trip to startup.
    """
    name = get_provider_name()
    if name != "sentence-transformers":
        return ""
    try:
        embed_query("jdocmunch warmup")
        return name
    except Exception:
        return ""


def should_embed(flag) -> bool:
    """Resolve a use_embeddings flag (bool, 'auto', or string boolean) to a concrete bool.

    'auto' → True when an embedding provider is configured, else False.

    Recognises common string booleans (case-insensitive, whitespace-trimmed):
    'true'/'false', '1'/'0', 'yes'/'no', 'on'/'off', 't'/'f', 'y'/'n'.
    Unknown strings fall through to bool(flag) so previously-truthy strings
    keep their behavior (1.x compat).
    """
    if isinstance(flag, str):
        s = flag.strip().lower()
        if s == "auto":
            return get_provider_name() is not None
        if s in ("true", "1", "yes", "on", "y", "t"):
            return True
        if s in ("false", "0", "no", "off", "n", "f", ""):
            return False
    return bool(flag)


# ---------------------------------------------------------------------------
# Query-embedding cache (v1.13.0)
#
# The same query gets re-embedded across hybrid + semantic_only retries within
# one search, and across consecutive paginated calls. A small TTL'd LRU keeps
# the second hit free. Keyed by (provider_signature, query) — provider rotates
# implicitly invalidate when get_provider_name() changes (the cache key looks
# up the live provider's signature).
# ---------------------------------------------------------------------------

_QUERY_CACHE: "OrderedDict[tuple, tuple[float, list]]" = None  # type: ignore[assignment]
_QUERY_CACHE_MAXSIZE = 256
_QUERY_CACHE_TTL_SECONDS = 300.0  # 5 minutes


def _query_cache() -> "OrderedDict":
    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        from collections import OrderedDict
        _QUERY_CACHE = OrderedDict()
    return _QUERY_CACHE


def _reset_query_cache() -> None:
    """Test hook — clears the query embedding cache."""
    cache = _query_cache()
    cache.clear()


def embed_query(query: str) -> Optional[list]:
    """Embed a search query. Returns None if no provider is configured.

    Caches by (provider_signature, query) for ``_QUERY_CACHE_TTL_SECONDS``.
    Provider rotation invalidates implicitly via the signature key.
    """
    import time as _time

    name = get_provider_name()
    if not name:
        return None
    sig = _provider_signature(name)
    key = (sig, query)
    cache = _query_cache()
    now = _time.time()

    cached = cache.get(key)
    if cached is not None:
        ts, vec = cached
        if now - ts < _QUERY_CACHE_TTL_SECONDS:
            cache.move_to_end(key)
            return vec
        # Stale — drop and refetch.
        del cache[key]

    provider = _get_provider()
    if not provider:
        return None
    try:
        results = provider.embed_texts([query], task_type="retrieval_query")
        vec = results[0] if results and results[0] else None
    except Exception:
        return None
    if vec is None:
        return None

    cache[key] = (now, vec)
    cache.move_to_end(key)
    while len(cache) > _QUERY_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return vec
