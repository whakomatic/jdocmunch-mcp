"""An incremental embed pass must not delete the vectors it was not given.

``embed_sections`` ends by rewriting the WHOLE sidecar from the section list it
was handed, and ``index_local``'s incremental branch hands it the CHANGED FILES
ONLY, so every incremental reindex deleted the vectors of every file it did not
touch. Measured 2026-08-25 on a three-file corpus: a full pass wrote 9 vectors,
editing one file and reindexing left 3, while the response still reported
``semantic_search: true``. On the machine it was found on, this had reduced two
real indexes to 2 embedded sections out of 420 and 790, which is what the
standing "24% embedded" reading actually was: a steady state, re-truncated at
every session start, not a half-finished build.
"""

from types import SimpleNamespace

import pytest

from jdocmunch_mcp.embeddings import cache as cache_mod
from jdocmunch_mcp.embeddings import provider as provider_mod


class _StubProvider:
    """Returns a distinct constant vector per call batch."""

    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts, task_type=None):
        self.calls += 1
        return [[float(self.calls), 0.5, 0.25] for _ in texts]


def _section(content_hash: str):
    return SimpleNamespace(content_hash=content_hash, title="t", summary="s",
                           doc_path="d.md", embedding=None)


@pytest.fixture
def stub_provider(monkeypatch):
    p = _StubProvider()
    monkeypatch.setattr(provider_mod, "_get_provider", lambda: p)
    monkeypatch.setattr(provider_mod, "get_provider_name", lambda: "stub")
    monkeypatch.setattr(provider_mod, "_provider_identity", lambda name: ("stub-model", 3))
    monkeypatch.setattr(provider_mod, "_section_embed_text", lambda s: "text")
    return p


def _cached_hashes(store_path, owner="local", name="idx"):
    return set(cache_mod.load(str(store_path), owner, name,
                              provider="stub", model="stub-model", dim=3))


def test_incremental_pass_keeps_the_vectors_of_untouched_sections(stub_provider, tmp_path):
    provider_mod.embed_sections(
        [_section("hash-a"), _section("hash-b")],
        owner="local", name="idx", storage_path=str(tmp_path),
    )
    assert len(_cached_hashes(tmp_path)) == 2

    provider_mod.embed_sections(
        [_section("hash-a")],
        owner="local", name="idx", storage_path=str(tmp_path),
        prune=False,
    )
    assert _cached_hashes(tmp_path) == {"hash-a#pv1", "hash-b#pv1"}


def test_a_full_pass_still_prunes_a_section_that_is_gone(stub_provider, tmp_path):
    """The total rewrite is CORRECT when the caller has the whole corpus, and
    must stay: it is what keeps a deleted document's vector from living forever."""
    provider_mod.embed_sections(
        [_section("hash-a"), _section("hash-b")],
        owner="local", name="idx", storage_path=str(tmp_path),
        prune=True,
    )
    provider_mod.embed_sections(
        [_section("hash-a")],
        owner="local", name="idx", storage_path=str(tmp_path),
        prune=True,
    )
    assert _cached_hashes(tmp_path) == {"hash-a#pv1"}


def test_an_incremental_pass_after_a_provider_change_still_discards(tmp_path, monkeypatch):
    """Merging must never mix vector spaces: a different model's vectors are
    dropped even on an incremental pass, because the merge source is the cache
    LOAD, which already refuses a foreign identity header."""
    cache_mod.write(str(tmp_path), "local", "idx",
                    provider="other", model="other-model", dim=7,
                    entries=[("hash-old#pv1", [9.0, 9.0, 9.0])])
    p = _StubProvider()
    monkeypatch.setattr(provider_mod, "_get_provider", lambda: p)
    monkeypatch.setattr(provider_mod, "get_provider_name", lambda: "stub")
    monkeypatch.setattr(provider_mod, "_provider_identity", lambda name: ("stub-model", 3))
    monkeypatch.setattr(provider_mod, "_section_embed_text", lambda s: "text")

    provider_mod.embed_sections(
        [_section("hash-a")],
        owner="local", name="idx", storage_path=str(tmp_path),
        prune=False,
    )
    assert _cached_hashes(tmp_path) == {"hash-a#pv1"}


def test_a_cache_hit_is_not_re_embedded(stub_provider, tmp_path):
    """Guards the reason the cache exists at all, so a merge cannot be 'fixed'
    by simply re-embedding everything every time."""
    provider_mod.embed_sections([_section("hash-a")], owner="local", name="idx",
                                storage_path=str(tmp_path))
    calls_after_first = stub_provider.calls
    provider_mod.embed_sections([_section("hash-a")], owner="local", name="idx",
                                storage_path=str(tmp_path), prune=False)
    assert stub_provider.calls == calls_after_first
