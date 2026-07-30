"""A bare cloud API key must never auto-enable paid embedding.

An ambient `OPENAI_API_KEY` made `index_local`'s default `use_embeddings="auto"`
select OpenAI and start embedding the corpus. Two costs, and the second is the
one that matters: it bills a remote account per call, and it sends the indexed
text off the machine. It was found by pointing `index_local` at a PRIVATE
memory store and watching it run for 25 minutes.

⚠ The summarizer path already had exactly this guard
(`summarizer/batch_summarize._PAID_CLOUD_PROVIDERS` + `JDOCMUNCH_ALLOW_PAID_SUMMARIES`),
added for the same reason and carrying a comment naming the same stray-key
scenario. It was never ported to the embeddings path, so AI summaries were
correctly suppressed while embeddings sailed through the identical hazard. The
lesson is the one in the jcm#392 guard: a protection applied to one of two paths
that share a hazard is a protection with a hole in it.

Naming the provider explicitly is always honored. The guard is aimed at
*ambient* configuration, not deliberate choices.
"""

from __future__ import annotations

import importlib

import pytest

import jdocmunch_mcp.embeddings.provider as provider


_CLOUD_ENV = (
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "JDOCMUNCH_EMBEDDING_PROVIDER",
    "JDOCMUNCH_ALLOW_PAID_EMBEDDINGS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No ambient embedding configuration, and no cross-test warn-once bleed."""
    for var in _CLOUD_ENV:
        monkeypatch.delenv(var, raising=False)
    provider._WARNED_SUPPRESSED_PAID_EMBED.clear()
    provider._WARNED_MISSING_PACKAGE.clear()
    yield monkeypatch
    provider._WARNED_SUPPRESSED_PAID_EMBED.clear()
    provider._WARNED_MISSING_PACKAGE.clear()


@pytest.fixture
def packages_installed(monkeypatch):
    """Pretend every provider's backing package is importable.

    These tests are about the PAID-CLOUD GATE, not installation. Under the
    package-availability guard (a name is only returned when it can embed), a
    selected-but-uninstalled provider resolves to None for its own reason,
    which would shadow the gate decision under test — and the dev env installs
    none of the cloud SDKs.
    """
    monkeypatch.setattr(provider, "_provider_package_available", lambda name: True)


@pytest.fixture
def no_local(monkeypatch):
    """Force the offline fallback unavailable so the cloud decision is visible.

    With sentence-transformers installed the guard falls through to it, which is
    the desired product behaviour but hides which branch was taken.
    """
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: False)


@pytest.mark.parametrize("env_var,name", [("OPENAI_API_KEY", "openai"), ("GOOGLE_API_KEY", "gemini")])
def test_bare_cloud_key_does_not_auto_select(clean_env, no_local, env_var, name):
    """The defect, exactly. Fails on revert: returns the provider name."""
    clean_env.setenv(env_var, "sk-not-a-real-key")
    assert provider.get_provider_name() is None, (
        f"a bare {env_var} auto-selected {name}: that bills a remote account and "
        "sends the indexed corpus to a third party with no opt-in"
    )


@pytest.mark.parametrize("env_var,name", [("OPENAI_API_KEY", "openai"), ("GOOGLE_API_KEY", "gemini")])
def test_explicit_opt_in_restores_the_provider(clean_env, no_local, packages_installed, env_var, name):
    """The guard must be an opt-in, not a removal. Non-vacuity for the test above:
    if this fails, the provider is simply broken rather than gated."""
    clean_env.setenv(env_var, "sk-not-a-real-key")
    clean_env.setenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "1")
    assert provider.get_provider_name() == name


@pytest.mark.parametrize("name", ["openai", "gemini"])
def test_naming_the_provider_is_always_honored(clean_env, no_local, packages_installed, name):
    """Explicit is a deliberate choice and bypasses the guard, matching the
    summarizer's contract. Without this the guard would break real users who
    configured a cloud embedder on purpose."""
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", name)
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    clean_env.setenv("GOOGLE_API_KEY", "not-a-real-key")
    assert provider.get_provider_name() == name


def test_should_embed_auto_follows_the_guard(clean_env, no_local, packages_installed):
    """`use_embeddings="auto"` is index_local's DEFAULT, so this is the path the
    incident actually took. Guarding get_provider_name is only a fix because
    should_embed('auto') is defined in terms of it."""
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert provider.should_embed("auto") is False
    clean_env.setenv("JDOCMUNCH_ALLOW_PAID_EMBEDDINGS", "1")
    assert provider.should_embed("auto") is True


def test_offline_provider_still_wins_when_available(clean_env, monkeypatch):
    """Suppressing paid cloud must not disable embedding outright when a local
    provider exists. Semantic search keeps working, on-machine."""
    monkeypatch.setattr(provider, "_sentence_transformers_available", lambda: True)
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert provider.get_provider_name() == "sentence-transformers"


def test_suppression_is_logged_not_silent(clean_env, no_local, caplog):
    """A capability that vanishes without saying so is its own bug. The user
    must be able to find out why semantic search went quiet."""
    clean_env.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    with caplog.at_level("WARNING"):
        provider.get_provider_name()
    assert any("opt-in" in r.message or "opt-in" in r.getMessage() for r in caplog.records), (
        f"expected a warning naming the opt-in; got {[r.getMessage() for r in caplog.records]}"
    )


def test_openai_compatible_is_not_gated(clean_env, no_local, packages_installed):
    """openai-compatible requires an explicitly configured URL + model, which is
    itself the opt-in, and the common target is a local runtime. Gating it would
    punish the offline-friendly option."""
    clean_env.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "openai-compatible")
    clean_env.setenv("JDOCMUNCH_OPENAI_COMPAT_URL", "http://localhost:11434/v1")
    clean_env.setenv("JDOCMUNCH_OPENAI_COMPAT_MODEL", "nomic-embed-text")
    assert provider.get_provider_name() == "openai-compatible"


def test_guard_module_has_a_logger(clean_env):
    """provider.py had NO module logger before this change, so the warning above
    would have raised NameError inside the guard and taken indexing down with it.
    Same trap recorded for doc_store.py (jcm v1.108.100 NameError-in-except)."""
    mod = importlib.import_module("jdocmunch_mcp.embeddings.provider")
    assert getattr(mod, "logger", None) is not None
