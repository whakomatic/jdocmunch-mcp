"""jdoc#118 phase 1 — sentence-transformers in a child process.

⚠ These go through REAL subprocesses. The whole mechanism is "the import
happens in another process", and an in-process test of that tests the mock
(the jdoc#129 lesson, which was learned on the fd swap and applies verbatim
here).

The protocol tests drive a *fake* child written to ``tmp_path``: it speaks the
same wire format with no sentence-transformers anywhere, so they run on a
machine with no embedding stack, with a broken one, or on CI. The real child
module is exercised separately, on the path that does not need a working model.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from jdocmunch_mcp.embeddings import provider as prov
from jdocmunch_mcp.embeddings import worker as w


# ---------------------------------------------------------------------------
# Fake children
# ---------------------------------------------------------------------------

_FAKE_CHILD = '''
import json, sys

def send(msg):
    sys.stdout.write(json.dumps(msg) + "\\n")
    sys.stdout.flush()

MODE = {mode!r}
DIM = 4
ready_seen = 0

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    op = msg.get("op")
    if op == "shutdown":
        sys.exit(0)
    if op == "ready":
        ready_seen += 1
        if MODE == "die_on_ready":
            sys.exit(3)
        if MODE == "silent":
            # Never answer. The parent must bound this itself.
            for _ in sys.stdin:
                pass
            sys.exit(0)
        if MODE == "broken_model":
            send({{"op": "ready", "ok": False, "error": "ImportError: no HybridCache"}})
            continue
        send({{"op": "ready", "ok": True, "error": "", "dim": DIM, "stdout_private": True}})
        continue
    if op == "embed":
        texts = msg.get("texts") or []
        if MODE == "short_answer":
            rows = texts[:-1]
        else:
            rows = texts
        vecs = [[float(len(t)), float(sum(t.encode("utf-8")) % 97), 0.25, -1.5] for t in rows]
        payload, dim = ("", 0)
        if vecs:
            import base64
            from array import array
            flat = array("f")
            for row in vecs:
                flat.extend(row)
            if sys.byteorder != "little":
                flat.byteswap()
            payload, dim = base64.b64encode(flat.tobytes()).decode("ascii"), DIM
        send({{"id": msg.get("id"), "ok": True, "dim": dim, "vecs": payload}})
        continue
'''


def _fake_child(tmp_path, mode="ok"):
    path = tmp_path / f"fake_child_{mode}.py"
    path.write_text(_FAKE_CHILD.format(mode=mode), encoding="utf-8")
    return [sys.executable, str(path)]


def _expected(text):
    return [float(len(text)), float(sum(text.encode("utf-8")) % 97), 0.25, -1.5]


@pytest.fixture
def fake_worker(tmp_path):
    """Build a WorkerProvider against a fake child; always tear the child down."""
    built = []

    def _build(mode="ok", **kwargs):
        provider = w.WorkerProvider("fake-model", command=_fake_child(tmp_path, mode), **kwargs)
        built.append(provider)
        return provider

    yield _build
    for provider in built:
        provider.close()


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

class TestWireFormat:
    def test_roundtrip_preserves_values_and_shape(self):
        rows = [[0.5, -1.5, 2.25], [1.0, 0.0, -0.125]]
        payload, dim = w.encode_vectors(rows)
        assert dim == 3
        assert w.decode_vectors(payload, dim) == rows

    def test_empty_rows_encode_to_nothing_and_decode_back(self):
        payload, dim = w.encode_vectors([])
        assert (payload, dim) == ("", 0)
        assert w.decode_vectors("", 0) == []

    def test_ragged_rows_raise_rather_than_silently_truncating(self):
        # jdoc#109: a width mismatch that returns a plausible number is worse
        # than one that raises. cosine_similarity's zip() truncated a 768-dim
        # query against a 384-dim vector and returned a confident 0.707.
        with pytest.raises(ValueError):
            w.encode_vectors([[1.0, 2.0], [3.0]])

    def test_base64_is_smaller_than_json_floats(self):
        # The reason the format is not JSON floats: a 5,300-section corpus at
        # 384 dims is ~30 MB as text.
        rows = [[0.123456789] * 384 for _ in range(64)]
        payload, _ = w.encode_vectors(rows)
        assert len(payload) < len(json.dumps(rows)) / 2


# ---------------------------------------------------------------------------
# Enablement — phase 1 changes no default
# ---------------------------------------------------------------------------

class TestEnablement:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("JDOCMUNCH_EMBED_WORKER", raising=False)
        monkeypatch.delenv("JDOCMUNCH_PRELOAD_EMBEDDINGS", raising=False)

    def test_on_when_unset(self):
        # ⚠⚠ The default IS the fix. v1.132.0 left the user choosing between a
        # probabilistic hang and a cold-start failure, and #118 stayed open
        # because that choice is the defect restated as configuration. An
        # opt-in worker would have been a third switch on the same pile.
        assert w.worker_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_on_for_affirmative_values(self, monkeypatch, value):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", value)
        assert w.worker_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_off_when_explicitly_disabled(self, monkeypatch, value):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", value)
        assert w.worker_enabled() is False

    @pytest.mark.parametrize("value", ["", "maybe"])
    def test_an_unrecognised_value_falls_through_to_the_default(self, monkeypatch, value):
        # A typo must not silently disable the fix.
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", value)
        assert w.worker_enabled() is True

    def test_the_1_132_0_preload_flag_turns_the_worker_off(self, monkeypatch):
        # ⚠ That user explicitly chose the main-thread import and is entitled
        # to keep getting it. Running both would import the stack into the very
        # process the worker exists to keep clean.
        monkeypatch.setenv("JDOCMUNCH_PRELOAD_EMBEDDINGS", "1")
        assert w.worker_enabled() is False

    def test_naming_the_worker_beats_the_preload_flag(self, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_PRELOAD_EMBEDDINGS", "1")
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "1")
        assert w.worker_enabled() is True

    def test_garbage_timeout_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER_TIMEOUT", "soon")
        assert w.request_timeout() == w.REQUEST_TIMEOUT_DEFAULT
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER_TIMEOUT", "-4")
        assert w.request_timeout() == w.REQUEST_TIMEOUT_DEFAULT
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER_TIMEOUT", "12.5")
        assert w.request_timeout() == 12.5


# ---------------------------------------------------------------------------
# Real subprocess round trips
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_vectors_come_back_through_a_real_pipe(self, fake_worker):
        provider = fake_worker()
        texts = ["alpha", "beta gamma", "d"]
        vectors = provider.embed_texts(texts)
        assert vectors == [_expected(t) for t in texts]

    def test_no_texts_never_spawns_a_request(self, fake_worker):
        provider = fake_worker()
        assert provider.embed_texts([]) == []

    def test_a_corpus_larger_than_one_chunk_returns_in_order(self, fake_worker):
        # embed_sections hands over EVERY cache miss in one call, so chunking
        # is what makes the per-request timeout mean anything.
        provider = fake_worker()
        texts = [f"section-{i}" for i in range(w.CHUNK_SIZE * 2 + 5)]
        vectors = provider.embed_texts(texts)
        assert len(vectors) == len(texts)
        assert vectors == [_expected(t) for t in texts]

    def test_construction_does_not_wait_for_the_child(self, fake_worker):
        # ⚠ The whole gain is lost if building the provider blocks on the
        # import — that is jdoc#110's slow handshake in a different hat.
        provider = fake_worker("silent")
        assert provider._ready is None

    def test_a_process_that_never_calls_close_still_exits_clean(self, tmp_path):
        """atexit must reap the child, and the pipes must be closed.

        ⚠ Not a tidiness point. The PostToolUse reindex hook is a whole process
        per edited file, so "the caller forgot to close" is the common case,
        not the exceptional one. `-W error::ResourceWarning` turns an unclosed
        pipe into a non-zero exit.
        """
        script = tmp_path / "no_close.py"
        script.write_text(
            "import sys\n"
            "from jdocmunch_mcp.embeddings import worker as w\n"
            "p = w.WorkerProvider('fake-model', command=[sys.executable, sys.argv[1]])\n"
            "assert len(p.embed_texts(['x'])) == 1\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-W", "error::ResourceWarning", str(script),
             _fake_child(tmp_path, "ok")[1]],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert "ResourceWarning" not in proc.stderr

    def test_close_stops_the_child(self, fake_worker):
        provider = fake_worker()
        provider.embed_texts(["x"])
        proc = provider._proc
        provider.close()
        assert proc.poll() is not None


# ---------------------------------------------------------------------------
# Failure handling — the killable property is the point
# ---------------------------------------------------------------------------

class TestFailures:
    def test_a_silent_child_is_bounded_and_then_killed(self, fake_worker, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER_READY_TIMEOUT", "2")
        provider = fake_worker("silent")
        with pytest.raises(w.EmbedWorkerError) as excinfo:
            provider.embed_texts(["anything"])
        assert "timed out" in str(excinfo.value)
        # ⚠⚠ This is the assertion that separates a fix from a relocation. A
        # thread wedged in LdrLoadDll is a kernel-mode wait that no timeout,
        # thread kill or try/except can touch. A process can just be killed.
        assert provider._proc is None

    def test_a_timeout_raises_rather_than_returning_empty_vectors(self, fake_worker, monkeypatch):
        # embed_sections reads an exception as embed_failed and PRESERVES the
        # sidecar; a list of empty vectors reads as "this corpus has none",
        # which is the jdoc#107/#109 data-loss shape.
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER_READY_TIMEOUT", "2")
        provider = fake_worker("silent")
        with pytest.raises(w.EmbedWorkerError):
            provider.embed_texts(["anything"])

    def test_a_model_that_cannot_load_is_terminal_not_retried(self, fake_worker):
        # An import that RAISES is a broken install, not a race. Respawning
        # would fail identically and cost another cold import.
        provider = fake_worker("broken_model")
        with pytest.raises(w.EmbedWorkerError) as first:
            provider.embed_texts(["x"])
        assert "HybridCache" in str(first.value)
        spawns = provider._spawns
        with pytest.raises(w.EmbedWorkerError):
            provider.embed_texts(["x"])
        assert provider._spawns == spawns, "a broken model must not be respawned"

    def test_a_dying_child_is_respawned_once_and_then_given_up_on(self, fake_worker):
        provider = fake_worker("die_on_ready")
        for _ in range(w.WorkerProvider.MAX_SPAWNS):
            with pytest.raises(w.EmbedWorkerError):
                provider.embed_texts(["x"])
        assert provider._spawns == w.WorkerProvider.MAX_SPAWNS
        with pytest.raises(w.EmbedWorkerError) as final:
            provider.embed_texts(["x"])
        assert "not restarting" in str(final.value)
        assert provider._spawns == w.WorkerProvider.MAX_SPAWNS

    def test_a_short_answer_is_refused_rather_than_zipped_away(self, fake_worker):
        # embed_sections zips sections against the returned vectors, so a
        # short reply would silently shift every vector onto the wrong section.
        provider = fake_worker("short_answer")
        with pytest.raises(w.EmbedWorkerError) as excinfo:
            provider.embed_texts(["a", "bb", "ccc"])
        assert "2 vectors for 3 texts" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The real child module
# ---------------------------------------------------------------------------

class TestRealChild:
    def test_it_reports_a_load_failure_instead_of_dying_silently(self):
        # Runs anywhere: the model name cannot resolve, and with no
        # sentence-transformers installed the import raises first. Either way
        # the parent must get an answer — a child that dies quietly looks
        # exactly like the hang this design replaces.
        proc = subprocess.run(
            [sys.executable, "-m", "jdocmunch_mcp.embeddings.worker",
             "--model", "jdoc118-definitely-not-a-real-model"],
            input=json.dumps({"op": "ready"}) + "\n",
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        assert lines, f"the child said nothing (stderr: {proc.stderr[-500:]})"
        reply = json.loads(lines[-1])
        assert reply["op"] == "ready"
        assert reply["ok"] is False
        assert reply["error"]

    def test_it_exits_cleanly_on_shutdown(self):
        proc = subprocess.run(
            [sys.executable, "-m", "jdocmunch_mcp.embeddings.worker"],
            input=json.dumps({"op": "shutdown"}) + "\n",
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )
        assert proc.returncode == 0


# ---------------------------------------------------------------------------
# ⚠⚠ The mechanism guard — see the design's §6
# ---------------------------------------------------------------------------

_MODULES_PROBE = textwrap.dedent(
    """
    import json, sys
    from jdocmunch_mcp.embeddings import worker as w

    provider = w.WorkerProvider("fake-model", command=[sys.executable, sys.argv[1]])
    vectors = provider.embed_texts(["alpha", "beta"])
    provider.close()

    print(json.dumps({
        "vectors": len(vectors),
        "loaded": sorted(
            m for m in ("sentence_transformers", "transformers", "torch", "scipy", "sklearn")
            if m in sys.modules
        ),
    }))
    """
)


def test_the_server_process_never_imports_the_embedding_stack(tmp_path):
    """The property that closes jdoc#118 without reproducing the race.

    ⚠⚠ There is no A/B for this bug: the wedge stopped reproducing in BOTH
    arms after ~30 server starts, and the machine will not go cold on demand.
    This asserts something stronger and deterministic instead — the deadlock is
    *unreachable*, because the process that used to wedge no longer loads any
    of the libraries that wedge it. Reachability beats "it did not happen this
    time".
    """
    probe = tmp_path / "modules_probe.py"
    probe.write_text(_MODULES_PROBE, encoding="utf-8")
    child = _fake_child(tmp_path, "ok")[1]

    proc = subprocess.run(
        [sys.executable, str(probe), child],
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["vectors"] == 2, "the probe must actually embed, or it proves nothing"
    assert result["loaded"] == [], (
        "the parent process imported the embedding stack: "
        f"{result['loaded']} — jdoc#118's deadlock is reachable again"
    )


# ---------------------------------------------------------------------------
# Wiring into provider selection
# ---------------------------------------------------------------------------

class TestProviderWiring:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        prov._reset_provider_cache()
        monkeypatch.setenv("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")
        # A named provider resolves only when its package is importable, and
        # CI does not install sentence-transformers. These tests are about the
        # probe and worker wiring, not installation.
        monkeypatch.setattr(prov, "_provider_package_available", lambda name: True)
        yield
        prov._reset_provider_cache()

    def test_the_worker_is_used_by_default(self, monkeypatch):
        monkeypatch.delenv("JDOCMUNCH_EMBED_WORKER", raising=False)
        monkeypatch.delenv("JDOCMUNCH_PRELOAD_EMBEDDINGS", raising=False)
        seen = {}

        class _Stub:
            spawn_failed = False

            def __init__(self, model_name):
                seen["model"] = model_name

        monkeypatch.setattr(w, "WorkerProvider", _Stub)
        monkeypatch.setenv("JDOCMUNCH_ST_MODEL", "some-model")
        assert isinstance(prov._sentence_transformers_factory(), _Stub)
        assert seen["model"] == "some-model"

    def test_the_in_process_provider_is_used_when_the_worker_is_disabled(self, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "0")

        class _InProcess:
            DEFAULT_MODEL = "some-model"

        monkeypatch.setattr(prov, "_SentenceTransformersProvider", _InProcess)
        assert isinstance(prov._sentence_transformers_factory(), _InProcess)

    def test_a_child_that_cannot_spawn_falls_back_in_process(self, monkeypatch):
        """⚠⚠ The guard that makes defaulting the worker ON safe.

        Without it, anyone whose ``sys.executable`` cannot be spawned — a
        frozen bundle, a locked-down sandbox — silently loses semantic search
        that works for them today. That is a NEW defect traded for jdoc#118's,
        which is not a trade to make on a user's behalf.
        """
        monkeypatch.delenv("JDOCMUNCH_EMBED_WORKER", raising=False)
        monkeypatch.setenv("JDOCMUNCH_ST_MODEL", "some-model")

        class _Unspawnable:
            spawn_failed = True

            def __init__(self, model_name):
                pass

        class _InProcess:
            # ⚠ `_st_model_name` reads DEFAULT_MODEL off this class eagerly,
            # even when JDOCMUNCH_ST_MODEL is set, so a bare lambda stub blows
            # up before the code under test runs.
            DEFAULT_MODEL = "some-model"

        monkeypatch.setattr(w, "WorkerProvider", _Unspawnable)
        monkeypatch.setattr(prov, "_SentenceTransformersProvider", _InProcess)
        assert isinstance(prov._sentence_transformers_factory(), _InProcess)

    def test_a_real_unspawnable_command_sets_spawn_failed(self, tmp_path):
        # Through the real Popen, not a stub: the flag is only worth anything
        # if an actual spawn failure sets it.
        provider = w.WorkerProvider(
            "fake-model", command=[str(tmp_path / "no-such-interpreter")],
        )
        assert provider.spawn_failed is True

    def test_a_child_that_dies_later_does_not_fall_back_in_process(self, tmp_path):
        """A restart failure must NOT import the stack into a running server.

        By then the child HAS run, so the machine can spawn; something else
        broke. Recovering semantic search by importing in-process would trade a
        degraded feature for the deadlock this module exists to prevent.
        """
        provider = w.WorkerProvider("fake-model", command=_fake_child(tmp_path, "die_on_ready"))
        try:
            with pytest.raises(w.EmbedWorkerError):
                provider.embed_texts(["x"])
            assert provider.spawn_failed is False
        finally:
            provider.close()

    def test_the_import_probe_is_skipped_when_the_worker_owns_the_import(self, monkeypatch):
        # The probe protects THIS process from an import it no longer performs.
        # Running it anyway is a subprocess of pure cost.
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "1")
        called = []
        monkeypatch.setattr(
            prov, "_sentence_transformers_imports_cleanly",
            lambda: called.append(True) or True,
        )
        sentinel = object()
        monkeypatch.setitem(prov._PROVIDER_FACTORIES, "sentence-transformers", lambda: sentinel)
        assert prov._get_provider() is sentinel
        assert called == []

    def test_the_import_probe_still_runs_without_the_worker(self, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "0")
        called = []
        monkeypatch.setattr(
            prov, "_sentence_transformers_imports_cleanly",
            lambda: called.append(True) or True,
        )
        sentinel = object()
        monkeypatch.setitem(prov._PROVIDER_FACTORIES, "sentence-transformers", lambda: sentinel)
        assert prov._get_provider() is sentinel
        assert called == [True]

    def test_warmup_skips_the_probe_when_the_worker_owns_the_import(self, monkeypatch):
        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "1")
        monkeypatch.delenv("JDOCMUNCH_EMBED_WARMUP", raising=False)
        called = []
        monkeypatch.setattr(
            prov, "_sentence_transformers_imports_cleanly",
            lambda: called.append(True) or False,
        )
        monkeypatch.setattr(prov, "_st_model_is_cached", lambda _model: True)
        monkeypatch.setattr(prov, "embed_query", lambda _q: [0.0])
        assert prov.warmup() == "sentence-transformers"
        assert called == []

    def test_the_preload_declines_when_the_worker_owns_the_import(self, monkeypatch):
        # ⚠⚠ Preloading here would import the stack into the very process the
        # worker keeps clean — paying the slow handshake and getting nothing.
        from jdocmunch_mcp import preload

        monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "1")
        monkeypatch.setenv("JDOCMUNCH_PRELOAD_EMBEDDINGS", "1")
        report = preload.preload_embedding_stack()
        assert report == {"sentence_transformers": "absent: embedding worker owns the import"}


def test_the_identity_header_does_not_change_when_the_worker_is_enabled(monkeypatch):
    """⚠⚠ Flipping dim from None to 384 would re-embed every existing corpus.

    ``_provider_identity`` returns None for sentence-transformers and the cache
    treats that as a wildcard. Filling it in from what the child reports would
    make ``identity_matches`` fail on every sidecar written before this change
    — jdoc#109's escalation, triggered by a refactor rather than a rotation.
    """
    monkeypatch.setenv("JDOCMUNCH_ST_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "0")
    without = prov._provider_identity("sentence-transformers")
    signature_without = prov._provider_signature("sentence-transformers")
    monkeypatch.setenv("JDOCMUNCH_EMBED_WORKER", "1")
    assert prov._provider_identity("sentence-transformers") == without
    assert prov._provider_signature("sentence-transformers") == signature_without
    assert without[1] is None
