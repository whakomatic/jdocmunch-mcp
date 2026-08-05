#!/usr/bin/env python
"""Rebuild jdocmunch doc indexes WITH local (sentence-transformers) embeddings.

Zero billed API: summaries off, provider pinned to sentence-transformers.
Each rebuild is a full (--incremental false) pass so vectors are written even
for indexes that previously had none.

Status: prints a live heartbeat per repo (elapsed + which output file was last
written) since the underlying tool is silent during the embed + sidecar passes.
Rebuilds ALL repos unconditionally.

    python tmp/rebuild_embeddings.py

Roots verified against each index's own first doc_path. Edit REPOS if a path moves.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Smallest-first (by section count): fast repos confirm the pipeline and give
# quick [OK]s before the multi-minute big ones. aisimulator (~6.9k) and
# jdocmunch-mcp (~14k) are the long poles, so they run last.
# (name, root, ignore_patterns). ignore_patterns is gitignore-syntax passed as
# extra_ignore_patterns to index_local. jdocmunch-mcp and aisimulator index a
# huge amount of NON-doc JSON (benchmark results, fixtures, prototype/gamelog
# data) -- neither repo has real OpenAPI specs (verified), so we drop *.json
# from those two. This cuts jdocmunch-mcp ~14k->~500 and aisimulator ~6.9k->~5.4k
# sections and removes the related-graph O(n^2) blowup on junk.
_JSON_HEAVY = ["*.json", "benchmarks/", "tests/fixtures/"]
REPOS = [
    ("AshesofXCom",   r"E:/ModBuddy/AshesofXCom",       None),  # ~144 sections
    ("Recall",        r"D:/OneDrive/Projects/Recall",   None),  # ~765
    ("holdings",      r"D:/OneDrive/Projects/holdings", None),  # ~4070
    ("xcom2",         r"E:/xcom2/docs",                 None),  # ~6367
    ("aisimulator",   r"E:/xcom2/aisimulator",          _JSON_HEAVY),  # 6.9k->~5.4k
    ("jdocmunch-mcp", r"D:/OneDrive/Projects/jdocmunch-mcp", _JSON_HEAVY),  # 14k->~500
]
DELETE_FIRST = ["recall", "Recall"]

ENV = {**os.environ, "JDOCMUNCH_EMBEDDING_PROVIDER": "sentence-transformers"}
PER_REPO_TIMEOUT = 2400   # 40 min; big indexes + 15MB sidecars + save are slow
HEARTBEAT_SECS = 5
INDEX_DIR = Path.home() / ".doc-index" / "local"


def emb_path(name: str) -> Path:
    return INDEX_DIR / f"{name}.embeddings.jsonl"


def emb_lines(name: str) -> int:
    p = emb_path(name)
    if not p.is_file():
        return 0
    try:
        with p.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def run_tool_quiet(tool: str, args: list[str], timeout: int = 120) -> dict:
    proc = subprocess.run(
        ["jdocmunch-mcp", "run", tool, *args],
        capture_output=True, text=True, env=ENV, timeout=timeout,
    )
    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return {"_unparsed": proc.stdout.strip()[-300:]}


def index_with_status(name: str, root: str, ignore: list | None = None) -> dict:
    """Run index_local in a thread; heartbeat from the embeddings cache."""
    args = [
        "run", "index_local",
        "--path", root, "--name", name,
        "--use_ai_summaries", "false",
        "--use_embeddings", "true",
        "--incremental", "false",
    ]
    if ignore:
        # run-bridge JSON-parses values, so a JSON array reaches the tool as a
        # Python list for extra_ignore_patterns (gitignore syntax).
        args += ["--extra_ignore_patterns", json.dumps(ignore)]
    # Keep streams SEPARATE: stdout is the pretty-printed JSON result (one
    # document); stderr is the model "Loading weights" chatter. Merging them
    # interleaves chatter into the JSON and makes the result unparseable.
    proc = subprocess.Popen(
        ["jdocmunch-mcp", *args],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=ENV,
    )

    captured: list[str] = []

    def _drain():
        for line in proc.stdout:  # type: ignore[union-attr]
            captured.append(line)
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    # Two-signal heartbeat:
    #  - last_write: which output artifact was written most recently. The
    #    rebuild emits each in one atomic write at the END of its stage
    #    (embed -> terms -> duplicates -> related -> boilerplate -> json),
    #    so this shows stage TRANSITIONS, never intra-stage motion.
    #  - cpu(+Ns): CPU seconds the worker burned since the last tick. The
    #    indexer is silent on stdout/stderr during a stage, so this is the
    #    ONLY continuous "alive and computing" signal. cpu +~tick = working;
    #    cpu +0 with elapsed climbing = genuinely stuck.
    stages = ["embeddings.jsonl", "terms.json", "duplicates.json",
              "related.json", "boilerplate.json", "json"]

    def latest_stage() -> str:
        newest, newest_mt = "-", 0.0
        for s in stages:
            p = INDEX_DIR / (f"{name}.{s}" if s != "json" else f"{name}.json")
            if p.is_file() and p.stat().st_mtime > newest_mt:
                newest_mt, newest = p.stat().st_mtime, s
        return newest

    def worker_stats() -> tuple[float, float]:
        """(cpu_seconds, io_bytes) summed over our process tree (PID + kids).

        cpu = KernelModeTime+UserModeTime (100-ns ticks). io = Read+Write
        TransferCount: ALL bytes the process moves to/from any file or pipe,
        which is NOT the same as output-file size growth. The slow
        related-graph stage is pure in-RAM compute: output files stay flat AND
        process I/O stays flat -- only CPU moves. So CPU is the liveness signal;
        I/O just distinguishes read/write-bound phases from compute-bound ones.
        Both flat with elapsed climbing = genuinely stuck."""
        try:
            ps = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process | Select-Object ProcessId,"
                 "ParentProcessId,KernelModeTime,UserModeTime,ReadTransferCount,"
                 "WriteTransferCount | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=10,
            )
            rows = json.loads(ps.stdout or "[]")
            if isinstance(rows, dict):
                rows = [rows]
            by_parent: dict[int, list[dict]] = {}
            cpu_of: dict[int, float] = {}
            io_of: dict[int, float] = {}
            for r in rows:
                pid = int(r["ProcessId"])
                by_parent.setdefault(int(r["ParentProcessId"]), []).append(r)
                cpu_of[pid] = (int(r["KernelModeTime"]) + int(r["UserModeTime"])) / 1e7
                io_of[pid] = float(int(r["ReadTransferCount"]) + int(r["WriteTransferCount"]))
            cpu_total, io_total, stack, seen = 0.0, 0.0, [proc.pid], set()
            while stack:
                pid = stack.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                cpu_total += cpu_of.get(pid, 0.0)
                io_total += io_of.get(pid, 0.0)
                stack.extend(int(c["ProcessId"]) for c in by_parent.get(pid, []))
            return cpu_total, io_total
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
            return -1.0, -1.0

    start = time.time()
    prev_cpu, prev_io = worker_stats()
    while proc.poll() is None:
        el = time.time() - start
        if el > PER_REPO_TIMEOUT:
            proc.kill()
            return {"success": False, "error": f"timeout after {PER_REPO_TIMEOUT}s"}
        cur_cpu, cur_io = worker_stats()
        dcpu = (cur_cpu - prev_cpu) if (cur_cpu >= 0 and prev_cpu >= 0) else -1
        dio = (cur_io - prev_io) if (cur_io >= 0 and prev_io >= 0) else -1
        prev_cpu, prev_io = cur_cpu, cur_io
        cpu_str = f"cpu+{dcpu:4.1f}s" if dcpu >= 0 else "cpu=n/a"
        io_str = f"io+{dio/1e6:6.1f}MB" if dio >= 0 else "io=n/a"
        stuck = (dcpu == 0 and dio == 0)
        print(f"    {name:14s} {el:5.0f}s  {latest_stage():16s} "
              f"{cpu_str} {io_str}{'  <STALLED?>' if stuck else ''}", flush=True)
        time.sleep(HEARTBEAT_SECS)

    t.join(timeout=5)
    out = "".join(captured).strip()
    # stdout is exactly one pretty-printed JSON document (streams are separate).
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Fallback: grab the outermost {...} span if anything leaked in.
        i, j = out.find("{"), out.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(out[i:j + 1])
            except json.JSONDecodeError:
                pass
        return {"_unparsed": out[-300:]}


def has_embeddings(name: str) -> bool:
    p = emb_path(name)
    return p.is_file() and p.stat().st_size > 0


def verify_semantic(name: str) -> tuple[bool, str]:
    """Run a real --semantic search and confirm the vectors actually drive it.

    The authoritative proof that embeddings work end to end: search_mode must
    be 'hybrid' or 'semantic_only' (not 'lexical'). A present-but-unloadable or
    dim-mismatched cache would still leave search at 'lexical', which a file
    existence check alone would miss.

    NOTE: must call the tool IN-PROCESS. The `run search_sections` CLI bridge
    returns a trimmed response with NO _meta (verified), so search_mode is
    unreachable that way. The in-process tool returns the full _meta. This
    verify pays the ~6s sentence-transformers load once per repo -- acceptable
    for a correctness gate. Returns (ok, detail)."""
    try:
        # ensure this subprocess resolves the same local provider
        os.environ.setdefault("JDOCMUNCH_EMBEDDING_PROVIDER", "sentence-transformers")
        from jdocmunch_mcp.tools.search_sections import search_sections
        r = search_sections(
            repo=f"local/{name}",
            query="overview architecture configuration usage",
            semantic=True, max_results=1,
        )
    except Exception as e:  # import or runtime failure -> report, don't crash run
        return False, f"verify error: {type(e).__name__}: {e}"
    meta = r.get("_meta") or {}
    mode = meta.get("search_mode")
    tip = meta.get("tip")
    ok = mode in ("hybrid", "semantic_only")
    detail = f"search_mode={mode}" + (f" tip={tip!r}" if tip else "")
    return ok, detail


def main() -> int:
    print(f"provider pinned: {ENV['JDOCMUNCH_EMBEDDING_PROVIDER']}  "
          f"timeout={PER_REPO_TIMEOUT}s  (rebuilding ALL repos)\n")

    print("== deleting duplicate recall indexes (idempotent) ==")
    for n in DELETE_FIRST:
        if (INDEX_DIR / f"{n}.json").is_file():
            r = run_tool_quiet("delete_index", ["--repo", f"local/{n}"], timeout=60)
            print(f"  delete local/{n}: {r.get('success', r)}")
        else:
            print(f"  delete local/{n}: already gone")
    print()

    print("== rebuilding with embeddings (full, summaries off) ==")
    failures = []
    for name, root, ignore in REPOS:
        if not Path(root).is_dir():
            print(f"  SKIP {name}: root missing -> {root}")
            failures.append(name)
            continue

        t0 = time.time()
        r = index_with_status(name, root, ignore)
        dt = time.time() - t0
        built = r.get("success") is True
        sem = r.get("semantic_search")
        emb = has_embeddings(name)
        # Authoritative gate: a live --semantic search must report a non-lexical
        # mode. Only run it if the build looked good, to surface the real error
        # otherwise.
        if built and sem and emb:
            sem_ok, sem_detail = verify_semantic(name)
        else:
            sem_ok, sem_detail = False, "skipped (build did not produce vectors)"
        flag = "OK" if (built and sem and emb and sem_ok) else "CHECK"
        print(f"  [{flag}] {name:14s} sections={r.get('section_count')} "
              f"semantic_search={sem} emb_file={emb} verify[{sem_detail}] {dt:.0f}s")
        if flag == "CHECK":
            failures.append(name)
            if not built:
                print(f"        build error: {r.get('error') or r.get('_unparsed')}")
        print()

    if failures:
        print(f"DONE with issues: {', '.join(failures)}")
        return 1
    print("DONE: all repos rebuilt with embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
