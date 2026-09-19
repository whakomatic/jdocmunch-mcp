"""v1.105.0 — office-format ingestion (.pdf/.docx/.pptx/.epub) via markitdown.

All tests run WITHOUT markitdown installed: availability is forced through
``parser.office._AVAILABLE`` and conversion through a stubbed converter, so
CI needs no heavy office dependencies.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jdocmunch_mcp.parser import parse_file, ALL_EXTENSIONS  # noqa: E402
from jdocmunch_mcp.parser import office  # noqa: E402
from jdocmunch_mcp.tools.index_local import index_local, discover_doc_files  # noqa: E402
from jdocmunch_mcp.tools.index_file import index_file  # noqa: E402
from jdocmunch_mcp.watch import _doc_extensions  # noqa: E402


class _StubConverter:
    """Stands in for markitdown.MarkItDown."""

    def __init__(self, text="# Converted Title\n\nConverted body prose.\n"):
        self.text = text
        self.calls = 0

    def convert(self, path):
        self.calls += 1

        class _R:
            pass

        r = _R()
        r.text_content = self.text
        return r


@pytest.fixture(autouse=True)
def _reset_office_state():
    office._reset_for_tests()
    yield
    office._reset_for_tests()


@pytest.fixture
def stub_converter(monkeypatch):
    stub = _StubConverter()
    monkeypatch.setattr(office, "_AVAILABLE", True)
    monkeypatch.setattr(office, "_get_converter", lambda: stub)
    return stub


def test_office_extensions_not_in_all_extensions():
    # Deliberate: ALL_EXTENSIONS drives the remote GitHub leg and other
    # text-only consumers. Office support gates each local leg explicitly.
    for ext in office.OFFICE_EXTENSIONS:
        assert ext not in ALL_EXTENSIONS


def test_discovery_skips_office_without_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", False)
    (tmp_path / "spec.docx").write_bytes(b"PK\x03\x04 not really a docx")
    (tmp_path / "readme.md").write_text("# hi\n", encoding="utf-8")
    skip_counts = {}
    files, warnings, discovered, _ = discover_doc_files(tmp_path, skip_counts=skip_counts)
    names = {f.name for f in files}
    assert names == {"readme.md"}
    assert skip_counts.get("office_extra_not_installed") == 1


def test_discovery_accepts_office_with_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", True)
    (tmp_path / "spec.docx").write_bytes(b"PK\x03\x04")
    files, warnings, discovered, _ = discover_doc_files(tmp_path)
    assert {f.name for f in files} == {"spec.docx"}


def test_office_size_cap_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", True)
    import jdocmunch_mcp.tools.index_local as il
    monkeypatch.setattr(il, "OFFICE_MAX_FILE_SIZE", 4)
    (tmp_path / "big.pdf").write_bytes(b"12345678")
    skip_counts = {}
    files, _, _, _ = discover_doc_files(tmp_path, skip_counts=skip_counts)
    assert files == []
    assert skip_counts.get("oversize") == 1


def test_parse_file_routes_office_content_to_markdown():
    sections = parse_file("# Heading\n\nBody text here.\n", "manual.pdf", "o/r")
    assert sections, "office content must parse via the markdown parser"
    assert any(s.title == "Heading" for s in sections)


def test_index_local_end_to_end_with_stub(tmp_path, stub_converter):
    (tmp_path / "guide.docx").write_bytes(b"PK\x03\x04")
    storage = tmp_path / ".store"
    result = index_local(
        path=str(tmp_path), name="officetest", use_ai_summaries=False,
        use_embeddings=False, storage_path=str(storage),
    )
    assert result.get("success"), result
    assert stub_converter.calls == 1
    from jdocmunch_mcp.storage import DocStore
    store = DocStore(base_path=str(storage))
    index = store.load_index("local", "officetest")
    assert "guide.docx" in index.doc_paths
    titles = {s.get("title") for s in index.sections}
    assert "Converted Title" in titles


def test_conversion_cache_prevents_reconversion(tmp_path, stub_converter):
    doc = tmp_path / "deck.pptx"
    doc.write_bytes(b"PK\x03\x04 slide bytes")
    cache = tmp_path / "cache"
    first = office.convert_office(doc, cache_dir=cache)
    second = office.convert_office(doc, cache_dir=cache)
    assert first == second == stub_converter.text
    assert stub_converter.calls == 1  # second call served from cache
    doc.write_bytes(b"PK\x03\x04 CHANGED bytes")
    office.convert_office(doc, cache_dir=cache)
    assert stub_converter.calls == 2  # changed bytes -> new key -> reconvert


def test_index_file_errors_cleanly_without_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", False)
    doc = tmp_path / "notes.epub"
    doc.write_bytes(b"epub bytes")
    result = index_file(str(doc), storage_path=str(tmp_path / ".store"))
    assert result["success"] is False
    assert "[office]" in result["error"]


def test_explicit_paths_office_gating(tmp_path, monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", False)
    (tmp_path / "spec.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.md").write_text("# a\n", encoding="utf-8")
    result = index_local(
        path=str(tmp_path), name="pathgate", use_ai_summaries=False,
        use_embeddings=False, storage_path=str(tmp_path / ".store"),
        paths=["spec.pdf", "a.md"],
    )
    assert result.get("success"), result
    assert any("[office]" in w for w in result.get("warnings", []))


def test_watch_extensions_include_office_only_when_available(monkeypatch):
    monkeypatch.setattr(office, "_AVAILABLE", False)
    assert not (office.OFFICE_EXTENSIONS & _doc_extensions())
    monkeypatch.setattr(office, "_AVAILABLE", True)
    assert office.OFFICE_EXTENSIONS <= _doc_extensions()
