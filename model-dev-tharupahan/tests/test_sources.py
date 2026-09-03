import hashlib
import json

from sinhala_asr.data.sources import inventory_registry, inventory_source


def test_inventory_is_content_addressed_and_ignores_partial_downloads(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.txt").write_text("audio", encoding="utf-8")
    (root / "ignored.part").write_text("partial", encoding="utf-8")
    result = inventory_source(root)
    assert result["file_count"] == 1
    assert result["files"][0]["sha256"] == hashlib.sha256(b"audio").hexdigest()


def test_registry_preserves_provenance_and_adds_inventory(tmp_path):
    (tmp_path / "data" / "source").mkdir(parents=True)
    (tmp_path / "data" / "source" / "sample").write_bytes(b"x")
    registry = tmp_path / "sources.json"
    registry.write_text(
        json.dumps({"schema_version": "v1", "sources": [{"id": "s", "local_path": "data/source"}]}),
        encoding="utf-8",
    )
    result = inventory_registry(registry, tmp_path)
    assert result["sources"][0]["id"] == "s"
    assert result["sources"][0]["inventory"]["total_bytes"] == 1
