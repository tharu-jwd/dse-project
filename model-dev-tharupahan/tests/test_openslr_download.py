import zipfile

import pytest

from scripts.download_openslr52 import safe_extract
from scripts.index_openslr52 import read_transcripts


def test_safe_extract_accepts_normal_members(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data/audio.flac", b"audio")
    output = tmp_path / "output"
    safe_extract(archive, output)
    assert (output / "data" / "audio.flac").read_bytes() == b"audio"


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside", b"bad")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(archive, tmp_path / "output")


def test_transcript_reader_does_not_treat_quotes_as_csv_syntax(tmp_path):
    path = tmp_path / "transcripts.tsv"
    path.write_text('id1\tsp1\t"quoted transcript\nid2\tsp2\tplain\n', encoding="utf-8")
    ids, speakers, texts = read_transcripts(path)
    assert ids == ["id1", "id2"]
    assert speakers == ["sp1", "sp2"]
    assert texts == ['"quoted transcript', "plain"]
