from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.combine import combine_manifests, cross_source_report


def test_combine_and_cross_source_overlap(tmp_path: Path) -> None:
    left = pa.Table.from_pylist(
        [
            {
                "source_dataset": "a",
                "sample_id": "a-1",
                "audio_pcm_sha256": "same",
                "audio_sha256": "x",
                "transcript_sha256": "t",
            }
        ]
    )
    right = pa.Table.from_pylist(
        [
            {
                "source_dataset": "b",
                "sample_id": "b-1",
                "audio_pcm_sha256": "same",
                "audio_sha256": "y",
                "transcript_sha256": "t",
            }
        ]
    )
    pq.write_table(left, tmp_path / "left.parquet")
    pq.write_table(right, tmp_path / "right.parquet")

    combined = combine_manifests(
        [tmp_path / "left.parquet", tmp_path / "right.parquet"],
        tmp_path / "out.parquet",
    )
    report = cross_source_report(combined)

    assert combined.num_rows == 2
    assert report["decoded_audio_overlap"][0]["count"] == 1
    assert report["encoded_audio_overlap"][0]["count"] == 0
    assert report["transcript_overlap"][0]["count"] == 1
