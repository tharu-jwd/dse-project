import json

import pandas as pd
import pytest

from sinhala_asr.review.store import load_adjudications, save_adjudication, validate_queue


def test_queue_requires_unique_nonempty_sample_ids():
    with pytest.raises(ValueError, match="unique"):
        validate_queue(pd.DataFrame({"sample_id": ["x", "x"], "text_original": ["a", "b"]}))


def test_adjudication_is_resumable_and_latest_decision_replaces_prior(tmp_path):
    path = tmp_path / "review.jsonl"
    save_adjudication(path, {"sample_id": "b", "decision": "uncertain"})
    save_adjudication(path, {"sample_id": "a", "decision": "correct"})
    save_adjudication(path, {"sample_id": "b", "decision": "edited", "text_corrected": "නිවැරදි"})
    records = load_adjudications(path)
    assert records["b"]["decision"] == "edited"
    assert records["b"]["text_corrected"] == "නිවැරදි"
    assert [json.loads(line)["sample_id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_invalid_decision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="decision"):
        save_adjudication(tmp_path / "review.jsonl", {"sample_id": "x", "decision": "delete"})
