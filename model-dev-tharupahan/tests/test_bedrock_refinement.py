import pytest

from scripts.refine_transcripts_bedrock import parse_json_array, validate


def test_parse_and_validate_id_aligned_response() -> None:
    source = [{"sample_id": "a", "original": "helo"}]
    result = parse_json_array(
        '```json\n[{"sample_id":"a","corrected":"hello","change_type":"spelling",'
        '"reason":"typo","confidence":"high"}]\n```'
    )
    result = validate(source, result)
    assert result[0]["original"] == "helo"


def test_validation_rejects_unknown_id() -> None:
    source = [{"sample_id": "a", "original": "helo"}]
    result = [
        {
            "sample_id": "b",
            "corrected": "hello",
            "change_type": "spelling",
            "reason": "typo",
            "confidence": "high",
        }
    ]
    with pytest.raises(ValueError, match="unknown or duplicate"):
        validate(source, result)


def test_sparse_response_fills_unchanged_rows() -> None:
    source = [{"sample_id": "a", "original": "hello"}]
    result = validate(source, [])
    assert result[0]["corrected"] == "hello"
    assert result[0]["changed"] is False


def test_parser_uses_last_valid_array() -> None:
    assert parse_json_array('[{"bad": true}] reconsidering: []') == []


def test_validation_rejects_unchanged_claim() -> None:
    source = [{"sample_id": "a", "original": "hello"}]
    result = [
        {
            "sample_id": "a",
            "corrected": "hello",
            "change_type": "spelling",
            "reason": "none",
            "confidence": "high",
        }
    ]
    with pytest.raises(ValueError, match="unchanged row"):
        validate(source, result)
