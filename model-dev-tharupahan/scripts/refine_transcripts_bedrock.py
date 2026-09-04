#!/usr/bin/env python3
"""Request bounded, ID-aligned transcript-label suggestions from AWS Bedrock."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You review ASR training labels for Sinhala speech.
You cannot hear the audio. Preserve what was spoken; do not improve grammar,
rewrite style, translate, add omitted words, or guess unclear pronunciations.
For Sinhala, correct only definite spelling, Unicode, and word-boundary errors.
For latin_only rows, correct only definite English spelling errors; preserve
names, brands, URLs, transliterations, casing, and query-like grammar when
uncertain. Return only a JSON array containing rows that definitely need a
change. Return [] when none need changes. Each returned object must use these
keys: sample_id, corrected, change_type, reason, confidence. Copy sample_id
exactly. Do not return unchanged rows or the original field. confidence is
high, medium, or low."""


def extract_text(response: dict[str, Any]) -> str:
    blocks = response["output"]["message"]["content"]
    return "".join(str(block.get("text", "")) for block in blocks)


def parse_json_array(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    arrays = []
    for index, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            arrays.append(value)
    if not arrays:
        raise ValueError("model response does not contain a JSON array")
    return arrays[-1]


def validate(
    source: list[dict[str, Any]], result: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected = {str(row["sample_id"]): str(row["original"]) for row in source}
    seen: set[str] = set()
    for row in result:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in expected or sample_id in seen:
            raise ValueError(f"unknown or duplicate sample_id: {sample_id}")
        if not str(row.get("corrected", "")).strip():
            raise ValueError(f"blank correction for {sample_id}")
        if row.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError(f"invalid confidence for {sample_id}")
        if not str(row.get("change_type", "")).strip() or not str(
            row.get("reason", "")
        ).strip():
            raise ValueError(f"missing change metadata for {sample_id}")
        if str(row["corrected"]) == expected[sample_id]:
            # Some providers include explicit unchanged rows despite the sparse
            # instruction. Treat them as omissions rather than false changes.
            continue
        row["original"] = expected[sample_id]
        row["changed"] = True
        seen.add(sample_id)
    changes = {
        str(row["sample_id"]): row
        for row in result
        if str(row["sample_id"]) in seen
    }
    return [
        changes.get(
            str(row["sample_id"]),
            {
                "sample_id": str(row["sample_id"]),
                "original": str(row["original"]),
                "corrected": str(row["original"]),
                "changed": False,
                "change_type": None,
                "reason": "No definite correction returned.",
                "confidence": "high",
            },
        )
        for row in source
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--model", default="openai.gpt-5.6-luna")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()
    source = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Review these rows:\n"
                    + json.dumps(source, ensure_ascii=False)
                }
            ],
        }
    ]
    raw_output = args.raw_output.expanduser().resolve()
    if raw_output.is_file():
        text = raw_output.read_text(encoding="utf-8")
        response: dict[str, Any] = {}
    else:
        completed = subprocess.run(
            [
                "aws",
                "bedrock-runtime",
                "converse",
                "--region",
                args.region,
                "--model-id",
                args.model,
                "--system",
                json.dumps([{"text": SYSTEM_PROMPT}], ensure_ascii=False),
                "--messages",
                json.dumps(messages, ensure_ascii=False),
                "--inference-config",
                json.dumps({"maxTokens": args.max_tokens, "temperature": 0}),
                "--output",
                "json",
                "--no-cli-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise SystemExit(completed.stderr.strip() or "AWS Bedrock request failed")
        response = json.loads(completed.stdout)
        text = extract_text(response)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(text + "\n", encoding="utf-8")
    result = parse_json_array(text)
    result = validate(source, result)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "region": args.region,
        "usage": response.get("usage", {}),
        "stop_reason": response.get("stopReason"),
        "rows": result,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    changed = sum(bool(row.get("changed")) for row in result)
    print(
        json.dumps(
            {"rows": len(result), "changed": changed, "usage": payload["usage"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
