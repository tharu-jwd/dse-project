#!/usr/bin/env python3
"""Streamlit UI for validating proposed audio boundary crops."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

DECISIONS = ("safe", "cuts_speech", "uncertain")


def paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.queue.expanduser().resolve(), args.output.expanduser().resolve()


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            str(item["sample_id"]): item
            for line in handle
            if line.strip()
            for item in [json.loads(line)]
        }


def save(path: Path, records: dict[str, dict], record: dict) -> None:
    record["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    records[str(record["sample_id"])] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(records):
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    st.set_page_config(page_title="Audio Crop Review", layout="wide")
    queue_path, output_path = paths()
    queue = pd.read_parquet(queue_path)
    records = load(output_path)
    st.title("Audio boundary-crop review")
    st.progress(
        len(records) / len(queue), text=f"{len(records)} / {len(queue)} reviewed"
    )
    view = st.sidebar.radio("Review status", ("Unreviewed", "Reviewed", "All"))
    visible = queue
    if view == "Unreviewed":
        visible = queue[~queue.sample_id.astype(str).isin(records)]
    elif view == "Reviewed":
        visible = queue[queue.sample_id.astype(str).isin(records)]
    if visible.empty:
        st.success("No samples remain in this view.")
        return
    options = visible.index.tolist()
    key = f"crop-position-{view}"
    st.session_state[key] = min(st.session_state.get(key, 0), len(options) - 1)

    def move(delta: int) -> None:
        st.session_state[key] += delta

    left, middle, right = st.columns([1, 3, 1])
    with left:
        st.button(
            "← Previous",
            disabled=st.session_state[key] == 0,
            on_click=move,
            args=(-1,),
            shortcut="Left",
        )
    with middle:
        position = st.number_input("Queue position", 0, len(options) - 1, key=key)
    with right:
        st.button(
            "Next →",
            disabled=st.session_state[key] == len(options) - 1,
            on_click=move,
            args=(1,),
            shortcut="Right",
        )
    row = visible.loc[options[int(position)]]
    sample_id = str(row.sample_id)
    st.write(f"Transcript: **{row.text_original}**")
    original, cropped = st.columns(2)
    with original:
        st.subheader("Original")
        replay_original = st.button(
            "Play original", shortcut="O", use_container_width=True
        )
        st.audio(bytes(row.original_audio), autoplay=replay_original)
    with cropped:
        st.subheader("Proposed crop")
        replay_crop = st.button(
            "Play cropped", shortcut="Space", use_container_width=True
        )
        st.audio(bytes(row.cropped_audio), autoplay=replay_crop)
    st.caption(
        f"Removes {row.crop_start_seconds:.2f}s from the start and "
        f"{row.original_duration_seconds - row.crop_end_seconds:.2f}s from the end "
        f"({row.saved_seconds:.2f}s total)."
    )

    def decide(decision: str) -> None:
        save(
            output_path,
            records,
            {
                "sample_id": sample_id,
                "decision": decision,
                "crop_start_seconds": float(row.crop_start_seconds),
                "crop_end_seconds": float(row.crop_end_seconds),
                "queue_path": str(queue_path),
            },
        )
        st.rerun()

    st.write("Does the cropped version contain the complete spoken utterance?")
    columns = st.columns(3)
    for column, decision, label, shortcut in zip(
        columns, DECISIONS, ("Safe", "Cuts speech", "Uncertain"), ("1", "2", "3")
    ):
        with column:
            st.button(
                label,
                key=f"{decision}-{sample_id}",
                shortcut=shortcut,
                use_container_width=True,
                on_click=decide,
                args=(decision,),
            )


if __name__ == "__main__":
    main()
