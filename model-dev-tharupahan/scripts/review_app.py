#!/usr/bin/env python3
"""Streamlit UI for native-speaker ASR transcript adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sinhala_asr.review.store import (
    DECISIONS,
    load_adjudications,
    load_queue,
    reviewed_count,
    save_adjudication,
)


def parse_paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.queue.expanduser().resolve(), args.output.expanduser().resolve()


def audio_value(row: pd.Series) -> bytes | str | None:
    value = row.get("audio")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        return value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    path = row.get("audio_path")
    return str(path) if isinstance(path, str) and path else None


def main() -> None:
    st.set_page_config(page_title="Sinhala ASR Review", layout="wide")
    queue_path, output_path = parse_paths()
    queue = load_queue(queue_path)
    records = load_adjudications(output_path)
    reviewed = reviewed_count(queue, records)

    st.title("Sinhala ASR Ground-Truth Review")
    st.progress(reviewed / len(queue), text=f"{reviewed} / {len(queue)} reviewed")
    view = st.sidebar.radio("Review status", ("Unreviewed", "Reviewed", "All"))
    categories = sorted(
        str(value)
        for value in queue.get("review_category", pd.Series(dtype=str))
        .dropna()
        .unique()
    )
    selected_categories = st.sidebar.multiselect(
        "Categories", categories, default=categories
    )
    visible = queue
    if categories and selected_categories:
        visible = visible[
            visible["review_category"].astype(str).isin(selected_categories)
        ]
    reviewed_ids = set(records)
    if view == "Unreviewed":
        visible = visible[~visible["sample_id"].astype(str).isin(records)]
    elif view == "Reviewed":
        visible = visible[visible["sample_id"].astype(str).isin(reviewed_ids)]
    if visible.empty:
        st.success("No samples remain in this view.")
        return

    options = visible.index.tolist()
    position_key = f"queue-position-{view}"
    if position_key not in st.session_state:
        st.session_state[position_key] = 0
    st.session_state[position_key] = min(
        st.session_state[position_key], len(options) - 1
    )

    def move_position(delta: int) -> None:
        st.session_state[position_key] += delta

    previous_column, position_column, next_column = st.columns([1, 3, 1])
    with previous_column:
        st.button(
            "← Previous",
            disabled=st.session_state[position_key] == 0,
            use_container_width=True,
            on_click=move_position,
            args=(-1,),
        )
    with position_column:
        index = st.number_input(
            "Queue position",
            min_value=0,
            max_value=len(options) - 1,
            step=1,
            key=position_key,
        )
    with next_column:
        st.button(
            "Next →",
            disabled=st.session_state[position_key] >= len(options) - 1,
            use_container_width=True,
            on_click=move_position,
            args=(1,),
        )
    row = visible.loc[options[int(index)]]
    sample_id = str(row["sample_id"])
    previous = records.get(sample_id, {})

    left, right = st.columns([2, 1])
    with left:
        audio = audio_value(row)
        if audio is None:
            st.error("This queue row has no playable audio payload or path.")
        else:
            st.audio(audio)
        st.text_area(
            "Original transcript", str(row["text_original"]), disabled=True, height=120
        )
        corrected = st.text_area(
            "Verified transcript",
            value=str(previous.get("text_corrected") or row["text_original"]),
            height=120,
            key=f"text-{sample_id}",
        )
    with right:
        st.code(sample_id, language=None)
        for field in (
            "source_dataset",
            "speaker_id",
            "duration_seconds",
            "review_category",
            "validation_flags",
        ):
            if field in row and pd.notna(row[field]):
                value = row[field]
                if field == "validation_flags" and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                st.write(f"**{field}:**", value)
        decision = st.radio(
            "Decision",
            DECISIONS,
            index=DECISIONS.index(previous.get("decision", "correct")),
        )
        notes = st.text_area(
            "Notes", value=str(previous.get("notes") or ""), key=f"notes-{sample_id}"
        )

        def persist(selected_decision: str) -> None:
            if (
                selected_decision == "edited"
                and corrected.strip() == str(row["text_original"]).strip()
            ):
                st.error("An edited decision requires a changed transcript.")
                return
            save_adjudication(
                output_path,
                {
                    "sample_id": sample_id,
                    "decision": selected_decision,
                    "text_original": str(row["text_original"]),
                    "text_corrected": corrected.strip(),
                    "notes": notes.strip(),
                    "queue_path": str(queue_path),
                },
            )
            st.rerun()

        st.caption(
            "Quick decisions: 1 correct · 2 edited · 3 bad audio · 4 mismatch · 5 duplicate · 6 uncertain"
        )
        labels = {
            "correct": "Correct",
            "edited": "Edited",
            "bad_audio": "Bad audio",
            "mismatch": "Mismatch",
            "duplicate": "Duplicate",
            "uncertain": "Uncertain",
        }
        columns = st.columns(3)
        for position, selected_decision in enumerate(DECISIONS, start=1):
            with columns[(position - 1) % 3]:
                if st.button(
                    labels[selected_decision],
                    key=f"quick-{selected_decision}-{sample_id}",
                    shortcut=str(position),
                    use_container_width=True,
                ):
                    persist(selected_decision)
        if st.button(
            "Save selected and continue",
            type="primary",
            shortcut="Enter",
            use_container_width=True,
        ):
            persist(decision)


if __name__ == "__main__":
    main()
