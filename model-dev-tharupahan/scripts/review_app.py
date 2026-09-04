#!/usr/bin/env python3
"""Streamlit UI for native-speaker ASR transcript adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from sinhala_asr.review.store import (
    REVIEW_DECISIONS,
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
    suggestion_path = (
        queue_path.parent / "gpt-suggestions" / "analysis" / "suggestions.parquet"
    )
    suggestions = (
        {
            str(row["sample_id"]): row
            for row in pq.read_table(suggestion_path).to_pylist()
        }
        if suggestion_path.is_file()
        else {}
    )
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
            shortcut="Left",
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
            shortcut="Right",
        )
    row = visible.loc[options[int(index)]]
    sample_id = str(row["sample_id"])
    previous = records.get(sample_id, {})
    original_text = str(row["text_original"])
    text_key = f"text-{sample_id}"
    notes_key = f"notes-{sample_id}"
    decision_key = f"decision-{sample_id}"

    def save_transcript_edit() -> None:
        changed_text = str(st.session_state.get(text_key) or "").strip()
        if changed_text and changed_text != original_text.strip():
            save_adjudication(
                output_path,
                {
                    "sample_id": sample_id,
                    "decision": "edited",
                    "text_original": original_text,
                    "text_corrected": changed_text,
                    "notes": str(
                        st.session_state.get(notes_key) or previous.get("notes") or ""
                    ).strip(),
                    "queue_path": str(queue_path),
                },
            )

    if previous.get("decision") and previous["decision"] not in REVIEW_DECISIONS:
        st.warning(
            f"This row has the older '{previous['decision']}' decision. "
            "Review it again and save one of the four current decisions."
        )

    left, right = st.columns([2, 1])
    with left:
        audio = audio_value(row)
        if audio is None:
            st.error("This queue row has no playable audio payload or path.")
        else:
            play_requested = st.button(
                "▶ Play / replay audio",
                shortcut="Space",
                use_container_width=True,
                key=f"play-{sample_id}",
            )
            st.audio(audio, autoplay=play_requested)
        suggestion = suggestions.get(sample_id)
        previous_v3 = row.get("previous_v3_transcript")
        if isinstance(previous_v3, str) and previous_v3 != original_text:
            st.info(
                "Previous v3 text-only revision (not audio-verified):\n\n"
                f"{previous_v3}"
            )

            def use_previous_v3() -> None:
                st.session_state[text_key] = previous_v3
                save_adjudication(
                    output_path,
                    {
                        "sample_id": sample_id,
                        "decision": "edited",
                        "text_original": original_text,
                        "text_corrected": previous_v3,
                        "notes": "Selected previous v3 revision after checking audio.",
                        "queue_path": str(queue_path),
                    },
                )

            st.button(
                "Use previous v3 version after checking audio",
                on_click=use_previous_v3,
                key=f"use-v3-{sample_id}",
            )
        if suggestion and suggestion["suggested_transcript"] != original_text:
            st.info(
                f"GPT text-only suggestion ({suggestion['confidence']}, "
                f"{suggestion['change_class']}):\n\n"
                f"{suggestion['suggested_transcript']}\n\n"
                f"Reason: {suggestion['reason']}"
            )

            def accept_suggestion() -> None:
                proposed = str(suggestion["suggested_transcript"])
                st.session_state[text_key] = proposed
                save_adjudication(
                    output_path,
                    {
                        "sample_id": sample_id,
                        "decision": "edited",
                        "text_original": original_text,
                        "text_corrected": proposed,
                        "notes": f"Accepted GPT text-only suggestion: {suggestion['reason']}",
                        "queue_path": str(queue_path),
                    },
                )

            st.button(
                "Use suggestion after checking audio",
                on_click=accept_suggestion,
                key=f"accept-suggestion-{sample_id}",
            )
        st.text_area("Original transcript", original_text, disabled=True, height=120)
        corrected = st.text_area(
            "Verified transcript",
            value=str(previous.get("text_corrected") or original_text),
            height=120,
            key=text_key,
            on_change=save_transcript_edit,
        )
        if previous.get("decision") == "edited":
            st.success("Correction saved as Edited.")
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

        def save_radio_decision() -> None:
            selected_decision = str(st.session_state[decision_key])
            changed_text = str(st.session_state.get(text_key) or "").strip()
            if selected_decision == "edited" and changed_text == original_text.strip():
                return
            effective_decision = (
                "edited"
                if selected_decision == "correct"
                and changed_text != original_text.strip()
                else selected_decision
            )
            save_adjudication(
                output_path,
                {
                    "sample_id": sample_id,
                    "decision": effective_decision,
                    "text_original": original_text,
                    "text_corrected": changed_text,
                    "notes": str(
                        st.session_state.get(notes_key) or previous.get("notes") or ""
                    ).strip(),
                    "queue_path": str(queue_path),
                },
            )

        st.radio(
            "Decision",
            REVIEW_DECISIONS,
            index=(
                REVIEW_DECISIONS.index(previous["decision"])
                if previous.get("decision") in REVIEW_DECISIONS
                else REVIEW_DECISIONS.index("uncertain")
            ),
            key=decision_key,
            on_change=save_radio_decision,
        )
        notes = st.text_area(
            "Notes", value=str(previous.get("notes") or ""), key=notes_key
        )

        def persist(selected_decision: str) -> None:
            effective_decision = selected_decision
            if (
                selected_decision == "correct"
                and corrected.strip() != original_text.strip()
            ):
                effective_decision = "edited"
            if (
                effective_decision == "edited"
                and corrected.strip() == original_text.strip()
            ):
                st.error("An edited decision requires a changed transcript.")
                return
            save_adjudication(
                output_path,
                {
                    "sample_id": sample_id,
                    "decision": effective_decision,
                    "text_original": original_text,
                    "text_corrected": corrected.strip(),
                    "notes": notes.strip(),
                    "queue_path": str(queue_path),
                },
            )
            st.rerun()

        st.caption("Quick decisions: 1 correct · 2 edited · 3 bad audio · 4 uncertain")
        labels = {
            "correct": "Correct",
            "edited": "Edited",
            "bad_audio": "Bad audio",
            "uncertain": "Uncertain",
        }
        columns = st.columns(3)
        for position, selected_decision in enumerate(REVIEW_DECISIONS, start=1):
            with columns[(position - 1) % 3]:
                if st.button(
                    labels[selected_decision],
                    key=f"quick-{selected_decision}-{sample_id}",
                    shortcut=str(position),
                    use_container_width=True,
                ):
                    persist(selected_decision)


if __name__ == "__main__":
    main()
