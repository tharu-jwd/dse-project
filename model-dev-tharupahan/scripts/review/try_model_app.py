#!/usr/bin/env python3
"""Streamlit UI to interactively try any completed experiment's model.

Pick a completed experiment's adapter (or the untouched baseline), then feed
it either your own uploaded audio, a real row from the frozen 206-row Sinhala
validation set, or a row from the 2,620-row English-retention benchmark.
Shows the raw prediction, a word-level diff against the reference when one
exists, and single-utterance strict/canonical WER/CER -- a qualitative
complement to the aggregate WER/CER numbers in the experiment reports, not a
replacement for them (one clip proves nothing statistically; the frozen
validation/test evaluations remain the real evidence).

Runs entirely locally and offline once the base model is cached (it already
is, at ~/.cache/huggingface/hub/models--openai--whisper-small).

Usage (run from the repo root):
    PYTHONPATH=src streamlit run scripts/review/try_model_app.py
"""

from __future__ import annotations

import difflib
import io
import random
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import streamlit as st
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sinhala_asr.evaluation.metrics import _characters, edit_counts
from sinhala_asr.text.normalizer import metric_normalize

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "openai/whisper-small"

VALIDATION_BUNDLE = (
    ROOT / "reports/kaggle/e003-input-dataset/v4-validation-206-audio.parquet"
)
ENGLISH_BENCHMARK = (
    ROOT / "reports/kaggle/e003-english-evaluation-inputs/english-retention-test-clean.parquet"
)

# Each entry's adapter directory is the exact final-adapter this project
# treats as that experiment's canonical, hash-verified result (see each
# experiment's report in docs/experiments/). Missing directories are filtered
# out at startup rather than shown as broken options.
CANDIDATE_EXPERIMENTS: dict[str, Path | None] = {
    "E000 -- untouched whisper-small": None,
    "E001 -- wide LoRA, 100 steps, 2k rows": ROOT
    / "reports/experiments/e001-whisper-small-wide-lora-r16-100-step-v4/artifacts/final-adapter",
    "E002 -- wide LoRA, 500 steps, 10k rows": ROOT
    / "reports/experiments/e002-whisper-small-wide-lora-r16-500-step-v4/artifacts/final-adapter",
    "E003 -- + 10% raw-reference English replay": ROOT
    / "reports/experiments/e003-english-replay-lora/attempts/kaggle-training-002/output/e003-training/final-adapter",
    "E004 -- + teacher-target English replay": ROOT
    / "reports/experiments/e004-teacher-behavior-replay/attempts/kaggle-training-001/output/e004-training/final-adapter",
    "E005 -- scaled to 50k Sinhala rows": ROOT
    / "reports/experiments/e005-scale-50k-teacher-replay/attempts/kaggle-training-003/output/e005-training/final-adapter",
    "E006 -- scaled to 100 Sinhala hours": ROOT
    / "reports/experiments/e006-scale-100h-teacher-replay/attempts/kaggle-training-001/output/e006-training/final-adapter",
}


def available_experiments() -> dict[str, Path | None]:
    return {
        label: path
        for label, path in CANDIDATE_EXPERIMENTS.items()
        if path is None or (path / "adapter_model.safetensors").is_file()
    }


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_processor() -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(BASE_MODEL)


@st.cache_resource(show_spinner="Loading model (first use per experiment is slower)...")
def load_model(adapter_dir: str | None) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.generation_config.forced_decoder_ids = None
    model.to(pick_device()).eval()
    return model


@st.cache_resource(show_spinner="Loading frozen validation set...")
def load_validation_rows() -> list[dict]:
    return pq.read_table(VALIDATION_BUNDLE).to_pylist()


@st.cache_resource(show_spinner="Loading English-retention benchmark...")
def load_english_rows() -> list[dict]:
    return pq.read_table(ENGLISH_BENCHMARK).to_pylist()


def decode_audio(raw: bytes) -> tuple:
    samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sample_rate != 16000:
        import numpy as np

        duration = len(samples) / sample_rate
        target_len = int(round(duration * 16000))
        samples = np.interp(
            np.linspace(0, len(samples), target_len, endpoint=False),
            np.arange(len(samples)),
            samples,
        ).astype("float32")
    return samples, 16000


def transcribe(samples, language: str) -> str:
    processor = load_processor()
    model = st.session_state["_active_model"]
    device = pick_device()
    features = processor.feature_extractor(
        samples, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
    )
    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.inference_mode():
        generated = model.generate(
            features.input_features.to(device),
            attention_mask=features.attention_mask.to(device),
            forced_decoder_ids=forced_ids,
            max_new_tokens=64,
            no_repeat_ngram_size=3,
        )
    return processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def render_word_diff(reference: str, prediction: str) -> str:
    ref_words = metric_normalize(reference).split()
    hyp_words = metric_normalize(prediction).split()
    sm = difflib.SequenceMatcher(None, ref_words, hyp_words, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            spans.append(" ".join(ref_words[i1:i2]))
        elif tag == "replace":
            spans.append(
                f'<span style="background:#442222;color:#ff9999;text-decoration:line-through">'
                f'{" ".join(ref_words[i1:i2])}</span> '
                f'<span style="background:#224422;color:#99ff99">{" ".join(hyp_words[j1:j2])}</span>'
            )
        elif tag == "delete":
            spans.append(
                f'<span style="background:#442222;color:#ff9999;text-decoration:line-through">'
                f'{" ".join(ref_words[i1:i2])}</span>'
            )
        elif tag == "insert":
            spans.append(
                f'<span style="background:#224422;color:#99ff99">{" ".join(hyp_words[j1:j2])}</span>'
            )
    return " ".join(spans)


def show_result(prediction: str, reference: str | None) -> None:
    st.markdown("**Prediction:**")
    st.code(prediction or "(empty)")
    if not reference:
        return
    st.markdown("**Diff vs. reference** (red/strikethrough = missed, green = wrong/extra):")
    st.markdown(render_word_diff(reference, prediction), unsafe_allow_html=True)
    ref_norm = metric_normalize(reference)
    hyp_norm = metric_normalize(prediction)
    w = edit_counts(ref_norm.split(), hyp_norm.split())
    c = edit_counts(_characters(ref_norm), _characters(hyp_norm))
    col1, col2 = st.columns(2)
    col1.metric("WER (this clip)", f"{w['errors'] / max(w['reference_units'], 1):.1%}")
    col2.metric("CER (this clip)", f"{c['errors'] / max(c['reference_units'], 1):.1%}")
    st.caption(
        "One clip is not a statistically meaningful result -- see the experiment's "
        "own report for the real, paired, confidence-interval-backed evaluation."
    )


def main() -> None:
    st.set_page_config(page_title="Try a Sinhala ASR model", layout="wide")
    st.title("Try a Sinhala ASR model")

    experiments = available_experiments()
    label = st.sidebar.selectbox("Experiment", list(experiments))
    language = st.sidebar.radio("Transcribe as", ["Sinhala", "English"], horizontal=True)
    lang_code = "si" if language == "Sinhala" else "en"

    adapter_dir = experiments[label]
    st.session_state["_active_model"] = load_model(str(adapter_dir) if adapter_dir else None)

    tab_upload, tab_val, tab_english = st.tabs(
        ["Upload audio", "Frozen Sinhala validation (206 rows)", "English-retention benchmark (2,620 rows)"]
    )

    with tab_upload:
        uploaded = st.file_uploader("Audio file (wav/flac work reliably)", type=None)
        if uploaded is not None:
            raw = uploaded.read()
            st.audio(raw)
            if st.button("Transcribe", key="transcribe_upload"):
                try:
                    samples, _ = decode_audio(raw)
                except Exception as exc:  # noqa: BLE001 -- surface decode errors directly
                    st.error(f"Could not decode this audio file: {exc}")
                else:
                    show_result(transcribe(samples, lang_code), reference=None)

    with tab_val:
        rows = load_validation_rows()
        ids = [row["sample_id"] for row in rows]
        col_a, col_b = st.columns([3, 1])
        selected_id = col_a.selectbox("Sample", ids, key="val_select")
        if col_b.button("Random", key="val_random"):
            selected_id = random.choice(ids)
            st.session_state["val_select"] = selected_id
            st.rerun()
        row = next(r for r in rows if r["sample_id"] == selected_id)
        st.caption(
            f"speaker {row['speaker_id']} | {row['language_class']} | "
            f"{row['duration_seconds']:.1f}s"
        )
        st.audio(bytes(row["audio"]))
        st.markdown(f"**Reference:** {row['reference']}")
        if st.button("Transcribe", key="transcribe_val"):
            samples, _ = decode_audio(bytes(row["audio"]))
            show_result(transcribe(samples, lang_code), row["reference"])

    with tab_english:
        rows = load_english_rows()
        ids = [row["sample_id"] for row in rows]
        col_a, col_b = st.columns([3, 1])
        selected_id = col_a.selectbox("Sample", ids, key="en_select")
        if col_b.button("Random", key="en_random"):
            selected_id = random.choice(ids)
            st.session_state["en_select"] = selected_id
            st.rerun()
        row = next(r for r in rows if r["sample_id"] == selected_id)
        st.audio(bytes(row["audio"]))
        st.markdown(f"**Reference:** {row['reference']}")
        if st.button("Transcribe", key="transcribe_en"):
            samples, _ = decode_audio(bytes(row["audio"]))
            show_result(transcribe(samples, lang_code), row["reference"])


if __name__ == "__main__":
    main()
