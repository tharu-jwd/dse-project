"""WER scoring shared between evaluate_asr.py and train_asr.py.

Both scripts route through collect_predictions() + compute_wer() so a model's
score can never drift between "checked during/after training" and "the
standalone evaluator's number": there's only one place that does the
actual scoring.
"""

import io

import jiwer
import soundfile as sf


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Word Error Rate over a list of reference/hypothesis transcript pairs."""
    return jiwer.wer(refs, hyps)


def collect_predictions(transcribe_fn, dataset, num_samples: int, echo: bool = True):
    """Run `transcribe_fn` over the first `num_samples` rows of a streamed dataset.

    Expects each row to have a `text` reference and an `audio` field holding
    raw encoded bytes (`{"bytes": ...}`), matching how the OpenSLR dataset
    exposes audio when loaded with `Audio(decode=False)`. Returns (refs, hyps).
    """
    refs, hyps = [], []
    for i, sample in enumerate(dataset.take(num_samples)):
        ref = sample["text"]
        array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
        hyp = transcribe_fn(array, sr)
        refs.append(ref)
        hyps.append(hyp)
        if echo:
            print(f"\n--- sample {i + 1} ---")
            print(f"REF: {ref}")
            print(f"HYP: {hyp}")
    return refs, hyps
