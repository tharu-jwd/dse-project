"""TEMPORARY stand-in for the real preprocessing pipeline's output.

The actual fine-tuning dataset will come from a separate preprocessing
pipeline (dataset cleaning, transcript normalization, filtering, etc.) that
someone else is building against OpenSLR. Its exact output format isn't
finalized yet. Until it's ready, we use the public
SPEAK-ASR/openslr-sinhala-asr dataset directly as a development/smoke-test
source.

This is the ONLY place in the fine-tuning workflow that knows anything about
this specific dataset (its id, columns, splits). When the real preprocessed
dataset is ready, only this function needs to change. train_asr.py just
expects a `datasets` Dataset/IterableDataset with `audio` and `text` columns,
whatever produces it.
"""

from datasets import Value, load_dataset

DEV_DATASET = "SPEAK-ASR/openslr-sinhala-asr"

# The dataset's native "audio" column is an Audio feature, which datasets
# encodes/decodes through torchcodec, unreliable in this environment (ABI
# mismatch between the installed torchcodec and torch builds). Audio is
# stored as a {bytes, path} struct underneath either way, so reinterpreting
# it as plain Values sidesteps torchcodec entirely; decoding then happens
# downstream via soundfile, same as evaluate_asr.py already does.
RAW_AUDIO_TYPE = {"bytes": Value("binary"), "path": Value("string")}


def load_dev_openslr_split(split: str, streaming: bool = False):
    """Load one split of the placeholder dataset, audio as raw undecoded bytes."""
    ds = load_dataset(DEV_DATASET, split=split, streaming=streaming)
    return ds.cast_column("audio", RAW_AUDIO_TYPE)
