"""Minimal local server to test Whisper transcription on live mic recordings.

Run:  python3 server.py
Open: http://localhost:5005
"""

import os
import tempfile
import threading
import time

import whisper
from flask import Flask, jsonify, request, send_from_directory
from peft import PeftModel
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline as hf_pipeline

app = Flask(__name__)

# Loaded models are cached so switching back to one is instant.
_models = {}

# Whisper's decoder installs kv-cache hooks on the shared model object, so two
# concurrent transcriptions corrupt each other. Serialize them.
_transcribe_lock = threading.Lock()

ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3", "speak-asr-si"}

SPEAK_ASR_BASE = "openai/whisper-medium"
SPEAK_ASR_ADAPTER = "SPEAK-ASR/whisper-si-exp-10-medium-all"


def get_model(name: str):
    with _transcribe_lock:
        if name not in _models:
            if name == "speak-asr-si":
                print("Loading SPEAK-ASR Sinhala adapter (base whisper-medium + LoRA)...")
                processor = AutoProcessor.from_pretrained(SPEAK_ASR_BASE)
                base_model = WhisperForConditionalGeneration.from_pretrained(SPEAK_ASR_BASE)
                peft_model = PeftModel.from_pretrained(base_model, SPEAK_ASR_ADAPTER)
                merged_model = peft_model.merge_and_unload()
                merged_model.eval()
                # chunk_length_s/stride_length_s enable long-form transcription:
                # without this, the model only ever sees the first 30s of audio
                # (Whisper's encoder window) and silently drops the rest.
                _models[name] = hf_pipeline(
                    "automatic-speech-recognition",
                    model=merged_model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    stride_length_s=5,
                )
            else:
                print(f"Loading whisper model '{name}' (first time may download weights)...")
                _models[name] = whisper.load_model(name)
        return _models[name]


def transcribe_speak_asr(asr_pipeline, audio):
    forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="si", task="transcribe")
    result = asr_pipeline(
        {"array": audio, "sampling_rate": 16000},
        generate_kwargs={"forced_decoder_ids": forced_decoder_ids, "max_new_tokens": 440},
    )
    return {"text": result["text"].strip(), "segments": []}


@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.post("/transcribe")
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file in request"}), 400

    model_name = request.form.get("model", "small")
    if model_name not in ALLOWED_MODELS:
        return jsonify({"error": f"unknown model '{model_name}'"}), 400

    language = request.form.get("language", "si")
    lang_arg = None if language == "auto" else language

    suffix = os.path.splitext(request.files["audio"].filename or "clip.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        request.files["audio"].save(f.name)
        audio_path = f.name

    try:
        model = get_model(model_name)
        try:
            audio = whisper.load_audio(audio_path)
        except Exception as e:
            return jsonify({"error": f"could not decode audio: {e}"}), 422
        if audio.shape[0] < 1600:  # under 0.1s — partial container chunks decode to nothing
            return jsonify({"text": "", "language": language, "model": model_name, "seconds": 0, "segments": []})
        start = time.time()
        with _transcribe_lock:
            if model_name == "speak-asr-si":
                result = transcribe_speak_asr(model, audio)
            else:
                result = model.transcribe(audio, language=lang_arg, fp16=False)
        elapsed = time.time() - start
        return jsonify(
            {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "model": model_name,
                "seconds": round(elapsed, 2),
                "segments": [
                    {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                    for s in result.get("segments", [])
                ],
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        os.unlink(audio_path)


if __name__ == "__main__":
    print("Preloading 'small' model...")
    get_model("small")
    print("Ready — open http://localhost:5005")
    app.run(host="127.0.0.1", port=5005, debug=False)
