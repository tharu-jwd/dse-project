"""
generate_data.py
One-shot script: generates English training clips for the command classifier.
Sources: Microsoft Edge TTS (several voices) + Google gTTS (several accents)
         + your own recorded voice (with augmentation).
Run once with: python generate_data.py
"""

import os
import asyncio
import numpy as np
import librosa
import soundfile as sf
import edge_tts
from gtts import gTTS

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
OUT_DIR = "data_train"
SR = 16000                      # openWakeWord needs 16 kHz
MY_RECORDINGS_DIR = "my_recordings"   # put your own WAV files here (see bottom)

COMMANDS = {
    "delete":   ["Delete", "Delete.", "Delete!"],
    "save":     ["save", "Save.", "Save!"],
    "submit":   ["Submit", "Submit.", "Submit!"],
    "stop":     ["Stop", "Stop.", "Stop!"],
    "option1":  ["One", "One.", "One!", "number one", "option one"],
    "option2":  ["Two", "Two.", "Two!", "number two", "option two"],
    "option3":  ["Three", "Three.", "Three!", "number three", "option three"],
    "option4":  ["Four", "Four.", "Four!", "number four", "option four"],
    "next":     ["Next", "Next.", "Next!", "next one", "go next"],
    "previous": ["Previous", "Previous.", "Previous!", "go back", "previous one"],
    "start":    ["Start", "Start.", "Start!", "start now", "let's start"],
    "cancel":   ["Cancel", "Cancel.", "Cancel!", "cancel this", "cancel that"],
}

# "none" = things that are NOT a command, including near-misses that share words
# with your commands (very important so the model doesn't over-trigger)
NONE_TEXTS = [
    "what time is it", "I am going to the shop", "the weather is nice today",
    "someone", "done", "to", "for", "start of the day", "stop by later",
    "next week", "save money", "cancel culture", "one day", "two weeks ago",
    "three of them", "four people came", "submit your application by friday",
    "delete key on the keyboard", "previous experience", "hello how are you",
]

EDGE_VOICES = [
    "en-US-JennyNeural", "en-US-GuyNeural",
    "en-GB-SoniaNeural", "en-GB-RyanNeural",
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
]
EDGE_RATES = ["-15%", "+0%", "+15%"]
EDGE_PITCHES = ["-15Hz", "+0Hz", "+15Hz"]

GTTS_ACCENTS = ["com", "co.uk", "co.in", "com.au"]   # US, UK, Indian, Australian
GTTS_SPEEDS = [False, True]                           # normal, slow

MY_VOICE_SPEEDS = [0.85, 1.0, 1.15]
MY_VOICE_PITCHES = [-4, -2, 0, 2, 4]                  # bigger range than TTS

TMP_FILE = "tmp_clip.mp3"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def all_texts():
    """Yields (label, text) for every command AND every 'none' sentence."""
    for label, texts in COMMANDS.items():
        for text in texts:
            yield label, text
    for text in NONE_TEXTS:
        yield "none", text


def save_clip(y, sr, label, name):
    """Converts to 16 kHz mono, normalizes volume, saves as WAV."""
    y = np.asarray(y, dtype=np.float32).squeeze()
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    folder = os.path.join(OUT_DIR, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".wav")
    sf.write(path, y, SR, subtype="PCM_16")


def save_with_variants(y, sr, label, name, speeds, pitches):
    """Saves the clip multiple times with different speed/pitch (for sources with few voices)."""
    y = np.asarray(y, dtype=np.float32).squeeze()
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    for s in speeds:
        ys = y if s == 1.0 else librosa.effects.time_stretch(y, rate=s)
        for p in pitches:
            yp = ys if p == 0 else librosa.effects.pitch_shift(ys, sr=SR, n_steps=p)
            save_clip(yp, SR, label, f"{name}_s{s}_p{p}")


def done(label, name):
    return os.path.exists(os.path.join(OUT_DIR, label, name + ".wav"))


def load_audio_file(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr


# ---------------------------------------------------------------------------
# SOURCE 1: Microsoft Edge TTS
# ---------------------------------------------------------------------------

async def generate_edge():
    print("\n=== Edge TTS ===")
    for voice in EDGE_VOICES:
        print("Voice:", voice)
        for i, (label, text) in enumerate(all_texts()):
            for ri, rate in enumerate(EDGE_RATES):
                for pi, pitch in enumerate(EDGE_PITCHES):
                    name = f"edge_{voice}_{i:03d}_r{ri}_p{pi}"
                    if done(label, name):
                        continue
                    try:
                        tts = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                        await tts.save(TMP_FILE)
                        y, sr = load_audio_file(TMP_FILE)
                        save_clip(y, sr, label, name)
                    except Exception as e:
                        print("  Failed:", name, e)


# ---------------------------------------------------------------------------
# SOURCE 2: Google gTTS
# ---------------------------------------------------------------------------

def generate_gtts():
    print("\n=== gTTS ===")
    for tld in GTTS_ACCENTS:
        for slow in GTTS_SPEEDS:
            print("Accent:", tld, "slow:", slow)
            for i, (label, text) in enumerate(all_texts()):
                name = f"gtts_{tld}_{i:03d}_{'slow' if slow else 'normal'}"
                if done(label, name + "_s1.0_p0"):
                    continue
                try:
                    gTTS(text=text, lang="en", tld=tld, slow=slow).save(TMP_FILE)
                    y, sr = load_audio_file(TMP_FILE)
                    save_with_variants(y, sr, label, name, speeds=[1.0], pitches=[-2, 0, 2])
                except Exception as e:
                    print("  Failed:", name, e)


# ---------------------------------------------------------------------------
# SOURCE 3: Your own recordings
# ---------------------------------------------------------------------------

def generate_my_voice():
    print("\n=== Your own recordings ===")
    if not os.path.isdir(MY_RECORDINGS_DIR):
        print(f"  Skipped: '{MY_RECORDINGS_DIR}' folder not found.")
        print(f"  Put your recordings in {MY_RECORDINGS_DIR}/<label>/*.wav and re-run.")
        return
    for label in os.listdir(MY_RECORDINGS_DIR):
        folder = os.path.join(MY_RECORDINGS_DIR, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.lower().endswith((".wav", ".mp3", ".m4a")):
                continue
            y, sr = load_audio_file(os.path.join(folder, file))
            name = "me_" + os.path.splitext(file)[0]
            save_with_variants(y, sr, label, name,
                               speeds=MY_VOICE_SPEEDS, pitches=MY_VOICE_PITCHES)
            print("  Done:", label, file)


# ---------------------------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------------------------

async def main():
    await generate_edge()
    generate_gtts()
    generate_my_voice()

    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)

    print("\n=== Done. Clip counts per folder ===")
    for label in sorted(os.listdir(OUT_DIR)):
        folder = os.path.join(OUT_DIR, label)
        n = len(os.listdir(folder))
        print(f"  {label:15s} {n}")


if __name__ == "__main__":
    asyncio.run(main())