"""
app.py
Plan Section 2 & 7 / Step 8.6: the IDLE/ACTIVE session state machine.

Unlike the single-shot README example, this app opens a SESSION on
"hey nexa" during which any number of commands can be spoken, until
"bye nexa" closes it (or the session times out from inactivity).

Both guard models (hey_nexa, bye_nexa) run continuously, even while
ACTIVE and mid-command-classification, since they share the EAR with the
command classifier and cost almost nothing extra.

NOTE: this file is generated but NOT run automatically — run it yourself
with: python app.py
"""

import time
import numpy as np
import joblib
import sounddevice as sd
from openwakeword.model import Model
from openwakeword.utils import AudioFeatures

from prepare_audio import SR, CLIP_LEN, prepare, to_int16

# ---------------------------------------------------------------------------
# SETTINGS — tune these empirically per plan Section 7 & Step 8.8
# ---------------------------------------------------------------------------
HEY_MODEL_PATH = "models/hey_nexa.onnx"
BYE_MODEL_PATH = "models/bye_nexa.onnx"
COMMAND_MODEL_PATH = "command_model.joblib"

WAKE_THRESHOLD = 0.5          # hey_nexa / bye_nexa detection threshold
COMMAND_CONFIDENCE_THRESHOLD = 0.8   # higher than the README's 0.7 default:
                                      # 13 classes means more chances for a
                                      # marginal call to be wrong (plan Sec. 7)
SESSION_TIMEOUT_SEC = 20.0    # auto-return to IDLE if no command this long
COMMAND_COOLDOWN_SEC = 1.5    # debounce: ignore new commands right after one fires

FRAME_SAMPLES = 1280          # 80 ms @ 16 kHz, openWakeWord's expected chunk size
COMMAND_WINDOW_SAMPLES = CLIP_LEN   # rolling window fed to the command classifier

# Map classifier labels -> app actions. Replace with real app logic.
ACTIONS = {
    "cancel":   lambda: print("[ACTION] cancel"),
    "next":     lambda: print("[ACTION] next"),
    "option1":  lambda: print("[ACTION] option1"),
    "option2":  lambda: print("[ACTION] option2"),
    "option3":  lambda: print("[ACTION] option3"),
    "option4":  lambda: print("[ACTION] option4"),
    "previous": lambda: print("[ACTION] previous"),
    "start":    lambda: print("[ACTION] start"),
    "delete":   lambda: print("[ACTION] delete"),
    "save":     lambda: print("[ACTION] save"),
    "stop":     lambda: print("[ACTION] stop"),
    "submit":   lambda: print("[ACTION] submit"),
}

IDLE, ACTIVE = "IDLE", "ACTIVE"


def classify_command(ear, receptionist, audio_int16_window):
    y = prepare(audio_int16_window.astype(np.float32) / 32767.0)
    x = to_int16(y)[None]
    numbers = ear.embed_clips(x).reshape(1, -1)
    probs = receptionist.predict_proba(numbers)[0]
    best = probs.argmax()
    return receptionist.classes_[best], probs[best]


def main():
    guard = Model(
        wakeword_models=[HEY_MODEL_PATH, BYE_MODEL_PATH],
        inference_framework="onnx",
    )
    ear = AudioFeatures()
    receptionist = joblib.load(COMMAND_MODEL_PATH)

    state = IDLE
    session_last_activity = None
    last_command_time = 0.0
    rolling_buffer = np.zeros(COMMAND_WINDOW_SAMPLES, dtype=np.int16)

    print("Listening for 'hey nexa'...")

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME_SAMPLES) as mic:
        while True:
            frame, _ = mic.read(FRAME_SAMPLES)
            frame = frame[:, 0]

            scores = guard.predict(frame)
            hey_score = scores.get("hey_nexa", 0.0)
            bye_score = scores.get("bye_nexa", 0.0)

            now = time.time()

            if state == IDLE:
                if hey_score > WAKE_THRESHOLD:
                    print("hey nexa detected -> session ACTIVE")
                    state = ACTIVE
                    session_last_activity = now
                    guard.reset()
                continue

            # state == ACTIVE
            if bye_score > WAKE_THRESHOLD:
                print("bye nexa detected -> session IDLE")
                state = IDLE
                guard.reset()
                continue

            if now - session_last_activity > SESSION_TIMEOUT_SEC:
                print("session timeout -> IDLE")
                state = IDLE
                guard.reset()
                continue

            # keep a rolling window of recent audio for command classification
            rolling_buffer = np.roll(rolling_buffer, -len(frame))
            rolling_buffer[-len(frame):] = frame

            if now - last_command_time < COMMAND_COOLDOWN_SEC:
                continue

            label, confidence = classify_command(ear, receptionist, rolling_buffer)

            if label == "none" or confidence < COMMAND_CONFIDENCE_THRESHOLD:
                continue

            print(f"command detected: {label} ({confidence:.2f})")
            action = ACTIONS.get(label)
            if action:
                action()

            session_last_activity = now
            last_command_time = now


if __name__ == "__main__":
    main()
