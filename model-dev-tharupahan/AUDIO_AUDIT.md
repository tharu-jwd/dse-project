# Audio audit and preprocessing status

No source waveform has been rewritten, denoised, normalized, resampled, or
trimmed. Dataset v3 references the immutable source audio.

All 184,664 retained train/validation/test clips are mono 16 kHz PCM-16 FLAC.
There are no globally silent retained clips. Two clips above the 30-second
limit were excluded. Exact encoded and decoded waveform hashes were used for
duplicate and leakage checks.

## Boundary silence

The complete retained dataset was measured in 20 ms frames at a linear RMS
threshold of 0.01 (-40 dBFS). Median leading silence is 1.06 seconds and median
trailing silence is 0.82 seconds. Of the retained clips, 101,839 have over one
second of leading silence, 62,101 have over one second of trailing silence, and
137,417 have more than 40% combined boundary silence.

The current audit proposal uses the original -40 dBFS threshold with 250 ms
leading and 400 ms trailing margins. It would reduce 223.63 hours to 144.45
hours, saving 79.19 hours (35.4%). Intermediate -50 dBFS and 750 ms-padding
audits were discarded at the project owner's request. These are heuristic
boundaries, not approved training inputs. They must pass the listening check
and a controlled trimmed-versus-original smoke experiment.

The owner reviewed 50 risk-stratified proposals: 30 train, 10 validation, and
10 test. All 50 were marked safe, including all 20 most aggressive proposals
in the queue. No reviewed crop was marked as cutting speech or uncertain.

Matched one-step Whisper-tiny CPU smoke runs completed for original and
dynamically cropped training audio. Both trained, evaluated against untouched
validation audio, reported metrics, and saved their model successfully. This
proves the crop plumbing, not an accuracy benefit. Both runs reported identical
model FLOPs because the current Whisper feature extractor pads inputs to a
fixed 30-second feature shape. Cropping may improve alignment, but it should not
be claimed as a GPU-compute saving unless a later measured GPU pilot proves it.

## Clipping

The peak check flagged 916 clips at or above 0.999 amplitude. A severity scan
found only 61 with at least 0.01% clipped samples, one with at least 0.1%, and
none with at least 1%. Eight contain a continuous clipped run of at least 1 ms;
none reaches 5 ms. The evidence does not support excluding all peak-flagged
clips. They remain in v3.

## Next gate

Before paid full training, run otherwise identical GPU pilots with original and
proposed-cropped training audio. Enable cropping only if it does not worsen
strict validation WER/CER or introduce boundary truncation errors.
