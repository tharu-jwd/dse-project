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

The initial -40 dBFS proposal, with 250 ms leading and 400 ms trailing margins,
was rejected after listening showed that it could truncate low-volume speech.
It must not be used for training.

The current recheck proposal keeps the quieter -50 dBFS threshold but restores
the original margins: 250 ms at the beginning and 400 ms at the end. It would
reduce 223.63 hours to 179.53 hours, saving 44.10 hours (19.7%). The temporary
750 ms padding proposal was not adopted. These are still heuristic boundaries,
not approved training inputs. They must pass a fresh listening check and a
controlled trimmed-versus-original smoke experiment.

## Clipping

The peak check flagged 916 clips at or above 0.999 amplitude. A severity scan
found only 61 with at least 0.01% clipped samples, one with at least 0.1%, and
none with at least 1%. Eight contain a continuous clipped run of at least 1 ms;
none reaches 5 ms. The evidence does not support excluding all peak-flagged
clips. They remain in v3.

## Next gate

Before paid full training, listen to a stratified sample of proposed crop
boundaries, then run otherwise identical short experiments with original and
proposed-cropped audio. Enable cropping only if it reduces compute without
worsening strict validation WER/CER or introducing boundary truncation errors.
