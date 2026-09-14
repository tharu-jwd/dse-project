# Backend Test Results

Run: `pytest -v` from the repo root, `2026-09-14`, using `backend/.venv`.

```
92 passed in 3.73s
```

All 92 tests pass. No failures, no skips, no errors. This document explains what each test file and test actually checks, what "pass" means for it, whether the result is good or reveals a gap, and what (if anything) should be done about it.

Legend used in the "Verdict" column:
- **Good** - behavior verified is correct and the test meaningfully guards it.
- **Good, but gap** - the test itself is fine, but it exposes something the suite doesn't cover yet (noted in Action).

---

## 1. `test_command_resolution.py` - combining fuzzy + embedding voice-command matches

Tests `resolve_command`, the decision function that decides whether a spoken command should **execute**, ask the user to **confirm**, or be treated as **ordinary dictation**, given a fuzzy-text match and a voice-embedding match that may agree, disagree, or both be silent. `match_command` and `best_match` are stubbed with fake strong/borderline results so this file tests only the combination logic, not the matchers themselves.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_both_strong_same_command_executes_and_agrees` | If both fuzzy and embedding strongly agree on the same command, it executes and is flagged `agreed=True`. | PASS | Good |
| `test_strong_fuzzy_alone_executes` | A strong fuzzy match executes even with no embedding support. | PASS | Good |
| `test_strong_embedding_alone_executes` | A strong embedding match executes even with no fuzzy support. | PASS | Good |
| `test_both_strong_different_commands_asks_for_confirmation` | Two strong but *conflicting* matches never auto-execute either one - the system asks for confirmation instead of guessing. | PASS | Good - this is the important safety property of the whole module |
| `test_both_weak_with_a_borderline_candidate_asks_for_confirmation` | A borderline (not strong) fuzzy candidate with nothing else asks for confirmation rather than silently executing or silently dropping. | PASS | Good |
| `test_no_candidate_at_all_is_ordinary_dictation` | Plain speech with no match at all falls through to `"none"` (dictation), not a false command trigger. | PASS | Good |
| `test_destructive_command_below_its_higher_bar_is_not_strong` | A destructive command (e.g. delete) that clears the *normal* threshold but not the *destructive* one is not treated as strong. | PASS | Good - guards against an accidental delete from a marginal match |
| `test_destructive_command_can_still_execute_on_one_sufficiently_strong_signal` | A destructive command *can* still execute from one channel alone, provided it clears the higher destructive bar. | PASS | Good |
| `test_disabled_flag_falls_back_to_fuzzy_only_behaviour` | When embedding matching is disabled via settings, `best_match` is never even called. | PASS | Good - confirms the feature flag actually short-circuits, not just ignores the result |
| `test_unenrolled_student_empty_bank_falls_back_to_fuzzy_only` | A student with no enrolled voice samples (empty bank) still gets fuzzy-only matching instead of erroring. | PASS | Good |

**Overall**: this is the highest-value file in the suite - it's the safety logic that stops the app from mis-firing a destructive voice command. All 10 pass; no action needed.

---

## 2. `test_commands.py` - fuzzy text matching against the command vocabulary

Tests `skeleton` (Sinhala text normalization for fuzzy comparison), `match_command`, and the per-language command tables themselves.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_skeleton_strips_dependent_vowel_signs` | Sinhala dependent vowel signs are stripped so spelling variants still compare equal. | PASS | Good |
| `test_skeleton_ignores_whitespace_and_case` | Whitespace/case differences don't break matching. | PASS | Good |
| `test_skeleton_is_stable_for_command_vocabulary` | Every command phrase in the vocabulary produces a stable, non-empty skeleton (no command silently becomes unmatchable). | PASS | Good |
| `test_exact_match_returns_the_command_with_a_high_score` | An exact phrase match scores high and returns the right command. | PASS | Good |
| `test_near_miss_vowel_spelling_still_matches` | A near-miss spelling (common ASR vowel confusion) still matches. | PASS | Good |
| `test_unrelated_text_does_not_match` | Ordinary unrelated speech does not accidentally match a command. | PASS | Good |
| `test_below_threshold_returns_none_instead_of_a_guess` | A weak match below threshold returns `None` rather than a low-confidence guess. | PASS | Good |
| `test_destructive_commands_need_a_higher_threshold` | Destructive commands require a stricter score than non-destructive ones. | PASS | Good |
| `test_low_asr_confidence_rejects_a_borderline_match` | If Whisper's own confidence (`avg_logprob`) is low, a borderline match is rejected even if the text score alone would pass. | PASS | Good - two independent signals both have to agree, which reduces false positives |
| `test_low_asr_confidence_does_not_reject_a_strong_match` | Low ASR confidence does *not* reject a match that's strong enough to stand on its own. | PASS | Good |
| `test_empty_transcript_does_not_match` | Empty input never matches anything. | PASS | Good |
| `test_commands_default_is_the_sinhala_set` | Default language is Sinhala. | PASS | Good |
| `test_get_commands_returns_the_right_set_per_language` | `si`/`en` each return their own command set. | PASS | Good |
| `test_get_commands_falls_back_to_sinhala_for_an_unknown_language` | An unrecognized language code doesn't crash - it falls back to Sinhala. | PASS | Good |
| `test_same_ids_exist_in_both_languages` | Sinhala and English command tables define the same set of command IDs (no command exists in one language but not the other). | PASS | Good - catches a translation being forgotten |
| `test_destructive_flag_matches_across_languages` | The `destructive` flag for a given command ID agrees between languages. | PASS | Good |
| `test_match_command_defaults_to_sinhala` | Calling `match_command` without a language matches Sinhala phrases. | PASS | Good |
| `test_match_command_can_match_english_phrases` | English phrases match when `language="en"` is passed. | PASS | Good |
| `test_hotwords_for_returns_the_right_language` | Whisper hotword hints are generated for the correct language's vocabulary. | PASS | Good |

**Overall**: solid coverage of the text-matching layer, including the two cross-language consistency checks that guard against Sinhala/English command tables silently drifting apart. All 19 pass; no action needed.

---

## 3. `test_embeddings.py` - vector math for voice-embedding matching

Pure numpy tests for `mean_pool`, `l2_normalize`, `cosine_similarity`, `manhattan_similarity`, and `best_match`. Deliberately does not load the actual embedding model - that is exercised separately via `scripts/validate_command_embeddings.py`.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_mean_pool_collapses_time_axis` | Averaging over the time dimension produces the expected mean. | PASS | Good |
| `test_mean_pool_keeps_leading_batch_dim` | Batched input keeps its batch dimension after pooling. | PASS | Good |
| `test_l2_normalize_produces_unit_length` | A normalized vector has length 1. | PASS | Good |
| `test_l2_normalize_zero_vector_is_left_alone` | Normalizing a zero vector doesn't divide by zero / produce NaN. | PASS | Good - real edge case, e.g. total silence |
| `test_cosine_similarity_identical_vectors_is_one` | Identical vectors have similarity 1. | PASS | Good |
| `test_cosine_similarity_orthogonal_vectors_is_zero` | Orthogonal vectors have similarity 0. | PASS | Good |
| `test_cosine_similarity_opposite_vectors_is_negative_one` | Opposite vectors have similarity -1. | PASS | Good |
| `test_cosine_similarity_zero_vector_is_zero_not_nan` | Comparing against a zero vector returns 0, not NaN. | PASS | Good - same silent-audio edge case as above |
| `test_best_match_finds_the_closer_label` | Given multiple candidate labels, the closer one wins. | PASS | Good |
| `test_manhattan_similarity_identical_vectors_is_one` | Identical vectors score 1 under the Manhattan metric too. | PASS | Good |
| `test_manhattan_similarity_decreases_as_vectors_diverge` | Similarity decreases monotonically as vectors diverge. | PASS | Good |
| `test_best_match_uses_best_sample_not_mean` | Matching uses the single best enrolled sample per label, not an average of all samples (so one bad enrollment sample doesn't drag down a label that has one great sample). | PASS | Good - confirms an intentional, non-obvious design choice |
| `test_best_match_returns_none_below_threshold` | No label is returned if nothing clears the similarity threshold. | PASS | Good |
| `test_best_match_returns_none_for_empty_bank` | An empty voice bank (new/unenrolled user) returns `None` cleanly instead of erroring. | PASS | Good |
| `test_best_match_ignores_labels_with_no_samples` | A label with zero stored samples is skipped rather than crashing or matching by accident. | PASS | Good |

**Overall**: the numerically dangerous cases (zero vectors, empty banks) are explicitly covered. All 15 pass; no action needed.

---

## 4. `test_inference_kwargs.py` - Whisper hotword/prompt selection

Small, focused file on the logic that decides *how* to bias Whisper's decoding depending on mode.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_dictation_mode_never_applies_hotwords` | Normal dictation never injects command hotwords (so the model isn't biased toward command words while a student is just talking). | PASS | Good |
| `test_command_mode_uses_hotwords_when_supported` | Command mode uses Whisper's native hotword parameter when the backend supports it. | PASS | Good |
| `test_command_mode_falls_back_to_initial_prompt_when_unsupported` | If the backend doesn't support hotwords, command mode falls back to an initial-prompt trick instead of silently doing nothing. | PASS | Good |
| `test_command_mode_respects_the_disable_flag` | A settings flag can turn this biasing off entirely. | PASS | Good |

**Overall**: 4/4 pass; no action needed.

---

## 5. `test_streaming_buffer.py` - rolling PCM window (added in this task)

Unit tests for `StreamingBuffer` in `app/streaming/buffer.py`, the module that accumulates raw audio for one streaming connection and decides when to finalize a segment (VAD pause) versus force-cut it (hard time cap with no pause, keeping a trailing overlap so a word isn't cut in half).

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_rejects_overlap_not_smaller_than_max_buffer` | Construction fails fast if `overlap_seconds >= max_buffer_seconds` (a config that would make `force_cut` keep the entire buffer, defeating its purpose). | PASS | Good |
| `test_starts_empty` | A fresh buffer reports zero duration and `is_empty=True`. | PASS | Good |
| `test_append_accumulates_duration` | Appending PCM chunks correctly grows `duration_seconds`. | PASS | Good |
| `test_as_float32_normalizes_into_unit_range` | Int16 PCM converts to float32 in `[-1.0, 1.0]` (what Whisper expects), including exact min/max sample values. | PASS | Good |
| `test_exceeded_max_buffer_threshold` | `exceeded_max_buffer()` flips exactly when duration crosses the configured cap. | PASS | Good |
| `test_exceeded_memory_ceiling_threshold` | Same check for the higher, hard memory ceiling. | PASS | Good |
| `test_finalize_returns_segment_and_clears_buffer` | `finalize()` returns a segment with the correct `start`/`end` timestamps and empties the buffer. | PASS | Good |
| `test_finalize_segments_stack_on_consumed_seconds` | Two consecutive finalize cycles produce back-to-back, non-overlapping timestamps (`consumed_seconds` accumulates correctly), so a transcript's segment times stay accurate across an entire session. | PASS | Good - this is the timestamp-correctness guarantee the whole transcript UI depends on |
| `test_force_cut_keeps_only_trailing_overlap` | A forced cut discards the older audio and keeps only the trailing `overlap_seconds`, verified by checking the actual retained samples' values, not just the duration. | PASS | Good |
| `test_force_cut_on_buffer_shorter_than_overlap_keeps_everything` | If the buffer holds less audio than the overlap window, nothing is discarded and nothing is (incorrectly) marked as consumed. | PASS | Good - edge case that would otherwise double-count audio |
| `test_force_cut_with_zero_overlap_clears_buffer` | `overlap_seconds=0` cleanly empties the buffer instead of e.g. slicing with `[-0:]` (which in plain Python/numpy means "from index 0", i.e. keeps everything - a classic off-by-zero bug). | PASS | Good - this specific case is exactly the kind of bug that silently corrupts output rather than crashing |
| `test_force_cut_then_append_continues_from_kept_tail` | After a force-cut, subsequently appended audio and a later `finalize()` produce correct, continuous timestamps - i.e. force-cut and finalize compose correctly across multiple cycles, not just in isolation. | PASS | Good |

**Overall**: 12/12 pass. This module is the most timing-sensitive part of the streaming pipeline (wrong math here means transcript segments show the wrong time or duplicate/drop audio), and it's now fully covered in isolation from the network/DB/model layers, exactly as the module's own docstring says it was designed to be tested.

---

## 6. `test_streaming_commands_route.py` - the streaming route's command-vs-dictation branching

Integration-style tests (no real model, DB, or WebSocket - persistence functions are monkeypatched and a `FakeWebSocket` records what was sent) for `_emit_final` and the tick loop in the streaming route.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_delete_command_removes_last_segment_instead_of_persisting` | A recognized "delete" utterance removes the last segment instead of persisting it as new text. | PASS | Good |
| `test_stop_command_notifies_client_without_persisting` | A "stop" utterance notifies the client but is not persisted as dictated text. | PASS | Good |
| `test_ordinary_dictation_is_still_persisted_normally` | Regular speech (not a command) is still saved normally - commands don't accidentally swallow real dictation. | PASS | Good |
| `test_partial_dictation_never_invokes_embedding_matching` | Partial (in-progress) transcription results never trigger the (expensive) embedding match - only finalized utterances do. | PASS | Good - a performance/correctness guard |
| `test_unenrolled_student_still_gets_working_fuzzy_commands` | A student who hasn't enrolled voice samples still gets working commands via fuzzy matching alone. | PASS | Good |
| `test_command_executes_on_speech_end_not_on_a_tick` | Commands fire when speech actually ends (VAD), not on an arbitrary buffer tick - avoids double-firing or firing mid-word. | PASS | Good |
| `test_exactly_one_transcription_per_command_utterance` | Exactly one transcription call happens per spoken command (no duplicate inference work). | PASS | Good |
| `test_listening_sent_once_per_utterance_before_any_transcription` | The "listening" UI signal is sent exactly once, before transcription starts, per utterance. | PASS | Good |
| `test_command_mode_never_sends_partial_text` | In command mode, partial/interim text is never streamed to the client (avoids flickering half-recognized command text in the UI). | PASS | Good |
| `test_debounce_suppresses_rapid_repeat_of_same_command` | The same command spoken twice in rapid succession is debounced (doesn't fire twice from one continued utterance/echo). | PASS | Good |
| `test_debounce_allows_repeat_after_the_window_elapses` | The same command *is* allowed to fire again once the debounce window has passed. | PASS | Good |
| `test_debounce_does_not_suppress_a_different_command` | Debounce is per-command, not global - a different command right after is not suppressed. | PASS | Good |
| `test_buffer_clock_stays_correct_across_mixed_finalize_and_force_cut` | Route-level integration check that timestamps stay correct through a real mix of finalize/force-cut events (the route-level counterpart to the unit tests in `test_streaming_buffer.py`). | PASS | Good |
| `test_command_mode_never_starts_the_tick_loop` | Command mode does not start the periodic re-transcription tick loop (unnecessary in that mode - commands are event-driven). | PASS | Good |
| `test_note_mode_still_starts_the_tick_loop` | Ordinary note-taking mode does start the tick loop (contrast case to the above, confirming the branch is mode-specific, not just always-off). | PASS | Good |

**Overall**: 15/15 pass. This file is the most complex in the suite (async, multiple interacting concerns) and its docstring explicitly calls out that it deliberately avoids a real model/DB/WebSocket to stay fast and deterministic. No action needed.

---

## 7. `test_voice_enrollment.py` - voice-command enrollment and practice

The only file whose fixture (`db_user`, from `conftest.py`) writes to a **real Postgres database** (the same instance the app runs against), because command enrollments have a foreign key to a real user row. Everything else in this suite is a pure in-memory/monkeypatched unit test.

| Test | What it verifies | Result | Verdict |
|---|---|---|---|
| `test_progress_starts_at_zero_for_every_command` | A new user's enrollment progress is 0 for every command. | PASS | Good |
| `test_first_sample_for_a_command_is_always_accepted` | The very first voice sample for a command is always accepted (nothing to compare it against yet). | PASS | Good |
| `test_similar_second_sample_is_accepted_and_counted` | A second sample similar to the first is accepted and progress increments. | PASS | Good |
| `test_outlier_sample_is_rejected_not_stored` | A sample too dissimilar from prior ones is rejected and not stored (protects enrollment quality). | PASS | Good |
| `test_cannot_exceed_required_sample_count` | Enrollment stops accepting once the required number of samples is reached. | PASS | Good |
| `test_delete_samples_resets_progress` | Deleting a user's samples resets their progress back to 0. | PASS | Good |
| `test_unknown_command_id_is_rejected` | Enrolling against a command ID that doesn't exist is rejected, not silently accepted. | PASS | Good |
| `test_load_bank_groups_by_command_for_matching` | Loading the voice bank groups stored samples correctly by command for later matching. | PASS | Good |
| `test_load_bank_skips_samples_from_a_different_model_version` | Samples embedded with an older/different model version are excluded from the bank (prevents comparing embeddings from incompatible model versions). | PASS | Good - guards a real migration hazard if the embedding model is ever upgraded |
| `test_sinhala_and_english_samples_for_the_same_id_never_collide` | A command with the same ID in both languages keeps Sinhala and English samples separate. | PASS | Good |
| `test_progress_is_scoped_to_one_language` | Enrollment progress is tracked per language, not shared. | PASS | Good |
| `test_deleting_one_languages_samples_leaves_the_other_untouched` | Deleting one language's samples doesn't affect the other language's enrollment. | PASS | Good |
| `test_unknown_command_id_for_a_language_is_rejected` | Same "unknown command" rejection, scoped per language. | PASS | Good |
| `test_practice_reports_similarity_against_own_enrolled_samples` | The practice/self-check feature reports similarity against the user's own enrolled samples. | PASS | Good |
| `test_practice_with_no_enrolled_samples_reports_nothing_to_compare` | Practicing with nothing enrolled yet reports "nothing to compare" instead of crashing or a false result. | PASS | Good |
| `test_practice_flags_a_closer_match_to_a_different_enrolled_command` | If a practice sample actually sounds closer to a *different* enrolled command, that's flagged (helps a user notice they're saying the wrong word). | PASS | Good |
| `test_practice_rejects_unknown_command_id` | Practicing against an unknown command ID is rejected. | PASS | Good |

**Overall**: 17/17 pass. Functionally solid, but see the gap noted below - this is the one file where "pass" depends on a live database connection.

---

## Gaps found while reviewing the suite (not test failures - things worth doing next)

None of these are bad results; the tests all pass. These are honest gaps the review surfaced:

1. **`test_voice_enrollment.py` depends on a real Postgres connection**, per `conftest.py`'s own docstring: *"There's no isolated test-database setup in this project yet... this talks to the same Postgres instance the app runs against."* This means:
   - These 17 tests cannot run in an environment without the dev database up (e.g. a clean CI runner), unlike the other 75 tests which are pure in-memory.
   - Each test run writes and deletes a throwaway user row against a real instance - low risk today, but it's the one place a bug in a test's cleanup could leave orphaned rows in a shared database.
   - **Action: not taken.** Introducing an isolated test database (e.g. a Dockerized Postgres fixture, or SQLite for unit-level FK satisfaction) is a real infrastructure change outside the scope of "add buffer tests," and would need its own decision about tooling (testcontainers, a docker-compose override, etc.). Flagging it here so it's a visible, deliberate backlog item rather than a silent limitation.

2. **No frontend automated tests exist** (confirmed separately, and explicitly out of scope per your instruction not to add them). `frontend/` has no Jest/Vitest/Testing Library installed and no test script in `package.json`. Everything above only covers the backend.
   - **Action: not taken**, per your instruction.

3. **`app.streaming.embeddings`'s docstring notes `embed_audio` itself (the function that actually runs the model) is validated separately** via `scripts/validate_command_embeddings.py`, not by this pytest suite. That's an intentional split (pytest for pure math, a manual script for the real model), not a gap introduced by this review - noted here only so the coverage picture is complete.

No test in this run failed or needs a fix; the items above are suite-level infrastructure notes, not defects.
