# Sinhala ASR Fine-Tuning Strategy

This plan assumes fine-tuning starts again from a pretrained multilingual
Whisper checkpoint rather than training an ASR model from randomly initialized
weights. The goal is to reduce Sinhala word error rate (WER) while keeping the
experiments comparable and reproducible.

## 1. Establish a trustworthy baseline

Before changing the model, freeze one test set and one evaluation pipeline.

- Split by speaker so the same speaker cannot appear in training and testing.
- Keep identical or near-identical recordings and sentences in one split.
- Stratify by speaker, accent or district, source corpus, recording device,
  duration, and Sinhala-only versus code-switched speech where metadata allows.
- Report WER and character error rate (CER) on the same test set for every run.
- Report separate results for Sinhala-only speech, code-switched speech, named
  entities, numbers, and domain terminology.

A random clip-level split may underestimate real-world error because the model
can encounter the same speakers or sentences during training and testing.

## 2. Clean and normalize the training data

Data and label quality should be treated as the highest-priority experiment.

- Normalize Sinhala text to Unicode NFC.
- Standardize punctuation, whitespace, digits, dates, currencies,
  abbreviations, and English words.
- Adopt one explicit policy for Sinhala-English code-switching: preserve English
  terms in Latin script, transliterate them, or apply a documented controlled
  mixture consistently.
- Remove empty, corrupt, clipped, excessively silent, truncated, and
  audio-transcript-mismatched examples.
- Detect duplicate audio, repeated transcripts, and train-test leakage.
- Preserve original transcripts for traceability while training against a
  canonical transcript field.

Use the same canonical normalization rules for evaluation. Report both strict
WER, which represents readable output, and normalized WER, which focuses on
linguistic recognition accuracy.

## 3. Use error analysis to select the next experiment

The best completed Sinhala model's errors were dominated by substitutions,
followed by deletions and insertions. That makes label consistency, vocabulary
coverage, and acoustically confusable words higher priorities than simply adding
more synthetic noise.

For each model, calculate:

- The most frequent reference-to-prediction substitutions
- WER by speaker, source corpus, duration, and transcript frequency
- WER for Sinhala-only and code-switched speech
- Named-entity, number, financial-term, and command WER
- Silence, repetition, and hallucination rates

Manually review a representative sample of the most frequent and most severe
errors. Categorize each error as a labeling problem, acoustic problem,
vocabulary problem, segmentation problem, or decoding problem.

## 4. Tune full fine-tuning around the successful region

Run all configurations against the same frozen data and evaluation pipeline.
Save and evaluate every epoch, using validation WER for checkpoint selection and
early stopping with a patience of two evaluations.

| Experiment | Learning rate | Scheduler | Maximum epochs |
|---|---:|---|---:|
| A | `1e-5` | cosine | 6-8 |
| B | `2e-5` | cosine | 6 |
| C | `3e-5` | cosine | 4-6 |
| D | `5e-5` | cosine | 4 |

The earlier `3e-5` full fine-tune produced the best Sinhala result, while the
later `1e-5` experiment preserved English better but had worse Sinhala WER. The
runs used different dataset versions, so their difference cannot safely be
attributed to learning rate until they are repeated on identical data.

Track training loss, validation loss, WER, CER, gradient norm, learning rate,
and checkpoint identity. Select the checkpoint with the best validation WER,
not automatically the final checkpoint.

## 5. Compare model capacity

Apply the winning data and training recipe to:

- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3`, if training and deployment resources permit

Keep the split, normalization, augmentation, and decoding configuration fixed.
Compare accuracy alongside memory use, real-time factor, and deployment cost.

## 6. Test staged adaptation

A useful staged schedule is:

1. Train on the full cleaned and diverse Sinhala corpus.
2. Continue at a lower learning rate on manually verified, domain-matched data.
3. Optionally finish with a small, balanced collection targeting frequent error
   categories and rare vocabulary.

Example starting schedule:

- Stage 1: `2e-5` for 3-4 epochs on the broad corpus
- Stage 2: `5e-6` for 1-2 epochs on verified domain data
- Stage 3: `2e-6` for at most 1 epoch on error-focused data

Use a capped or weighted sampler instead of copying difficult examples many
times, which risks memorization.

## 7. Compare adaptation methods

Run controlled comparisons of:

- Full-model fine-tuning
- Encoder frozen during the first epoch, followed by full unfreezing
- Decoder-focused initial adaptation
- Wide-target LoRA over `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and
  `fc2`
- LoRA adaptation followed by a short, low-learning-rate full fine-tune

Full fine-tuning is currently the leading strategy for Sinhala-only accuracy.
Wide-target LoRA remains useful when English retention and a small checkpoint
are important.

## 8. Improve data diversity

Collect new speakers and acoustic conditions rather than many repetitions from
existing speakers. Priority categories include:

- Regional Sinhala accents
- Conversational and read speech
- Male, female, elderly, and child speakers
- Phones, headsets, laptop microphones, and compressed audio
- Quiet rooms, traffic, fans, television, classrooms, and multiple speakers
- The terminology and speaking style used in the deployment application
- Natural Sinhala-English code-switching

Every new collection should reserve unseen speakers for evaluation.

## 9. Add filtered pseudo-labeled speech

Use the best model, or a larger Whisper model, as a teacher for untranscribed
Sinhala audio.

1. Generate candidate transcripts and confidence indicators.
2. Reject wrong-language output, repetition, hallucinations, excessive silence,
   and disagreement between multiple decoding passes or teacher models.
3. Human-review a representative sample and important domain examples.
4. Mix the accepted pseudo-labeled data with supervised data at a controlled
   ratio, initially 25-50 percent.
5. Compare against a supervised-only control.

Low-quality pseudo-labels can reinforce existing spelling and vocabulary
errors, so filtering quality matters more than raw volume.

## 10. Run augmentation ablations

The existing pipeline supports SpecAugment, Gaussian noise, time stretching,
and pitch shifting. Test each contribution rather than enabling everything in
every experiment.

Recommended ablation sequence:

1. No augmentation
2. SpecAugment only
3. SpecAugment plus mild speed perturbation
4. SpecAugment plus real background noise and room impulse responses
5. The best combination from the preceding runs

Prefer real deployment-matched noise such as traffic, office noise, crowd
babble, fans, phone compression, and reverberation over Gaussian noise alone.
Keep pitch shifting conservative because aggressive shifts can distort useful
phonetic information.

## 11. Tune decoding separately

After training, tune decoding on the validation set rather than the test set.
Compare:

- Greedy decoding
- Beam sizes 3, 5, and 8
- Temperature zero
- Length and repetition penalties
- Explicitly forced Sinhala transcription
- Initial prompts containing verified domain terminology
- Previous-text context for adjacent segments of long recordings

Freeze the winning decoding parameters before the final test evaluation.

## Recommended experiment order

1. Freeze a speaker-disjoint test set and canonical normalizer.
2. Clean labels and eliminate duplicates and misaligned samples.
3. Reproduce the `3e-5` full-fine-tune baseline.
4. Sweep `1e-5`, `2e-5`, and `3e-5` with cosine scheduling.
5. Run augmentation ablations.
6. Train Whisper medium using the winning recipe.
7. Add a high-quality domain-specific adaptation stage.
8. Add carefully filtered pseudo-labeled Sinhala speech.
9. Tune decoding and contextual prompting.
10. Explore custom losses or larger architectural changes only after these
    experiments.

An initial target of 13-15 percent normalized WER is reasonable to test through
data cleaning, consistent labeling, controlled optimization, and decoding.
Moving substantially lower will likely require more diverse and accurately
transcribed Sinhala speech from unseen speakers and the deployment domain.
