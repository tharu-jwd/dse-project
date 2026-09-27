# Register normalization of the training data — approach and findings

This documents the work done on `Yohan2003/whisper-sl-data` (`stratified_v4`)
after run5's error analysis, targeting issues #1 and #2 from
`IMPROVEMENT_PLAN.md`: colloquial/formal register inconsistency and
word-boundary/ZWJ-conjunct inconsistency in the training transcripts. The end
result of this work is a new dataset version, **`stratified_v5`**, published
on Hugging Face (`Yohan2003/whisper-sl-data`, `data/stratified_v5/`), which is
`stratified_v4` with both corrections applied. `stratified_v4` itself was left
untouched so it stays reproducible for anything that already references it
(including run5's own training config).

## Why this happens: Sinhala has two registers, and transcribers don't agree on which one to write

Sinhala is a classic **diglossic** language — spoken Sinhala and written
Sinhala are close to two different registers of the same language, not just
"casual vs. formal tone" the way English has. Words routinely have a short,
elided spoken form and a longer, fuller written form for the exact same
meaning, with no difference in what's being said:

- `කරන්නෙ` (spoken) vs. `කරන්නේ` (written) — "does/is doing"
- `කියල` (spoken) vs. `කියලා` (written) — "having said/that"
- `තුල` vs. `තුළ` — dental vs. retroflex "within", a sound distinction that
  gets flattened in casual speech but is supposed to be kept in writing
- `සමග` vs. `සමඟ` — plain vs. prenasalized "with"

When people transcribe spoken audio, they constantly have to decide, word by
word, "do I write what was literally said, or do I write it the way it would
appear in a book?" Different transcribers (and even the same transcriber on
different days) make that call differently. Across four source corpora with
no shared style guide, this produces a dataset where **the same word is
spelled two different ways depending on who transcribed that particular
sentence**, even though the audio sounds identical either way.

This is why run5's `confusions.txt` showed pairs like `සමග → සමඟ` **and**
`සමඟ → සමග` as separate errors in both directions — the model isn't
mishearing anything, it's correctly learning that this word has two "correct"
spellings in training, and it has no way to guess which one a given test
sentence expects.

## How this was approached

### Step 1 — find out how bad it actually is
Rather than assume the problem from the confusion list alone, the whole
corpus (~155,000 transcribed sentences across train/validation/test) was
pulled down and every known colloquial/formal word pair from run5's
confusions was counted. None of them were close to a clean majority — the
worst, `සමග`/`සමඟ`, was a near-exact 51/49 split, meaning the model gets
almost zero signal on which spelling to prefer for that word.

### Step 2 — a naive first attempt failed, and that mattered
The first idea was: find any two vocabulary words one character-edit apart
(one insertion/deletion/substitution) and treat frequent pairs as spelling
variants. This produced garbage, because Sinhala has huge numbers of short,
completely unrelated words that happen to be one letter apart —
`අපි` ("we") vs. `අපේ` ("our"), `ගැන` ("about") vs. `ගෙන` ("having taken").
Comparing every possible single-character swap flagged thousands of these,
drowning out the real spelling-inconsistency pairs. This approach was
abandoned.

### Step 3 — targeted pattern search instead
Instead of "any edit distance 1," the search was restricted to 6 **specific,
linguistically known** Sinhala register/orthography alternations:

| Pattern | What it represents |
|---|---|
| `ෙ → ේ` | spoken vs. written word-ending vowel sign |
| `ල → ළ` | plain vs. retroflex "l" |
| `ග → ඟ` | plain vs. prenasalized "g" |
| `ද → ඳ` | plain vs. prenasalized "d" |
| `ථ → ත` | archaic/santhi vs. modern spelling |
| word vs. word+`ා` | spoken-clipped vs. full written verb/noun aspect |

For every vocabulary word, its counterpart under each rule was generated and
checked against the actual corpus: does that counterpart also occur, often
enough (≥5 times) and in a large enough minority share (≥15%) to be a real
pattern rather than a stray typo? This produced 543 candidate pairs — far
more precise than the naive approach, but still not perfect (see below).

### Step 4 — manual review caught real mistakes the automated pattern missed
This is the important part. The pattern search can tell you "these two words
differ by exactly the kind of edit that Sinhala register variation is known
to produce," but it **cannot tell you whether the two words actually mean
the same thing**. A colleague's filter script did a first automated pass at
this (a hand-built dictionary of known distinct pairs, e.g. flagging that
infinitive-vs-agentive-noun pairs shouldn't merge), cutting 543 down to 475.
But going through the `add-ා-suffix` category by hand, sentence by sentence,
turned up more mistakes that no automated rule caught:

**`ගන්න` vs. `ගන්නා`** — the example that started this check. `ගන්න` is the
infinitive "to take"; `ගන්නා` is a different word, "the one who takes" (an
agentive noun). They are not two spellings of the same word — merging them
would be a real error, not a fix.

The same shape of mistake was found repeatedly once actual sentences were
pulled and read, rather than trusting the pattern match on its own:

- **`මේස`/`මේසා`** — "tables" vs. an archaic word meaning "to this extent." Unrelated.
- **`මිත්`/`මිථ්`** — fragments of `මිත්‍ර` ("friend") and `මිථ්‍යා` ("false/illusion"). Unrelated, and only looked like a pair because a ZWJ character split both words at the same point (see below).
- **`කාල`/`කාලා`** — "time" (as in `කාල සීමා`, "time limit") vs. a colloquial past form of "to eat." Unrelated.
- **`රාම`/`රාමා`** — a fragment shared by `ග්‍රාම` ("village"), `විශ්‍රාම` ("retirement"), `සංග්‍රාම` ("battle") vs. the proper name "Rama." Unrelated.
- **`සිර`/`සිරා`** — "imprisonment" vs. slang for "genuine/for real." Unrelated.
- **`කොල්ල`/`කොල්ලා`** — "loot/plunder" (as in the idiom `කොල්ල කනවා`) vs. "boy." Unrelated.
- **`කරුණ`/`කරුණා`** — "a fact/point" vs. "compassion." Unrelated.
- **`නන්ද`/`නන්දා`** — two different proper names, one masculine (`නන්ද කුමාරයා`, "Prince Nanda") and one feminine (`නන්දා මාලනී`, a well-known performer).
- **`පැවිද්ද`/`පැවිද්දා`** — same shape of mistake as `ගන්න`/`ගන්නා`: "ordination" (the state) vs. "the monk" (the person).
- **`හිඟ`/`හිඟා`** — "shortage/scarce" vs. "begging." Unrelated.
- **`මහත්ම`/`මහත්මා`** — kept as risky rather than dropped outright: `මහත්මා` ("Mr./gentleman") is fine, but `මහත්ම` also independently means "great/supreme" as an adjective (`මහත්ම සැප`, "the greatest happiness") — a second, real sense that a blanket merge would corrupt.
- **`එකිනෙක`/`එකිනෙකා`** — flagged as risky rather than a clean pair: `එකිනෙක` ("each other") is used for things, `එකිනෙකා` for people — a grammatical agreement distinction, not a spelling error.

One pair, **`බල්ල`/`බල්ලා`** ("dog," possibly a case/emphasis-form
difference rather than a spelling error), was flagged as lower-confidence
but kept in the final list at the project owner's discretion, rather than
dropped automatically — a reminder that not every borderline call has a
clean answer, and some were left as a judgment call rather than a rule.

### Why this step couldn't be skipped
The pattern-matching in Step 3 is necessary — it's what makes the search
tractable across a 57,800-word vocabulary — but it is fundamentally a
**heuristic over spelling shape**, not over meaning. It will always produce
some false positives, because Sinhala (like most languages) has short words
that coincidentally look like other words' variant forms. The lesson from
this session: **every pair that reached the final list only got there after
someone actually read real sentences using both words** — first via an
automated dictionary filter for known cases, then via manual sentence-level
review for cases the dictionary didn't anticipate. Skipping the manual pass
would have silently corrupted at least 12 words' worth of meaning across
several hundred training sentences.

A second, related failure mode surfaced during manual review: Sinhala's
zero-width joiner (ZWJ) is used to render conjunct consonant clusters like
`ක්‍ර` and `ග්‍ර`. A simple word-splitting regex that only recognizes the
Sinhala Unicode block (and not the ZWJ) breaks these conjunct words apart at
the joiner, so `ග්‍රාම` gets fragmented into `ග්` and `රාම`. Several of the
false-positive pairs above (`මිත්`/`මිථ්`, `රාම`/`රාමා`) were artifacts of
this fragmentation, not real words at all colliding by coincidence. This is
the same underlying issue as run5's issue #2 (word-boundary/conjunct
inconsistency) — it showed up here as a tooling bug, and as a data problem
there.

The register-pair list went through one more correction pass after this:
manual review turned up 407 confirmed pairs (543 candidates → 475 after the
automated dictionary filter → 413 after manual review → 407 after removing
`කොල්ල`/`කොල්ලා` plus the 6 pairs the project owner asked to drop:
`කරුණ`, `එකිනෙක`, `නන්ද`, `පැවිද්ද`, `හිඟ`, `මහත්ම` — `බල්ල`/`බල්ලා` was kept
despite being flagged as lower-confidence, per the project owner's call).

## Step 5 — the same problem exists at the word-boundary level, not just spelling

Issue #2 from `IMPROVEMENT_PLAN.md` (word-boundary/conjunct inconsistency) was
tackled the same way, separately from the register work. The method: tokenize
every sentence by whitespace (not by Sinhala-Unicode-block, which breaks on
ZWJ — see Step 6), then look for adjacent word pairs whose concatenation
matches an existing single word elsewhere in the corpus (e.g. `ක්‍රියා කළ`
next to each other vs. `ක්‍රියාකළ` written as one word somewhere else in the
data). This found 62 candidate pairs, of which 2 were excluded as genuinely
ambiguous by Sinhala compounding convention (`ශ්‍රී ලංකන්`, where `ශ්‍රී` is
conventionally kept separate as an honorific; `ප්‍රධාන පෙලේ`, an ordinary
adjective+noun phrase rather than a fixed compound) — leaving **60 confirmed
word-boundary pairs**. Unlike the register pairs, these were resolved by
compounding-rule knowledge rather than a full sentence-by-sentence read of
all 60; the correct direction in every remaining case was "glued," matching
what run5's own reference data already showed (`ref: ක්‍රියාකළ යුතුයි.` was
the correct glued form; `pred: ක්‍රියා කළ` was the model's mistake).

## Step 6 — a chaining bug found during final verification

Before pushing anything, the applied correction was checked against the real
data by searching for any remaining occurrence of a "wrong" form. This found
19 leftover instances across 4 words, all traced to the same cause: a few
correction pairs chain into each other (`ලගට → ලඟට`, but `ලඟට` was *itself*
listed elsewhere as needing to become `ළඟට`). Since the correction is a
single dictionary lookup per word, chained fixes only applied halfway. Fixed
by resolving the mapping to its transitive closure (following each chain to
its final form) before reapplying. A second re-verification after the fix
found genuine zero remaining errors — the one flagged instance left
(`යවල`, 1 occurrence) turned out to be a false alarm in the verification
script itself, not a real error: the text `මාධ්‍යවල` ("media", correctly
glued by the Step 5 fix) contains a ZWJ character that the verification
script's own regex doesn't treat as part of a word, so it misread a
fragment of the correct compound as a match for an unrelated register pair.
Ironic, given this is exactly the class of bug Step 5 exists to fix in the
data itself — it just happened to also exist in the checking code.

## Step 7 — checking for a bigger problem before trusting run5's WER number at all

While pulling example sentences, the same sentence text was noticed appearing
in more than one split (test, validation, *and* train for at least one
example). Since `stratified_v4` is supposed to be speaker-disjoint, this was
checked properly: every row's audio was SHA-256 hashed, split by split, and
cross-referenced against every sentence with duplicate text. Result: **16,079
sentences (~10% of the corpus) share identical text across train and
test/validation, but in every single case the audio hash is different** —
zero instances of the same audio file appearing in more than one split. This
is a fixed prompt list read by different speakers (common in read-speech
corpora, not a defect), not audio leakage. Speaker-disjoint splitting held up
as advertised, and run5's WER isn't inflated by the model having literally
heard test audio during training.

## Step 8 — applying the correction to the real dataset and publishing it

The register and word-boundary corrections were combined into one pipeline
and applied for real, to the actual `stratified_v4` parquet files (audio
included, not just the extracted text), run on a GCP VM rather than locally
because of the local connection's slow bandwidth (~400KB/s vs. the VM's much
faster link to Hugging Face). The pipeline streams each split row-group by
row-group (never holding a whole file in memory), rewrites only the `text`
column, and leaves `audio`/`source_dataset` untouched.

Final result, applied to the real dataset:

| Split | Rows | Rows changed |
|---|---|---|
| test | 15,860 | 1,325 |
| validation | 15,763 | 1,412 |
| train | 123,205 | 10,383 |
| **Total** | **154,828** | **13,120 (~8.5%)** |

This was verified twice: once against the VM's local output, and again by
pulling the data back down fresh from Hugging Face after publishing, to rule
out any corruption in transit. Both checks agree: zero genuine remaining
errors from either correction set.

The result was published as a new dataset path, **`stratified_v5`**
(`Yohan2003/whisper-sl-data`, `data/stratified_v5/{train,validation,test}.parquet`),
rather than overwriting `stratified_v4` in place, so that run5's own training
config (which points at `stratified_v4`) and anything else already built
against v4 stay reproducible.

## A related discovery made along the way, not yet acted on

While looking for disk space on the training VM to do this work, a separate,
much larger cleanup effort was found already sitting there from before this
session: a manually-reviewed correction set of roughly **18,000 word/phrase
fixes** (two independent reviewers, cross-checked for conflicts) plus a
ZWNJ-based (a related but different invisible character from the ZWJ this
document focuses on) missing-space fix, which was used to build
`stratified_v2` from `stratified_v1`. This predates `stratified_v4` (which
was built later, in an earlier attempt before this session) and predates all
the work described above. It raises an open question — worth investigating
before assuming the current correction lists are the last word on data
quality — of whether anything in the `v2 → v3 → v4` transition dropped or
regressed corrections that earlier pass had already made.

## What's still open
- The `add-ා-suffix` register category (80 pairs) is the only one that
  received a full sentence-by-sentence manual read. The other ~330 register
  pairs (`ෙ→ේ`, `ල→ළ`, `ග→ඟ`, `ද→ඳ`, `ථ→ත`) are lower-risk by construction
  (pure single-character phonetic substitutions with no competing
  grammatical meaning) but haven't had the same sentence-level check.
- Similarly, the 60 word-boundary pairs were resolved by compounding-rule
  reasoning, not a full sentence-by-sentence read the way the risky register
  categories were.
- The pre-existing 18,000-correction pipeline (see above) hasn't been
  cross-checked against what actually made it into `stratified_v4`/`v5`.
- Issues #3 (underrepresented honorific/religious register) and #4 (rare
  proper nouns/abbreviations) from `IMPROVEMENT_PLAN.md` are acoustic/data-
  volume problems, not text-cleanup problems, and haven't been started.
- A dictionary/spellchecker pass to catch words that are *consistently*
  misspelled everywhere (not just inconsistently split between two existing
  spellings) hasn't been done — this method is structurally blind to that
  class of error, as demonstrated by `ප්‍රථිපල`/`ප්‍රතිපල`, both of which
  turned out to be typos of the correct word `ප්‍රතිඵල`, which appears
  nowhere in the corpus.
