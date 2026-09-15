# How voice commands are recognised (and what's currently broken)

Written 2026-09-15, from a real testing session of 66 spoken utterances captured in the backend log.

This file answers two questions:

1. **When I say "zimi makanna", does Whisper actually write down "zimi"?**
   → Mostly **no**. It appeared 35% of the time. Details in §3.
2. **Can we just drop the fuzzy text matching and use only the voice embeddings?**
   → **No, that would be unsafe.** Measured evidence in §4.

---

## 1. There are two separate recognisers, not one

When you speak a command, the backend runs **two completely independent checks** and then combines them.

```
your voice
    │
    ├──► [1] Whisper writes down words ──► compare the SPELLING  ──► score 0 to 100
    │                                       against each command
    │
    └──► [2] audio turned into numbers ──► compare the SOUND     ──► score 0.0 to 1.0
                                            against your recordings
```

A log line shows both:

```
voice_command_decision transcript='ඊළඟට' outcome=execute command=next
                       fuzzy=None(None)  embedding=next(0.851)
                       ─────┬──────────   ────────┬───────────
                    check 1: spelling      check 2: sound
                    found nothing          matched "next"
```

**Neither number is a probability.** `0.851` does not mean "85% sure". They are two different
measuring sticks, described below.

---

## 2. What the two numbers actually measure

### Check 1 — the spelling score (0 to 100)

This compares **letters**, nothing else. It does not know anything about sound.

First the text is simplified: Sinhala vowel signs and spaces are removed, so `දෙක` becomes `දක`.
Then it counts how many letters the two words have in common:

```
score  =  (2 × letters in common)  ÷  (total letters in both words)  × 100
```

Example — you said "eka", Whisper wrote `එක`, and the command is stored as `zimi එක`:

```
what you said :  එක        →  2 letters
what's stored :  zimiඑක    →  6 letters
in common     :  එ, ක      →  2 letters

score = 2 × 2 ÷ (2 + 6) × 100 = 50
```

The command only fires if this reaches **80** (or **90** for `delete` and `submit`).
50 is far below 80, so it was rejected — even though you said the word perfectly.

### Check 2 — the sound score (0.0 to 1.0)

This ignores spelling entirely. The audio is fed through a model that turns the sound into a long
list of numbers — think of it as coordinates describing "what this utterance sounds like". Your
enrolled recordings were turned into coordinates the same way and saved.

The score is how closely the two sets of coordinates point in the same direction:

- `1.0` = identical direction
- `0.0` = completely unrelated

It fires at **0.828**, and is treated as "maybe" from **0.74**.

Because this one listens to sound rather than reading spelling, it kept working while the spelling
check was broken. **The sound check has been carrying the whole system.**

---

## 3. Does "zimi" get transcribed? — Mostly not

Out of 66 utterances in the test session:

| What Whisper wrote for the wake word | Count | Share |
|---|---|---|
| Latin letters — `zimi` | 23 | **35%** |
| Sinhala letters — `සිමි` | 2 | 3% |
| Nothing at all — the word vanished | 41 | **62%** |

*(Caveat: some of those 41 were ordinary talking, not failed commands — so 62% overstates the true
drop rate somewhat. But the pattern is unmistakable.)*

### Why it vanishes

`zimi` is written in **Latin letters inside a Sinhala sentence**. Whisper is running in Sinhala
mode, so it expects Sinhala script. Faced with a sound that isn't a Sinhala word, it does one of
three things — and you can see all three in the log:

```
'zimi සුරකින්න'    ← wrote it in Latin      (works)
'සිමි මකන්න'       ← wrote it in Sinhala    (spelling check fails - stored form is Latin)
'ඊළඟට'             ← dropped it completely  (spelling check fails - 4 letters missing)
```

### The second, separate problem: the command word gets cut off

**10 of the 23** utterances where `zimi` *was* captured contained **only** the word `zimi` and
nothing else:

```
transcript='zimi'  →  fuzzy=option_1(80.0)  embedding=wake(0.857)
```

This happens because the system cuts a recording at a pause. If you pause even slightly between
"zimi" and "makanna", it is split into two separate recordings — the first holds only "zimi", and
the command word lands in a second recording that no longer has the wake word. Both halves then
fail.

Worse, `'zimi'` on its own scores **80.0** against `zimi එක` (option_1) — exactly the firing
threshold. **Saying just the wake word can trigger option_1 by itself.**

### Why this hits `next`, `eka` and `deka` hardest

When `zimi` is dropped, 4 letters go missing from the comparison. Short commands lose a much larger
share of themselves than long ones:

| You said | Stored as | Score | Passes 80? |
|---|---|---|---|
| `එක` (eka) | `zimiඑක` | **50** | no |
| `දක` (deka) | `zimiදක` | **50** | no |
| `ඊළඟට` (next) | `zimiඊළඟට` | **66.7** | no |
| `සුරකින්න` (save) | `zimiසරකනන` | 76.9 | no, but close |

`eka` and `deka` are only 2 letters long, so losing 4 letters is catastrophic. That is precisely
why those are your worst three.

### And a third problem: eka and deka now sound alike *to the spelling check*

With the shared prefix attached, the two commands became nearly identical on paper:

```
zimiඑක  vs  zimiදක      5 of 6 letters match   →  score 83
එක      vs  දක          1 of 2 letters match   →  score 50
```

83 is **above** the firing threshold of 80. So even when everything transcribes correctly, `eka`
and `deka` are only 17 points apart, and one mistyped letter flips the answer to the wrong command.
Without the prefix they sit a comfortable 50 points apart.

---

## 4. Can we use only the embeddings and drop the fuzzy matching?

**No.** The sound check on its own cannot tell a command from ordinary talking.

Here are the sound scores from the session, split by whether the spelling check also recognised
something:

| | count | lowest | middle | highest |
|---|---|---|---|---|
| Real command attempts | 19 | 0.792 | 0.847 | 0.911 |
| Ordinary talking | 43 | 0.767 | **0.820** | **0.917** |

**The two ranges overlap almost completely.** Ordinary talking reached **0.917** — higher than any
genuine command in the session. There is no threshold you can draw that separates them.

### What that means in practice

Today, a sound-only match that is uncertain produces a *confirmation prompt*, and the spelling
check acts as a second opinion. If you removed the spelling check, those would become **commands
that just execute**. Taking the 24 ordinary multi-word sentences from the session and applying the
0.828 firing threshold:

```
0.841  →  would fire "previous"  on: 'අපි ආපසු'
0.839  →  would fire "option_3"  on: 'මම උදේට නැගිටිනවා උදේ හතට විතර'
0.841  →  would fire "submit"    on: 'දිරිපත් කරන්න'      ← DESTRUCTIVE
0.829  →  would fire "submit"    on: 'දිරිපත් කරන්න'      ← DESTRUCTIVE
0.854  →  would fire "submit"    on: 'ඉදිරිපත් කරන්න'     ← DESTRUCTIVE
0.831  →  would fire "submit"    on: 'විපත් කරන්න'        ← DESTRUCTIVE
0.904  →  would fire "delete"    on: 'සිමි මකන්න'         ← DESTRUCTIVE
0.917  →  would fire "stop"      on: 'සිමි නවත්වන්න'
```

**8 of 24 ordinary sentences would have fired a command. 5 of those were `delete` or `submit`** —
the two actions that destroy a student's work.

`'මම උදේට නැගිටිනවා උදේ හතට විතර'` means "I get up around seven in the morning." A student saying
that while dictating a note would have silently selected multiple-choice option 3.

The whole reason two checks exist is that each covers the other's blind spot:

- the **sound** check doesn't care how badly Whisper spells things
- the **spelling** check doesn't fire on a sentence that merely *sounds* a bit like a command

Removing either one removes that protection. The answer is to **fix the spelling check**, not to
delete it.

### A related bug found in the same logs

There are 5 leftover recordings in the database for a command called `wake`, from back when "zimi"
was its own separate command. That command no longer exists, but the recordings were never deleted,
and they are still being loaded and matched — **18 matches in this session, scoring up to 0.878**.
The backend then sends the frontend a command named `wake`, which nothing handles, so it silently
does nothing. These rows should be deleted.

---

## 5. What should be done

In priority order:

1. **Stop putting the wake word inside the spelling comparison.** Check for it separately as a
   yes/no, strip it off, then compare only the real command word. This fixes every case in §3 at
   once: `එක` scores 100 instead of 50 whether or not `zimi` survived, and `eka` vs `deka` goes back
   to a 50-point gap instead of 17.
2. **Accept more than one spelling of the wake word** — at minimum Latin `zimi` and Sinhala `සිමි`,
   since Whisper demonstrably produces both.
3. **Delete the 5 orphaned `wake` recordings** from `command_enrollments`.
4. **Re-tune the sound threshold** against the overlap measured in §4 — 0.828 currently sits in the
   middle of the ordinary-talking range.
5. **Keep both checks.** Neither is reliable alone; this document is the evidence for why.

**Status (2026-09-15):** implemented as a different design than item 1 describes. Phrases are
bare words again, and "zimi" is detected on its own — by voice fingerprint (top match must be the
wake samples at ≥ 0.83), with transcript text (`zimi`/`zini`/`සිමි`) as the fallback. Detecting it
arms a 3-second window in which one command may run. Commands still use both checks. See
`_resolve_and_dispatch` in `backend/app/api/routes/streaming.py`.
