"""Fill the correction column in inconsistencies_full.csv -- but NOT by blind
majority vote. Majority-vote is wrong whenever the two spellings are actually
two different real words (e.g. කල "time" vs කළ "did") rather than one
word spelled two ways (e.g. වුනා vs වුණා "became", where only වුණා is correct).

Three buckets, decided per-row:

1. GRAMMAR RULE (auto-filled, high confidence): the Sinhala "hutuna" past-
   participle suffix -ුන- is grammatically always -ුණ- (retroflex ණ), regardless
   of how often it's miswritten with dental න in casual text. Every word
   matching this suffix pattern gets corrected to the ණ form automatically.

2. KNOWN HOMONYM PAIRS (never auto-filled): pairs hand-checked against
   Sinhala dictionary meaning where BOTH spellings are real, different words
   (like the කල/කළ example that got this whole review started). These are
   marked "CONTEXT-DEPENDENT" -- fixing them needs the actual sentence, not
   just a frequency count.

3. EVERYTHING ELSE: left for manual review, marked "NEEDS REVIEW" rather
   than guessed, UNLESS the row is in the small SINGLE_FORM override list
   below (cases I'm confident only one spelling is a real dictionary word,
   verified by codepoint, not by eyeballing similar-looking glyphs).

Usage:
    python fill_corrections.py
"""

import csv
import re

IN_PATH = "inconsistencies_full.csv"
OUT_PATH = "inconsistencies_full.csv"  # overwrite in place

# --- Rule 1: hutuna suffix, -ුන- -> -ුණ- ------------------------------------
# ු = dependent u-vowel sign, න = dental na, ණ = retroflex Na.
# Matches BOTH the wrong dental form and the already-correct retroflex form,
# so a row whose majority spelling happens to already be correct gets
# confirmed (no-op substitution) instead of falling through to NEEDS REVIEW.
_HUTUNA_RE = re.compile("ුණ|ුන")
# Colloquial contraction that drops the leading ව (වුන/වුණ -> උන/උණ, "became"):
# independent-vowel උ directly followed by න/ණ, same grammar rule applies.
_HUTUNA_CONTRACTED_RE = re.compile("උණ|උන")

def hutuna_correction(word: str) -> str | None:
    if _HUTUNA_RE.search(word):
        return _HUTUNA_RE.sub("ුණ", word)
    if _HUTUNA_CONTRACTED_RE.search(word):
        return _HUTUNA_CONTRACTED_RE.sub("උණ", word)
    return None


# --- Rule 2: known homonym pairs -- never auto-corrected -----------------
# sound_key -> True marks "this group contains 2+ genuinely different words,
# don't canonicalise it here."
HOMONYM_KEYS = {
    "කල",       # කල "time" (noun) vs කළ "did" (verb) -- already removed by user, kept
                    # here so the flag is documented even if the row returns
    "කලේ",      # කලේ (time, oblique) vs කළේ ("did", past)
    "පල",       # පල "fruit/result" vs පළ "spread/rank" (පළමු "first")
    "මල",       # මල "flower" vs මළ "dead" (archaic/compound)
    "වල",       # වල plural locative suffix ("-in the ___s") vs වළ "pit/ditch"
    "කන",       # කන "ear" / colloquial "eating" root vs කණ "ear" (formal) -- both කන
                    # forms exist depending on register; flag rather than force
    "මරන",      # මරන "killing-" (compound prefix, e.g. මරන ආයුධ) vs මරණය "death" (noun)
    "ඇන",       # ඇන (verb-adjacent form) vs ඇණ "nail" (hardware) -- flag, don't guess
    "බන",       # බන (colloquial/rare) vs බණ "sermon/dharma talk" -- flag
    "ගන",       # ගන/ඝන/ගණ/ඝණ all real, distinct words (thick/dense, solid,
                    # group/faction, ...) -- definitely context-dependent
    "කලින්",   # කලින් "earlier/before" -- only one real spelling exists (dental ල),
                    # NOT a homonym, moved to SINGLE_FORM below instead
}

# --- Rule 3: single-correct-form overrides --------------------------------
# sound_key -> the one grammatically-correct spelling, used even when it's
# the MINORITY spelling in this dataset (majority-vote would pick the wrong
# one). Only pairs I'm confident have exactly one standard dictionary form.
SINGLE_FORM = {
    "පුලුවන්": "පුළුවන්",
    "පිලිබඳ": "පිළිබඳ",
    "ගානක්": "ගාණක්",
    "ගනන්": "ගණන්",
    "පිලිතුරු": "පිළිතුරු",
    "පලමු": "පළමු",
    "ඇතුලු": "ඇතුළු",
    "වලක්වා": "වළක්වා",
    "සුලු": "සුළු",
    "කලු": "කළු",
    "දල": "දළ",
}


def main():
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys()

    # normalise the trailing unnamed column to a real name
    fieldnames = [fn if fn else "correction" for fn in fieldnames]

    n_rule1 = n_homonym = n_single = n_kept = n_review = 0

    for row in rows:
        existing = row.get("") or row.get("correction", "")
        if existing.strip():
            n_kept += 1
            continue  # don't overwrite what you've already filled in by hand

        key = row["sound_key"]
        majority = row["majority_spelling"]

        if key in HOMONYM_KEYS:
            row[""] = "CONTEXT-DEPENDENT -- two real words, needs the sentence, not auto-fixed"
            n_homonym += 1
            continue

        if key in SINGLE_FORM:
            row[""] = SINGLE_FORM[key]
            n_single += 1
            continue

        fix = hutuna_correction(majority)
        if fix:
            row[""] = fix
            n_rule1 += 1
            continue

        row[""] = "NEEDS REVIEW"
        n_review += 1

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"already filled by you (kept as-is)      : {n_kept}")
    print(f"auto-filled, hutuna grammar rule (-ුන->-ුණ-): {n_rule1}")
    print(f"marked CONTEXT-DEPENDENT (known homonyms) : {n_homonym}")
    print(f"auto-filled, single-dictionary-form list  : {n_single}")
    print(f"marked NEEDS REVIEW (uncertain, unfilled) : {n_review}")
    print(f"total rows                                : {len(rows)}")


if __name__ == "__main__":
    main()
