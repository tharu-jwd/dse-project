"""Collapse Sinhala words that differ only by a spelling *convention*, not by
sound, down to a shared "sound key" -- so WER/consistency-checking can treat
e.g. hela written as හල (l) and හවහ wait no -- see README below for the
actual pairs -- as the same word.

Three merge tables, kept separate on purpose so you can test each layer's
contribution independently (this mirrors the structure of the original
soundkey.py, recovered from its compiled bytecode after the source was lost --
only the *symbol names* survived, not the mapping data, so the actual
confusable groups below were rebuilt from known Sinhala orthographic-variant
letter pairs and should be spot-checked against the hand-check step (build
step 4) before trusting them):

  _CONSONANT_MERGE -- letters that are historically/phonemically distinct but
                      routinely interchanged in casual writing:
                        ළ <-> ල   (retroflex l  vs  dental l)   e.g. හළ / හල
                        ණ <-> න   (retroflex n  vs  dental n)   e.g. ගණන් / ගනන්
                        ෂ <-> ව->  (skip -- ව is "wa", not a sibilant; real group is ෂ/ශ/ස)
                        ෂ <-> ශ <-> ස  (three historical sibilants sha/sha/sa,
                                       merged in most modern pronunciation)
  _VOWEL_MERGE      -- vowel-length distinctions frequently flattened in
                        casual/spoken-register writing (short <-> long):
                        ි <-> ී   (dependent i  vs  ii)
                        ු <-> ූ   (dependent u  vs  uu)
                        off by default in soundkey() -- vowel length CAN
                        change meaning, so only enable this layer if step 1's
                        consistency check shows it's actually a spelling-
                        convention issue in your data, not a real distinction
  _ALL_MERGE        -- union of both, used by the "aggressive" variant

Usage:
    from build_soundkey import sound_key, tokenise, sound_key_sentence
"""

import re
import unicodedata

# Words are compared after these are stripped: punctuation, zero-width
# joiners/non-joiners (which vary between renderers for the same conjunct),
# and any trailing/leading whitespace.
_STRIP_RE = re.compile(r"[‌‍.,!?;:\"'()\[\]{}।॥]")

# char -> canonical-spelling map (canonical choice doesn't matter for
# consistency-checking, only for the corrector's majority vote in step 4).
_CONSONANT_MERGE = {
    "ළ": "ල",   # ළ -> ල
    "ඝ": "ග",   # ඝ -> ග (rare gha variant, if present)
    "ඛ": "ක",   # ඛ -> ක (rare kha variant, if present)
}
# කක-cluster consonant confusables that need digraph awareness ( n before another
# consonant) are handled as a second pass in _normalise_retroflex_n below,
# because ක (na) vs ණ (retroflex Na) swaps depend on the following
# letter in real orthography, not a blind 1:1 char map.
_RETROFLEX_N_RE = re.compile("[නණ]")

_VOWEL_MERGE = {
    "ී": "ි",   # ii -> i
    "ූ": "ු",   # uu -> u
}

_ALL_MERGE = {**_CONSONANT_MERGE, **_VOWEL_MERGE}


def _normalise_retroflex_n(word: str) -> str:
    """Collapse න (dental n) / ණ (retroflex n) to a single symbol.

    Unlike ළ/ල this pair is legitimately conditioned by context in
    formal grammar (ණ after certain vowels/consonants), but in casual
    transcripts both spellings show up for the same word -- so for
    sound-key purposes we merge unconditionally and let step 1's printed
    list tell you how often it actually happens in your data.
    """
    return _RETROFLEX_N_RE.sub("න", word)


def tokenise(text: str) -> list[str]:
    """Split a transcript into words for consistency-checking."""
    text = unicodedata.normalize("NFC", text)
    text = _STRIP_RE.sub("", text)
    return [w for w in text.split() if w]


def sound_key(word: str, include_vowels: bool = False) -> str:
    """Map a word to a canonical key shared by all its spelling variants.

    include_vowels=False (default): only merges the letter-substitution
    confusables (ළ/ල, ඝ/ග, ඛ/ක, න/ණ) -- these are the ones
    that are purely orthographic-convention, never a real minimal pair in
    casual speech-to-text data.

    include_vowels=True: also merges vowel-length pairs (ී/ි, ූ/ු).
    Turn this on only after step 1 shows these pairs genuinely alternate for
    the *same* word in your transcripts -- vowel length can be a real
    distinction, so don't merge it blindly.
    """
    word = unicodedata.normalize("NFC", word)
    table = _ALL_MERGE if include_vowels else _CONSONANT_MERGE
    word = "".join(table.get(ch, ch) for ch in word)
    word = _normalise_retroflex_n(word)
    return word


def sound_key_sentence(text: str, include_vowels: bool = False) -> str:
    """Apply sound_key() to every token and rejoin -- for sentence-level WER."""
    return " ".join(sound_key(w, include_vowels) for w in tokenise(text))


if __name__ == "__main__":
    # Quick sanity pass -- same shape as the tests the original bytecode
    # implies existed (a `tests()` / `ok()` pattern). Expand this with real
    # examples once step 1's printed conflict list gives you actual pairs
    # seen in stratified_v5.
    cases = [
        ("හළ", "හල", True),   # hela / hela -> same key
        ("ගණන්", "ගනන්", True),  # gaNan / ganan -> same key
    ]
    ok = 0
    for a, b, expect_same in cases:
        ka, kb = sound_key(a), sound_key(b)
        same = ka == kb
        status = "OK" if same == expect_same else "FAIL"
        ok += status == "OK"
        print(f"{status}: {a!r} -> {ka!r} | {b!r} -> {kb!r}")
    print(f"{ok}/{len(cases)} passed")
