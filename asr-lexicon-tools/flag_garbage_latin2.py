"""Refined version: instead of a flat dictionary-miss check (too many false
positives from brand names, acronyms, and valid words missing from a 104k
general-English wordlist), this prioritizes tokens that look like TYPOS of a
real word -- edit-distance 1-2 from a common English word -- over tokens
that are just unusual but plausible as-is (proper nouns, acronyms, tech
terms). Typo-shaped unrecognized tokens are far more likely to be genuine
transcription garbage than "unrecognized because it's a brand name."

Usage:
    python flag_garbage_latin2.py
"""
import csv
import re

IN_PATH = "reference_quality_flags.csv"
OUT_LIKELY_TYPO = "reference_likely_typos.csv"
OUT_UNKNOWN = "reference_unknown_latin.csv"
DICT_PATH = "/usr/share/dict/words"

EXTRA_ALLOWED = {
    "ai", "app", "apps", "wordpress", "google", "youtube", "instagram",
    "facebook", "tiktok", "adsense", "https", "http", "url", "dna", "rna",
    "trna", "covid", "wifi", "usb", "pdf", "id", "ok", "okay", "gpt",
    "chatgpt", "chatbots", "chatbot", "ios", "android", "api", "html",
    "css", "js", "seo", "aeo", "ssd", "cv", "idh", "mics", "cafe",
    "meetup", "pentose", "anyways", "impactful", "conversational",
}

_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def load_dictionary():
    common = set()
    with open(DICT_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            common.add(w)
    return common


def levenshtein(a, b, cap=2):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def main():
    dictionary = load_dictionary()
    # index by first letter + rough length for a fast typo-neighbour lookup
    by_bucket = {}
    for w in dictionary:
        if len(w) < 3:
            continue
        for L in (len(w) - 1, len(w), len(w) + 1):
            by_bucket.setdefault((w[0], L), []).append(w)

    def is_typo_of_real_word(tok):
        low = tok.lower()
        candidates = by_bucket.get((low[0], len(low)), [])
        return any(levenshtein(low, c, 2) <= 1 for c in candidates[:2000])

    typo_rows, unknown_rows = [], []
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row["text"]
            tokens = _LATIN_TOKEN_RE.findall(text)
            typo_tokens, unknown_tokens = [], []
            for tok in tokens:
                low = tok.lower()
                if len(tok) < 3 or low in dictionary or low in EXTRA_ALLOWED:
                    continue
                if is_typo_of_real_word(tok):
                    typo_tokens.append(tok)
                else:
                    unknown_tokens.append(tok)
            if typo_tokens:
                typo_rows.append([row["split"], text, ";".join(typo_tokens)])
            elif unknown_tokens:
                unknown_rows.append([row["split"], text, ";".join(unknown_tokens)])

    with open(OUT_LIKELY_TYPO, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "text", "likely_typo_tokens"])
        writer.writerows(typo_rows)

    with open(OUT_UNKNOWN, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "text", "unknown_tokens"])
        writer.writerows(unknown_rows)

    print(f"likely genuine typos/garbage : {len(typo_rows):,} -> {OUT_LIKELY_TYPO}")
    print(f"unknown but not typo-shaped  : {len(unknown_rows):,} -> {OUT_UNKNOWN}")
    print("(unknown bucket = probably proper nouns/brands/rare-but-valid terms,")
    print(" lower priority for review than the likely-typo bucket)")


if __name__ == "__main__":
    main()
