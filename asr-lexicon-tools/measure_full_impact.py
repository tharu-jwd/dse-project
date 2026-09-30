"""Combined impact of ALL FOUR resolved correction files on run6's actual
test predictions -- letter-level spelling (inconsistencies_full.csv),
ZWJ/conjunct (zwj_conjunct_inconsistencies.csv), word-level register pairs
(register_pairs_ruled.csv), and confirmed English typos
(reference_confirmed_typos.csv) -- applied together, on top of punctuation
normalization, same methodology used throughout this session.

Usage:
    python measure_full_impact.py
"""
import csv
import jiwer

PRED = "../../ErrorAnalysis/run6-v5-lr3e-5-bs64/run_summary/predictions.csv"
PUNCT = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

word_map = {}

def add_spellings(spellings_field, correction):
    for part in spellings_field.split(";"):
        if not part:
            continue
        word = part.rsplit(":", 1)[0]
        word_map[word] = correction

# 1. letter-level spelling
n1 = 0
with open("inconsistencies_full.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = row["correction"].strip()
        if not c:
            continue
        word_map[row["majority_spelling"]] = c
        for part in row["minority_spellings"].split(";"):
            if part:
                word_map[part.rsplit(":", 1)[0]] = c
        n1 += 1

# 2. ZWJ/conjunct
n2 = 0
with open("zwj_conjunct_inconsistencies.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = row["correction"].strip()
        if not c:
            continue
        add_spellings(row["spellings_seen"], c)
        n2 += 1

# 3. word-level register pairs (only rows resolved "yes" with a canonical form)
n3 = 0
with open("register_pairs_ruled.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["same_word_different_register"].strip().lower() != "yes":
            continue
        c = row["canonical_form"].strip()
        if not c:
            continue
        add_spellings(row["spellings_seen"], c)
        n3 += 1

# 4. confirmed English typos (case-insensitive on the Latin token)
n4 = 0
with open("reference_confirmed_typos.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        word_map[row["typo"]] = row["correction"]
        n4 += 1

print(f"loaded: {n1} letter rows, {n2} ZWJ rows, {n3} register rows, {n4} typo rows")
print(f"total word-level replacements: {len(word_map):,}")
print()

def apply_map(t):
    return " ".join(word_map.get(w, w) for w in t.split())

refs, hyps = [], []
with open(PRED, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        refs.append(row["reference"]); hyps.append(row["prediction"])

raw_wer = jiwer.wer(refs, hyps)

p_refs = [PUNCT(r) for r in refs]; p_hyps = [PUNCT(h) for h in hyps]
punct_wer = jiwer.wer(p_refs, p_hyps)

c_refs = [apply_map(r) for r in p_refs]; c_hyps = [apply_map(h) for h in p_hyps]
full_wer = jiwer.wer(c_refs, c_hyps)

def totals(rs, hs):
    out = jiwer.process_words(rs, hs)
    sub = sum((a.ref_end_idx-a.ref_start_idx) for al in out.alignments for a in al if a.type=="substitute")
    d = sum((a.ref_end_idx-a.ref_start_idx) for al in out.alignments for a in al if a.type=="delete")
    i = sum((a.hyp_end_idx-a.hyp_start_idx) for al in out.alignments for a in al if a.type=="insert")
    return sub, d, i, sub+d+i

t_raw = totals(refs, hyps)
t_punct = totals(p_refs, p_hyps)
t_full = totals(c_refs, c_hyps)

print(f"raw WER                          : {raw_wer*100:.2f}%")
print(f"+ punctuation normalized         : {punct_wer*100:.2f}%")
print(f"+ all 4 correction sets applied  : {full_wer*100:.2f}%")
print()
print(f"total gap (raw -> full)          : {(raw_wer-full_wer)*100:.2f} pts ({(raw_wer-full_wer)/raw_wer*100:.1f}% of raw errors)")
print(f"gap from corrections alone (punct-normalized -> full): {(punct_wer-full_wer)*100:.2f} pts")
print()
print(f"raw total word errors  : {t_raw[3]:,}")
print(f"punct-normalized errors: {t_punct[3]:,}  (-{t_raw[3]-t_punct[3]:,} from punctuation)")
print(f"full-corrected errors  : {t_full[3]:,}  (-{t_punct[3]-t_full[3]:,} from the 4 correction sets)")
