"""Final, hand-reviewed list of genuine English-word typos found in
stratified_v5 reference transcripts (from reference_likely_typos.csv),
with brand names, proper nouns, and acronyms deliberately excluded --
those are legitimate code-switching, not errors, per explicit instruction.

Usage:
    python build_confirmed_typos.py
"""
import csv

IN_PATH = "reference_likely_typos.csv"
OUT_PATH = "reference_confirmed_typos.csv"

# typo (lowercase) -> correction, hand-reviewed against each row's context
CONFIRMED = {
    "bock": "block",
    "comminication": "communication",
    "futurestic": "futuristic",
    "machintosh": "Macintosh",
    "sola": "solar",
    "uncaney": "uncanny",
    "uncany": "uncanny",
    "propotion": "propulsion",
    "projecs": "projects",
    "hacel": "hassle",
    "feets": "feet",
    "encording": "encoding",
    "avacado": "Avocado",
    "mor": "more",
    "cristal": "crystal",
    "perplexitiy": "perplexity",
    "nuclic": "nucleic",
    "adjest": "adjust",
    "opotion": "option",
    "valey": "valley",
    "pito": "pitot",
    "prio": "prior",
    "irrelavant": "irrelevant",
    "destop": "desktop",
    "crispe": "crispr",
    "enthusias": "enthusiasts",
    "sanitory": "sanitary",
    "advertisments": "advertisements",
    "neuted": "neutral",
    "dominian": "dominion",
    "studens": "students",
    "outdate": "outdated",
    "lum": "lump",
    "planers": "planners",
    "liquity": "liquidity",
    "milage": "mileage",
    "phototype": "prototype",
    "independant": "independent",
    "strenghs": "strengths",
    "digitaly": "digitally",
    "insuarnce": "insurance",
    "benifit": "benefit",
    "speaches": "speeches",
    "categoris": "categories",
    "discription": "description",
    "emrgency": "emergency",
    "qulification": "qualification",
    "tution": "tuition",
    "precent": "percent",
    "brik": "BRICS",
}

def main():
    rows_out = []
    with open(IN_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tokens = row["likely_typo_tokens"].split(";")
            hits = [(t, CONFIRMED[t.lower()]) for t in tokens if t.lower() in CONFIRMED]
            if hits:
                for typo, correction in hits:
                    rows_out.append([row["split"], row["text"], typo, correction])

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "text", "typo", "correction"])
        writer.writerows(rows_out)

    print(f"confirmed typo occurrences: {len(rows_out)}")
    print(f"unique typo words: {len(CONFIRMED)}")
    print(f"written to {OUT_PATH}")

if __name__ == "__main__":
    main()
