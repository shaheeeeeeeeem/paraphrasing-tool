"""
clean_parabank.py

Streams ParaBank v2 (data/raw/parabank2.tsv), applies the locked-in score
range filter + heuristic filters, and writes (reference, paraphrase_1)
pairs to data/processed/.

File is large (~9GB uncompressed) -- this script NEVER loads it fully into
memory. It reads line-by-line and writes output incrementally.

ParaBank v2 line format (TSV, no header):
    [dual-condition score]\t[reference]\t[paraphrase_1]\t...\t[paraphrase_5]
Rows are ragged -- 0 to 5 paraphrases per line. Do not use pd.read_csv.
"""

import argparse
import csv
import re
import sys

# ---- Locked-in filtering rules ------------------------------------------

SCORE_MIN = 0.02
SCORE_MAX = 0.1

WORD_COUNT_MIN = 6
WORD_COUNT_MAX = 40

BAD_START_CHARS = ("[", "|", "(")

KEYWORD_PATTERNS = [
    "list of",
    "subject:",
    "discharge",
    "article",
    "regulation",
    "directive",
    "(ec) no",
]

# excessive bracket/number density heuristic: reject if more than this
# fraction of characters are digits or brackets/parens
MAX_SYMBOL_DENSITY = 0.15

PROGRESS_EVERY = 500_000


def word_count(s: str) -> int:
    return len(s.split())


def has_bad_start(s: str) -> bool:
    return s.startswith(BAD_START_CHARS)


def has_bad_keyword(s_lower: str) -> bool:
    return any(pat in s_lower for pat in KEYWORD_PATTERNS)


def has_junk_markers(s_lower: str) -> bool:
    if "%1" in s_lower or "http" in s_lower:
        return True
    symbol_count = sum(1 for c in s_lower if c.isdigit() or c in "[]{}()|")
    if len(s_lower) == 0:
        return True
    density = symbol_count / len(s_lower)
    return density > MAX_SYMBOL_DENSITY


def passes_heuristics(reference: str) -> bool:
    ref_lower = reference.lower()

    wc = word_count(reference)
    if wc < WORD_COUNT_MIN or wc > WORD_COUNT_MAX:
        return False

    if has_bad_start(reference):
        return False

    if has_bad_keyword(ref_lower):
        return False

    if has_junk_markers(ref_lower):
        return False

    return True


def parse_line(line: str):
    """
    Returns (score, reference, paraphrase_1) if the line is well-formed
    and passes the score range + heuristic filters, else None.
    """
    fields = line.rstrip("\n").split("\t")

    # need at least score, reference, paraphrase_1 -> 3 fields
    if len(fields) < 3:
        return None

    raw_score = fields[0].strip()
    reference = fields[1].strip()
    paraphrase_1 = fields[2].strip()

    if not paraphrase_1:
        return None

    try:
        score = float(raw_score)
    except ValueError:
        return None

    if score < SCORE_MIN or score > SCORE_MAX:
        return None

    if not reference or not passes_heuristics(reference):
        return None

    return score, reference, paraphrase_1


def main():
    parser = argparse.ArgumentParser(description="Filter ParaBank v2 down to clean (reference, paraphrase_1) pairs.")
    parser.add_argument("--input", default="data/raw/parabank2.tsv", help="Path to raw ParaBank v2 TSV file")
    parser.add_argument("--output", default="data/processed/parabank_filtered.tsv", help="Path to write filtered pairs")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many rows are KEPT (not lines read). Useful for quick tests against the full file without waiting for a full pass.")
    args = parser.parse_args()

    lines_read = 0
    rows_kept = 0

    with open(args.input, "r", encoding="utf-8", errors="replace") as infile, \
         open(args.output, "w", encoding="utf-8", newline="") as outfile:

        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["score", "reference", "paraphrase_1"])

        for line in infile:
            lines_read += 1

            result = parse_line(line)
            if result is not None:
                score, reference, paraphrase_1 = result
                writer.writerow([score, reference, paraphrase_1])
                rows_kept += 1

                if args.limit is not None and rows_kept >= args.limit:
                    print(f"[stopped] hit --limit of {args.limit:,} kept rows after {lines_read:,} lines read", file=sys.stderr)
                    break

            if lines_read % PROGRESS_EVERY == 0:
                print(f"[progress] lines read: {lines_read:,} | rows kept: {rows_kept:,}", file=sys.stderr)

    print(f"[done] total lines read: {lines_read:,} | total rows kept: {rows_kept:,}", file=sys.stderr)


if __name__ == "__main__":
    main()