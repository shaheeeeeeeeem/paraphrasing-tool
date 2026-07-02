"""
analyze_filtered.py

Quick diagnostic pass over data/processed/parabank_filtered.tsv before
deciding how to downsample it. Checks:
  - score distribution (how much of the 7.7M came from the new 0.08-0.10 slice
    vs the original 0.02-0.08 band)
  - word count distribution on both reference and paraphrase_1
  - trivial pairs (reference == paraphrase_1, near-useless for training)
  - duplicate references (same sentence appearing multiple times)
  - a random sample from the 0.08-0.10 slice specifically, to eyeball whether
    that newly-added region is actually clean
"""

import pandas as pd

df = pd.read_csv("data/processed/parabank_filtered.tsv", sep="\t")

print(f"Total rows: {len(df):,}")
print()

# ---- Score distribution ----
print("=== Score distribution ===")
print(df["score"].describe())
print()

old_band = df[(df["score"] >= 0.02) & (df["score"] <= 0.08)]
new_band = df[(df["score"] > 0.08) & (df["score"] <= 0.10)]
print(f"Rows in original 0.02-0.08 band: {len(old_band):,} ({len(old_band)/len(df)*100:.1f}%)")
print(f"Rows in new 0.08-0.10 band:      {len(new_band):,} ({len(new_band)/len(df)*100:.1f}%)")
print()

# ---- Word count distribution ----
print("=== Word count: reference ===")
ref_wc = df["reference"].str.split().str.len()
print(ref_wc.describe())
print()

print("=== Word count: paraphrase_1 ===")
para_wc = df["paraphrase_1"].str.split().str.len()
print(para_wc.describe())
print()

# ---- Trivial pairs (reference == paraphrase_1, case-insensitive) ----
trivial = df[df["reference"].str.lower().str.strip() == df["paraphrase_1"].str.lower().str.strip()]
print(f"Trivial pairs (reference == paraphrase_1): {len(trivial):,} ({len(trivial)/len(df)*100:.2f}%)")
print()

# ---- Duplicate references ----
dupe_refs = df["reference"].duplicated().sum()
print(f"Duplicate references (same sentence appearing >1 time): {dupe_refs:,} ({dupe_refs/len(df)*100:.2f}%)")
print()

# ---- Random sample from the new 0.08-0.10 slice, to eyeball quality ----
print("=== Random sample from NEW 0.08-0.10 band (eyeball for junk) ===")
if len(new_band) > 0:
    sample = new_band.sample(n=min(30, len(new_band)), random_state=42)
    for _, row in sample.iterrows():
        print(f"[{row['score']:.4f}] {row['reference']}")
        print(f"         -> {row['paraphrase_1']}")
        print()
else:
    print("No rows in new band.")