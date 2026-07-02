"""
sample_parabank.py

Two-stage approach, now that we know ~7.7M clean rows are available
(far more than originally assumed) -- we don't have to guess a final
training size before running the style classifiers.

Stage 1: Clean the full pool.
  - Drop trivial pairs (reference == paraphrase_1)
  - Deduplicate by reference, keep first (highest-score) occurrence
  - Write the FULL cleaned pool -- no arbitrary size cap. This is your
    reserve: if a style bucket comes up thin after labeling, pull more
    from here rather than re-running clean_parabank.py.

Stage 2: Pull a WORKING SAMPLE from the cleaned pool, sized for a first
  pass through both style classifiers (formality + politeness). This is
  NOT your final training set size -- it's sized to get a real read on
  label distribution before committing to a final number.
"""

import pandas as pd

INPUT_PATH = "data/processed/parabank_filtered.tsv"
CLEANED_POOL_PATH = "data/processed/parabank_cleaned_pool.tsv"
WORKING_SAMPLE_PATH = "data/processed/parabank_working_sample.tsv"

WORKING_SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

# ---- Stage 1: clean the full pool ----
df = pd.read_csv(INPUT_PATH, sep="\t")
print(f"Loaded: {len(df):,} rows")

is_trivial = df["reference"].str.lower().str.strip() == df["paraphrase_1"].str.lower().str.strip()
df = df[~is_trivial]
print(f"After dropping trivial pairs: {len(df):,} rows")

df = df.drop_duplicates(subset="reference", keep="first")
print(f"After deduplicating by reference: {len(df):,} rows")

df.to_csv(CLEANED_POOL_PATH, sep="\t", index=False)
print(f"Wrote full cleaned pool: {len(df):,} rows -> {CLEANED_POOL_PATH}")
print()

# ---- Stage 2: pull a working sample for the first classifier pass ----
if len(df) < WORKING_SAMPLE_SIZE:
    print(f"WARNING: cleaned pool ({len(df):,}) is smaller than "
          f"WORKING_SAMPLE_SIZE ({WORKING_SAMPLE_SIZE:,}). Using full pool as the working sample.")
    working_sample = df
else:
    working_sample = df.sample(n=WORKING_SAMPLE_SIZE, random_state=RANDOM_STATE)

working_sample = working_sample.sort_values("score", ascending=False).reset_index(drop=True)
working_sample.to_csv(WORKING_SAMPLE_PATH, sep="\t", index=False)

print(f"Wrote working sample: {len(working_sample):,} rows -> {WORKING_SAMPLE_PATH}")