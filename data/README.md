# Data

This folder tracks the raw-to-labeled pipeline for the ParaBank v2 corpus.
All large intermediate/final files are gitignored (see repo root `.gitignore`)
— this README exists so the pipeline is reproducible without them being
committed.

## Pipeline overview

```
raw/parabank-2.0.zip (JHU download, ~2.3GB, gitignored)
  │
  ▼  clean_parabank.py   (score filter 0.02–0.10 + heuristic filters,
  │                       streamed line-by-line over 19.72M raw lines)
  │
processed/parabank_filtered.tsv        (7,738,958 rows)
  │
  ▼  sample_parabank.py Stage 1  (dedup by reference + trivial-pair drop)
  │
processed/parabank_cleaned_pool.tsv    (~7,338,992 rows — uncapped reserve)
  │
  ▼  sample_parabank.py Stage 2  (random 500k draw)
  │
processed/parabank_working_sample.tsv  (500,000 rows)
  │
  ▼  style_labeling.py  (formality + politeness classifiers,
  │                      margin-over-chance tie-break, balance to
  │                      largest common bucket size)
  │
  ├── polite bucket short of 66,667 target
  │
  ▼  topup_polite_reserve.py  (pulls fresh rows from
  │                            parabank_cleaned_pool.tsv, same
  │                            classifier + tie-break logic, keeps
  │                            only new polite rows, appends)
  │
processed/parabank_labeled.tsv         (200,001 rows — FINAL, balanced
                                         66,667 per style)
```

## Files

| File | Rows | Description | Tracked in git? |
|---|---|---|---|
| `raw/parabank-2.0/` | ~5M pairs | Untouched JHU download | No (gitignored) |
| `processed/parabank_filtered.tsv` | 7,738,958 | Score-range (0.02–0.10) + heuristic-filtered rows from the full 19.72M-line raw file | No (gitignored, large) |
| `processed/parabank_cleaned_pool.tsv` | ~7,338,992 | Deduplicated (by reference), trivial-pairs-dropped pool. Uncapped reserve, used to top up thin style buckets. | No (gitignored, large) |
| `processed/parabank_working_sample.tsv` | 500,000 | Random draw from the cleaned pool. First-pass input to the style classifiers. | No (gitignored, large) |
| `processed/parabank_labeled.tsv` | 200,001 | **Final tagged dataset.** Columns: `reference`, `paraphrase_1`, `style_label` (0=polite, 1=formal, 2=informal), `formality_label`, `formality_conf`, `politeness_label`, `politeness_conf`. Balanced at 66,667 rows per style. This is what the 3 architectures train on. | No (gitignored, large) |

## Regenerating this folder from scratch

1. Download ParaBank v2 from `cs.jhu.edu/~vandurme/data/parabank-2.0.zip`,
   unzip into `data/raw/`.
2. Run `src/data_prep/clean_parabank.py` → produces `parabank_filtered.tsv`.
   Streams the file line-by-line (never loads the full ~9GB into memory).
   Score ceiling is 0.02–0.10 — confirmed empirically that natural language
   dominates this range, while higher scores skew toward Wikipedia titles and
   EU legislative/procedural text (the score measures translation-pair
   confidence, not sentence naturalness).
3. Run `src/data_prep/sample_parabank.py` → Stage 1 produces
   `parabank_cleaned_pool.tsv` (dedup + trivial-pair drop), Stage 2 produces
   `parabank_working_sample.tsv` (random 500k draw from the cleaned pool).
4. Ensure the politeness classifier checkpoint exists at
   `checkpoints/politeness-classifier-v3/final/` (see root `README.md`,
   "Politeness Classifier" section, for training details).
5. Run `src/data_prep/style_labeling.py` → runs both classifiers over the
   500k working sample, applies the margin-over-chance tie-break, balances to
   the largest common bucket size. On the first run this produced 155,467 rows
   (66,667 formal, 66,667 informal, only 22,133 polite — short of target).
6. If the polite bucket is short of 66,667, run
   `src/data_prep/topup_polite_reserve.py` → pulls fresh rows from
   `parabank_cleaned_pool.tsv` (excluding rows already in the working sample),
   runs them through the same classifier + tie-break pipeline, keeps only new
   `[POLITE]` rows, and appends until `parabank_labeled.tsv` reaches 200,001
   rows (66,667 per style). Expect roughly a 4–4.5% polite yield rate per
   fresh batch pulled — pull generously (~1.3M rows) to avoid multiple runs.

## Style labeling logic (summary)

- **Formality** (`s-nlp/roberta-base-formality-ranker`, binary, off-the-shelf):
  every row defaults to `formal` or `informal`.
- **Politeness** (self-trained `roberta-base`, 3-class): only rows predicted
  `polite` with confidence > 0.5 are candidates for `[POLITE]`.
- **Tie-break**: `confidence - chance baseline` (0.5 for formality's 2 classes,
  0.333 for politeness's 3 classes) — higher margin wins. This normalizes for
  the fact that a 3-class softmax has a structurally lower "no-information"
  floor than a 2-class softmax, so raw confidence values aren't directly
  comparable without this adjustment.

## Known data notes

- ParaBank v2's dual-condition score measures translation-pair confidence, not
  sentence naturalness — high scores skew toward Wikipedia titles and EU
  legislative text, hence the 0.02–0.10 sampling window rather than taking the
  top-scored rows.
- File format: TSV, no header in the raw file (`clean_parabank.py`'s output
  does have headers). Each raw line is
  `[dual-condition score]\t[reference]\t[paraphrase_1]...[paraphrase_5]`,
  ragged column count (0–5 paraphrases per row) — parsed line-by-line, not
  via `pd.read_csv` directly on the raw file, since ragged rows misalign.
- Only `paraphrase_1` (best-ranked, leftmost) is used; paraphrases 2–5 are
  discarded.
- Some rows contain explicit/vulgar language (source is back-translated movie
  subtitle/screenplay text) — not filtered out, since register diversity
  (including impolite/informal registers) is relevant to this project's style
  labels.
- The `[POLITE]` bucket is structurally harder to fill than `[FORMAL]`/`[INFORMAL]`
  (gated by both a confidence threshold and a tie-break win, vs. one binary
  classification) — this is expected, not a bug, and is why
  `topup_polite_reserve.py` exists as a separate step.
- **Fixed bug**: the formality classifier's `id2label` mapping was initially
  inverted in `style_labeling.py` (`{0: "formal", 1: "informal"}` instead of
  the model's actual `{0: "informal", 1: "formal"}`, confirmed against the
  model's own `config.json`). This silently swapped every formal/informal
  label on the first run — caught via manual spot-check of raw sentence text
  against assigned labels, then fixed by correcting the mapping and relabeling
  the already-computed output (no re-inference needed, since only the string
  label was wrong, not the underlying confidence scores).

## Supplement datasets (only if a style bucket ends up thin post-labeling)

Not currently used, kept here as a fallback option:

- **PAWS-Wiki**: `https://huggingface.co/datasets/google-research-datasets/paws`
  — ~108k human-labeled pairs, Wikipedia-sourced (formal-leaning). Filter to
  `label==1` only.
- **QQP** (via GLUE): `https://huggingface.co/datasets/nyu-mll/glue/viewer/qqp`
  — ~400k question pairs (casual-leaning). Filter to `label==1`/duplicate only.
