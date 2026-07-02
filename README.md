# Style-Conditioned Paraphrasing Tool

A style-conditioned paraphrasing tool: input any English sentence, get a paraphrase
back in a controlled register, selected via a prepended style token:

- `[FORMAL]`
- `[CASUAL]`
- `[POLITE]`

This is a pivot from an earlier project (Eng2Bard, a Shakespeare-style translator)
that failed due to relying on unverified LLM-generated synthetic training data.
**No LLM-generated data is used anywhere in this project.** All data comes from
public datasets; style labeling is done via pretrained/self-trained classifiers,
never generation.

Style mode is **conditioned generation**, not style transfer -- the input can be
any tone, and the output tone is controlled purely by the token. No matched
cross-style pairs are required.

## Goal

Train and compare three architectures on the same style-tagged dataset:

1. **Vanilla Transformer (2017)** -- from-scratch PyTorch encoder-decoder.
2. **Modern Transformer** -- adds RoPE, Grouped-Query Attention, and RMSNorm.
3. **Pretrained fine-tune** -- `facebook/bart-large` fine-tuned on the tagged dataset.

Goal is architecture comparison + portfolio piece, evaluated via BLEU + BERTScore.

## Status

| Component | Status |
|---|---|
| Politeness classifier (`roberta-base`, fine-tuned on `Cleanlab/stanford-politeness`) | ✅ Done |
| Formality classifier (`s-nlp/roberta-base-formality-ranker`) | Off-the-shelf, not yet wired in |
| ParaBank v2 filtering script (`src/data_prep/clean_parabank.py`) | Written, not yet run on full file |
| Style labeling pipeline (`src/data_prep/style_labeling.py`) | Not started |
| Architecture 1 (vanilla transformer) | Architecture built, training not yet done |
| Architecture 2 (modern transformer) | Not started |
| Architecture 3 (BART-large fine-tune) | Not started |
| Comparison / eval notebook | Not started |
| FastAPI + Streamlit app | Not started |

## Politeness Classifier

Fine-tuned `roberta-base` on `Cleanlab/stanford-politeness` (`fine-tuning/train_full.csv`
+ `fine-tuning/test.csv`), 3-class classification (`impolite` / `neutral` / `polite`).

- Class distribution is imbalanced (~45% neutral, ~36% impolite, ~19% polite in both
  train and test) -- handled via inverse-frequency weighted `CrossEntropyLoss`
  (custom `WeightedTrainer` subclass), since `Trainer`/`TrainingArguments` has no
  built-in class-weighting hook.
- Trained 4 epochs on Colab (T4 GPU), batch size 32, lr 2e-5, `max_length=64`
  (covers ~94% of examples without truncation).
- Final model selected via `metric_for_best_model="eval_accuracy"` -- picked over
  `eval_loss` after an earlier run showed loss and classification quality
  (accuracy, F1) diverging across epochs.
- Final checkpoint: epoch 4 -- best on accuracy, all per-class F1 scores, and
  validation loss simultaneously.

## ParaBank v2 Filtering

Primary dataset is ParaBank v2 (~9GB uncompressed TSV, ~5M pairs), built via
Czech<->English back-translation. Key finding: the file's dual-condition score
measures translation-pair confidence, not sentence naturalness -- high-scoring
rows are dominated by Wikipedia titles and EU legislative text. Sampling strategy
locked in:

- Score range: **0.02-0.08** (natural language dominates this range empirically)
- Word count 6-40, drop rows starting with `[`, `|`, `(`, drop rows matching
  legal/procedural keyword patterns, drop rows with excessive symbol density
  or `http`/`%1`
- Take `paraphrase_1` (best-ranked) only

`src/data_prep/clean_parabank.py` implements this as a line-by-line streaming
filter (never loads the 9GB file into memory), writing filtered
`(score, reference, paraphrase_1)` rows incrementally to `data/processed/`.

## Repo Structure

See `docs/project-summary.md` for the full planned repo layout, architecture
details, and decisions log.

## Setup

```bash
pip install -r requirements.txt
```
