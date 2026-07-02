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

Style mode is **conditioned generation**, not style transfer — the input can be
any tone, and the output tone is controlled purely by the token. No matched
cross-style pairs are required.

## Goal

Train and compare three architectures on the same style-tagged dataset:

1. **Vanilla Transformer (2017)** — from-scratch PyTorch encoder-decoder.
2. **Modern Transformer** — adds RoPE, Grouped-Query Attention, and RMSNorm.
3. **Pretrained fine-tune** — `facebook/bart-large` fine-tuned on the tagged dataset.

Goal is architecture comparison + portfolio piece, evaluated via BLEU + BERTScore.
Same test split used across all three for a fair comparison.

## Status

| Component | Status |
|---|---|
| Politeness classifier (`roberta-base`, fine-tuned on `Cleanlab/stanford-politeness`) | Done |
| Formality classifier (`s-nlp/roberta-base-formality-ranker`) | Done — off-the-shelf, wired into labeling pipeline |
| ParaBank v2 filtering (`src/data_prep/clean_parabank.py`) | Done — run on full 19.72M-line file |
| Downsampling / pool cleaning (`src/data_prep/sample_parabank.py`) | Done — cleaned pool + 500k working sample |
| Style labeling pipeline (`src/data_prep/style_labeling.py`, `src/data_prep/topup_polite_reserve.py`) | Done — 200,001-row balanced, tagged dataset |
| Architecture 1 (vanilla transformer) | Architecture built (reused/adapted from Eng2Bard), training not yet done |
| Architecture 2 (modern transformer) | Not started |
| Architecture 3 (BART-large fine-tune) | Not started |
| Comparison / eval notebook | Not started |
| FastAPI + Streamlit app | Not started |

## Architectures

### 1. Vanilla Transformer (2017)

Standard encoder-decoder, multi-head attention, learned positional embeddings,
post-norm LayerNorm. Adapted from the Eng2Bard codebase — architecture was
already correctly implemented and verified there (causal masking, padding
masks, and teacher-forcing shift all confirmed correct). Needs: tokenizer
retrained on the new corpus with style tokens added as special tokens, a new
train/val/test split (Eng2Bard's stratified split doesn't apply — no
`category`/`family_id` columns in this dataset), and a `collate_fn` update to
prepend the style token to the source sentence.

### 2. Modern Transformer

Adds RoPE (rotary position embeddings, replaces learned positional
embeddings), Grouped-Query Attention (fewer K/V heads than Q heads, shared
across groups), and RMSNorm (LayerNorm without mean-centering or bias). Not
yet started.

### 3. Pretrained Fine-tune

`facebook/bart-large` fine-tuned on the tagged dataset, following Stanford
OVAL's paraphraser recipe (`stanford-oval/paraphraser-bart-large`), with style
tokens prepended to the input during fine-tuning as the novel addition. Not
yet started.

### 4. Comparison & Evaluation

BLEU + BERTScore across all three, same test split. Not yet started.

## Dataset

Primary source: ParaBank v2 (Hu et al., 2019, JHU Decompositional Semantics
Initiative), built via Czech↔English back-translation with lexically-constrained
decoding for diversity — not LLM-generated. See `data/README.md` for the full
pipeline from raw download through to the final labeled dataset.

**Key finding**: the file's dual-condition score measures translation-pair
confidence, not sentence naturalness — high-scoring rows are dominated by
Wikipedia titles and EU legislative text. Final sampling used a score range of
0.02–0.10 with heuristic filters (word count, junk-pattern drops), landing on
7,738,958 filtered rows out of 19.72M raw lines.

## Style Labeling

Two classifiers run over the data, with a margin-over-chance tie-break for
conflicts:

- **Formality** (`s-nlp/roberta-base-formality-ranker`, binary, off-the-shelf):
  every row gets `formal`/`informal` by default.
- **Politeness** (fine-tuned `roberta-base`, 3-class: impolite/neutral/polite):
  only rows predicted `polite` with confidence > 0.5 compete for the
  `[POLITE]` tag.
- **Tie-break**: for polite candidates, compare `confidence - chance baseline`
  (0.5 for the binary formality model, 0.333 for the 3-class politeness
  model) — whichever margin is higher wins. This corrects for binary vs.
  3-class softmax scores not being directly comparable (a 3-class model has a
  lower "no-information" floor, so raw confidence numbers aren't apples-to-apples).

`style_labeling.py` runs this over the initial 500k working sample and balances
to the largest common bucket size. `topup_polite_reserve.py` pulls additional
rows from the 7.34M reserve pool specifically to top up the `[POLITE]` bucket,
since it's gated by two conditions (confidence threshold + tie-break win)
while `[FORMAL]`/`[INFORMAL]` only need one binary classification to qualify —
roughly a 4–4.5% yield rate on `[POLITE]` per fresh sample pulled.

**Final dataset**: 200,001 rows, balanced at 66,667 per style
(`data/processed/parabank_labeled.tsv`, gitignored — see `data/README.md` to
regenerate). Columns: `reference`, `paraphrase_1`, `style_label` (0=polite,
1=formal, 2=informal), plus raw classifier labels/confidences for debugging.

**Known bug, fixed**: the formality classifier's `id2label` mapping was
initially inverted (`{0: "formal", 1: "informal"}` instead of the model's
actual `{0: "informal", 1: "formal"}`), silently swapping formal/informal
labels on the first labeling run. Caught via manual spot-check against the
raw sentence text, fixed by correcting the mapping and relabeling the existing
output (no re-inference needed, since the underlying confidence scores were
never affected — only the string labels were swapped).

## Politeness Classifier

Fine-tuned `roberta-base` on `Cleanlab/stanford-politeness`
(`fine-tuning/train_full.csv` + `fine-tuning/test.csv`), 3-class classification
(`impolite` / `neutral` / `polite`).

- Class distribution is imbalanced (~45% neutral, ~36% impolite, ~19% polite in
  both train and test) — handled via inverse-frequency weighted
  `CrossEntropyLoss` (custom `WeightedTrainer` subclass), since
  `Trainer`/`TrainingArguments` has no built-in class-weighting hook.
- `max_length=64` chosen after checking the actual token-length distribution
  (covers ~94% of examples without truncation).
- Final checkpoint (epoch 4) won on accuracy, all per-class F1 scores, and
  validation loss simultaneously — no overfitting across the 4-epoch run.
- Checkpoint lives at `checkpoints/politeness-classifier-v3/final/` (gitignored
  — see `checkpoints/` notes below).

## Repo Structure

```
paraphrase-tool/
├── notebooks/
│   ├── 01_transformer_scratch.ipynb        # 2017 arch -- adapted from Eng2Bard
│   ├── 02_transformer_modern.ipynb         # RoPE + GQA + RMSNorm -- not started
│   ├── 03_finetune_pretrained.ipynb        # BART-large fine-tune -- not started
│   └── 04_comparison_eval.ipynb            # BLEU/BERTScore across all 3
│
├── data/
│   ├── raw/                                # untouched downloads, gitignored
│   ├── processed/                          # filtered/cleaned/sampled/labeled TSVs, gitignored
│   └── README.md                           # full data pipeline docs
│
├── src/
│   ├── data_prep/
│   │   ├── clean_parabank.py               # score filter + heuristic filter logic
│   │   ├── sample_parabank.py              # dedup + trivial-drop + working-sample sampling
│   │   ├── style_labeling.py               # formality + politeness classifier inference + tie-break
│   │   ├── topup_polite_reserve.py         # pulls more polite rows from the reserve pool
│   │   └── tokenizer.py                    # shared tokenizer/vocab across all 3 models
│   │
│   ├── models/
│   │   ├── transformer_scratch/            # 2017 arch modules
│   │   ├── transformer_modern/             # RoPE, GQA, RMSNorm modules
│   │   └── finetune/                       # BART fine-tuning wrapper/config
│   │
│   └── eval/
│       └── metrics.py                      # shared BLEU/BERTScore functions
│
├── checkpoints/                            # saved model weights, gitignored
│   ├── politeness-classifier-v3/
│   │   └── final/                          # roberta-base, 3-class, epoch 4
│   ├── transformer_scratch/
│   ├── transformer_modern/
│   └── finetuned_bart/
│
├── app/
│   ├── backend/
│   │   ├── main.py                         # FastAPI app, model-serving endpoints
│   │   ├── inference.py                    # load checkpoint(s), run generation
│   │   └── schemas.py                      # Pydantic request/response models
│   ├── frontend/
│   │   └── streamlit_app.py                # UI, calls FastAPI backend
│   └── requirements.txt                    # app-specific deps
│
├── deployment/                             # fill in only if deploying
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   └── aws/
│       └── notes.md
│
├── docs/
│   ├── paraphrase-tool-pipeline.md         # detailed pipeline decisions doc
│   └── project-summary.md                  # architecture + decisions log
│
├── .gitignore
├── requirements.txt                        # training deps (torch, transformers, datasets)
└── README.md                               # this file
```

## Setup

```bash
pip install -r requirements.txt
```

Large data files and model checkpoints are gitignored — see `data/README.md`
for how to regenerate the dataset pipeline from scratch, and
`docs/project-summary.md` for the full architecture/decisions log.

## Data Attribution

This project's training data is derived from **ParaBank 2**:

> J. Edward Hu, Abhinav Singh, Nils Holzenberger, Matt Post, and Benjamin Van Durme. 2019.
> *Large-Scale, Diverse, Paraphrastic Bitexts via Sampling and Clustering.*
> In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 44–54.
> https://aclanthology.org/K19-1005/

**Changes made to the original data**: the raw ~19.72M-line corpus was filtered to a
0.02–0.10 dual-condition score range with heuristic quality filters (7,738,958 rows),
deduplicated and cleaned (~7.34M-row pool), randomly sampled (500k working sample),
labeled for formality/politeness via classifier inference, and balanced to a final
200,001-row set (66,667 rows per style). Only `paraphrase_1` from each original row
is used; `paraphrase_2`–`paraphrase_5` are discarded. See `data/README.md` for full
pipeline details.
