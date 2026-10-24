# Memory hygiene is not free

When does cleaning an LLM agent's long-term memory help, and when does it destroy it?

This repository injects five kinds of junk memory (irrelevant, duplicate, stale, contradictory,
temporary) into real long-term memory corpora, retrieves with four memory representations
(dense vectors, BM25, hybrid, entity graph), and measures what a memory-hygiene layer
(deduplication, supersession, NLI contradiction detection, small-talk filtering) does to
retrieval. It also implements **gated hygiene**, our method: a label-free gate that switches
off hygiene components on stores where they would mostly delete genuine memories.


## Datasets (downloaded automatically from their official sources)

| Corpus | Memory type | Store | Unit | Gold evidence | Source |
|---|---|---|---|---|---|
| LoCoMo | long multi-session dialogue | 1 conversation (10) | dialogue turn | QA evidence ids | github.com/snap-research/locomo |
| LongMemEval (cleaned) | user-assistant chat history | 1 question's haystack | chat turn | `has_answer` turns | HF `xiaowu0162/longmemeval-cleaned` |
| MemoryAgentBench, Conflict_Resolution | fact updates (natural stale facts) | 1 example | numbered fact | latest fact holding the answer | HF `ai-hyz/MemoryAgentBench` |
| CLUTRR | relational / kinship | 8 stories, disjoint names | sentence | sentences on the proof chain | CSVs behind HF `CLUTRR/v1` |
| Loghub | system logs (templated, numeric) | 200-line window | log line | lines with the queried id | github.com/logpai/loghub |

## Setup (Windows, macOS or Linux, Python 3.10+)

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/download_data.py
python -m pytest -q tests
```

## Running

```
python scripts/run_grid.py configs/smoke.yaml        # 1-2 minutes, checks the pipeline (numbers not meaningful)
python scripts/run_grid.py configs/main.yaml --corpora locomo   # the real grid, one corpus at a time
python scripts/make_tables.py results/main
python scripts/make_figures.py results/main
python scripts/fit_gate.py results/main gate_models  # learn the gate, leave-one-corpus-out
```

Then set `gate: {model_dir: gate_models}` in a copy of `configs/main.yaml` with a new `out`
folder and re-run to evaluate the learned gate.

LLM experiments (need `ANTHROPIC_API_KEY`; every call is cached in `cache/`):

```
python scripts/run_e2e.py configs/e2e.yaml           # E4: answer accuracy with an LLM reader + judge
python scripts/run_natural.py configs/natural.yaml   # E5: natural junk written by an LLM memory writer
```

The grid is resumable: finished cells are skipped, so you can stop and restart at any time.

**Faster runs.**
* *Free GPU:* `notebooks/kaggle_run.ipynb` runs the whole grid on a Kaggle GPU (instructions inside).
* *Datasets in parallel:* open one terminal per group and give each a share of the CPU, e.g.
  `set MEMHYG_THREADS=4` (Windows) or `export MEMHYG_THREADS=4`, then
  `python scripts/run_grid.py configs/main.yaml --corpora clutrr loghub`. The caches are safe to share.
Embeddings, NLI scores and LLM answers are cached on disk.

## Experiments

| | Question | Output |
|---|---|---|
| E1 | How much does each junk type hurt each memory representation? Does vector memory degrade faster than graph memory? | `degradation.csv`, `pollution_effect.csv`, `vector_vs_graph.csv` |
| E2 | Harm map: how much junk does hygiene catch, and how many genuine memories does it wrongly invalidate, per corpus and component? | `hygiene_rates.csv`, `harm_map.pdf` |
| E3 | Can label-free store features predict that harm? Does gated hygiene keep the gains without the collapse? | `fit_gate.py` report, `hygiene_effect.csv` |
| E4 | Does it matter for final answers? | `results/e2e/e2e.parquet` |
| E5 | Do the conclusions hold for natural junk written by an LLM memory writer? | `results/natural/` |

All comparisons are paired per query; 95% CIs come from a store-level cluster bootstrap
(queries sharing a memory store are not independent).

## Code map

```
memhygiene/
  schema.py         MemoryUnit, Query, Store, Corpus
  text.py           tokenizer, entities, value slots, informativeness
  data/             one loader per corpus + downloader
  pollution/        synthetic.py (5 junk types x level), natural.py (LLM memory writer + annotator)
  embed.py          sentence-transformers embedder with SQLite cache (+ offline hashing stand-in)
  retrieval/        dense, bm25, hybrid (RRF), graph (bipartite entity graph + Personalized PageRank)
  hygiene/          core.py detectors, apply.py hard/soft, gate.py gated hygiene, nli.py
  features.py       label-free store features
  eval/             metrics (Recall/nDCG/MRR/junk@k, AUDC), cluster bootstrap
  runner.py         the experiment grid
  analysis.py       paper tables and statistics
  llm.py, reader.py Claude client, LLM reader and judge
configs/            smoke, main, embedders, ablation, e2e, natural
scripts/            download_data, run_grid, make_tables, make_figures, fit_gate, run_e2e, run_natural
tests/              unit tests and a toy end-to-end run
```

## Definitions

* **Contamination level L**: `round(L x number of genuine memories)` junk memories are added.
* **Targeted junk**: half of duplicate/stale/contradictory junk is derived from gold evidence
  memories (configurable with `pollution.targeted`).
* **False invalidation rate**: share of genuine memories a hygiene setting flags (score >= 0.5).
* **Junk recall**: share of injected junk it flags.
* **Hygiene modes**: `none`; `hard` deletes flagged memories; `soft` down-weights them at
  retrieval time; `gated` is soft hygiene with the gate and a per-component budget.
