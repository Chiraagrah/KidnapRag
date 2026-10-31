# ReliabilityRAG MIS diagnostic on clean ReAct (HotpotQA + Qwen)

Runs the **no-attack** KidnapRAG ReAct pipeline and applies the **ReliabilityRAG
MIS defense (Algorithm 1)** to the 5 retrieved docs at **every ReAct loop**, as a
**read-only diagnostic** — the agent's observations are unchanged. Output tells
you **which ReAct loop is weakest** (least internally consistent retrieval).

Model + data match KidnapRAG: `Qwen/Qwen2.5-32B-Instruct` on `hotpotqa`.

## What was added / changed
- `ReAct/src/reliability_mis.py` — new. ReliabilityRAG Algorithm 1: isolated
  answering (Qwen) → pairwise DeBERTa NLI contradiction graph → rank-aware MIS →
  composite weakness metrics. Pure graph/MIS math is unit-tested.
- `ReAct/ReAct/attack_react_mis.py` — new. Clean run + per-loop MIS analysis +
  the two aggregations.
- `ReAct/scripts/plot_mis.py` — new. Aggregate weakness bar chart + single-loop
  contradiction-graph rendering.
- `ReAct/ReAct/e5_env.py` — 3-line read-only addition: stashes the ranked docs
  of the last search in `info['retrieved_docs']`. Does **not** change behavior.

## One-time setup on Wulver
```sh
conda activate KidnapRAG
pip install sentencepiece            # DeBERTa-v3 tokenizer needs this
pip install matplotlib               # only for plot_mis.py
```
The NLI checkpoint `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(~1.6 GB) downloads from the HuggingFace hub on first use. If the compute node
has no internet, pre-download it on a login node and point the job at the cache:
```sh
huggingface-cli download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
export HF_HUB_OFFLINE=1               # in the SLURM job, after the cache exists
```

## Run (from `ReAct/ReAct`)
Smoke test on one question first:
```sh
cd ReAct/ReAct
CUDA_VISIBLE_DEVICES=0,1 python attack_react_mis.py \
    --model_path Qwen/Qwen2.5-32B-Instruct --dataset hotpotqa --seed 1 \
    --nli_device cuda:0 --max_steps 15 --dry_run
```
Full run (drop `--dry_run`; `--limit N` for a subset):
```sh
CUDA_VISIBLE_DEVICES=0,1 python attack_react_mis.py \
    --model_path Qwen/Qwen2.5-32B-Instruct --dataset hotpotqa --seed 1 \
    --nli_device cuda:0 --max_steps 15
```
Notes
- Qwen loads with `tensor_parallel_size=1` on `cuda:1`; DeBERTa goes on the free
  `cuda:0`. On a single GPU pass `--nli_device cpu`.
- `--max_steps` is the raised ReAct loop cap (paper used 9).
- Isolated answering adds 5 short Qwen generations per search step.

## Outputs (`ReAct/results/mis_diagnostic/`)
- `hotpotqa_qwen25_mis_per_question.json` — per question, every loop's
  contradiction graph: `nodes` (rank + isolated answer + in-MIS flag),
  `edges` (contradiction probs), **`adjacency`** (binary k×k matrix) and
  **`contra_matrix`** (full k×k NLI contradiction probabilities, incl. sub-β),
  the rank-aware MIS, composite metrics, and that question's single weakest
  loop. Saved incrementally.
- `hotpotqa_qwen25_mis_summary.json` —
  - **aggregation A**: mean weakness / excluded-fraction / edge-density / MIS
    size per loop index, + `weakest_loop_index_overall`;
  - **aggregation B**: distribution of each question's weakest loop index.

Weakness scalar = `1 - |MIS|/k` (k=5). Higher = the retrieved set fractured into
contradicting camps = weaker, easier-to-hijack loop.

Both aggregations also carry an **NLI-only** column (`*_nli_only` /
`weakest_loop_index_overall_nli_only`): weakness recomputed from NLI edges alone
(`contra_prob >= beta`) with unknowns treated as neutral — the original
ReliabilityRAG rule. On clean HotpotQA the unknown rule saturates weakness near
the 0.8 ceiling, so this column keeps the raw-NLI fracture signal for comparison.
It is emitted natively by new runs; to add it to an already-finished run, use
`scripts/add_nli_only_weakness.py` (recomputes it from the stored `contra_matrix`,
no model re-run).

### Relevance: "unknown" answers count as contradictions (default)
A doc whose isolated answer is `unknown` could not answer the sub-query — it is
irrelevant to this loop. By default such a doc is forced to **contradict every
other doc** (including other unknowns), so irrelevant retrieval cannot form the
consistent majority and instead raises the loop's weakness. Each node carries an
`unknown` flag and each edge a `reason` (`nli` or `unknown`); `contra_matrix`
still holds the raw NLI probabilities. Pass `--unknown_neutral` to recover
ReliabilityRAG's original behavior (unknowns treated as neutral non-answers).
Note the k=5 ceiling: an all-unknown loop becomes a complete graph, so its MIS
is 1 and weakness caps at 0.8.

## Seeing the graph
Three ways, all from the same JSON:

1. **Rendered contradiction graph (PNG)** — needs matplotlib:
```sh
cd ReAct
python scripts/plot_mis.py --per_question results/mis_diagnostic/hotpotqa_qwen25_mis_per_question.json \
       --qid <QID> --step <N>
```
2. **Adjacency + probability matrix as text** — no matplotlib, prints to terminal:
```sh
python scripts/plot_mis.py --per_question results/mis_diagnostic/hotpotqa_qwen25_mis_per_question.json \
       --qid <QID> --step <N> --print_matrix
```
3. **Straight from the JSON** — every loop already carries `adjacency` and
   `contra_matrix`, so you can read the graph without running anything.

Aggregate weakness-by-loop bar chart (needs matplotlib):
```sh
python scripts/plot_mis.py --summary results/mis_diagnostic/hotpotqa_qwen25_mis_summary.json
```
