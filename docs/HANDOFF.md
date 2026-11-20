# Handoff: WebThinker KidnapRAG on BrowseComp-Plus (Wulver)

_Written 2026-11-12. Branch: `webthinker`. Companion docs: `docs/DATASET_MIGRATION_PLAN.md` (the why), this file (the state)._

## 1. Goal
Run the WebThinker KidnapRAG experiments on a dataset where several retrieved docs *each* answer the question, so the ReliabilityRAG MIS defense has an honest majority to find. HotpotQA failed this (mean MIS 1.28/5; most docs answer "unknown" alone). BrowseComp-Plus was chosen; see the plan for the reasoning.

## 2. Setup used
| Part | Value |
|---|---|
| Agent + aux LLM | `openai/gpt-oss-120b`, own vLLM server (`ReliabilityRAG/scripts/serve_gpt_oss_local.slurm`, address in `~/vllm_local_server.txt`, 4 GPUs, 72 h) |
| Retriever | E5-large-v2, fp16, avg pooling, dot product, top-5, max 1 chunk per source doc |
| Data | 100 queries (seed 1, >=3 gold docs, mean 4.68) from `data/browsecomp_plus_decrypted.jsonl`; dev corpus = 8,191 docs, 144,846 chunks (~250 words) |
| NLI (diagnostic) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, beta 0.5 |
| Cluster | Wulver, account `ss797`, repo at `/project/ss797/dk694/Repos/KidnapRag`. No conda; use the existing Python env (scripts do not activate any env). |

`data/` and `datasets/` are gitignored. Never commit decrypted BrowseComp text.

## 3. What exists (all in git unless noted)
- `Webthinker/scripts/prepare_browsecomp_plus.py`: converter (select queries, chunk, write KidnapRAG + ReliabilityRAG formats, `stats.json`). Run once; outputs in `datasets/browsecomp_plus/`.
- `Webthinker/src/retrieval/e5_retriever.py`: `max_chunks_per_doc` dedup (over-fetch 8k, keep N chunks per `docid`).
- `Webthinker/scripts/run_attack.py`: `--retriever_device` (default cuda:0), `--max_chunks_per_doc`, lazy model imports (no vLLM import for WebThinker), preflight check of the LLM server, post-run guard (>50% empty outputs => exit, raw results saved to `<output>.FAILED`).
- `Webthinker/models/webthinker*.py`: retriever device flag; `extract_relevant_info_e5` now also stores chunk `id` and `score` in `search_documents` (prompt unchanged).
- `Webthinker/scripts/generate_attack_doc_kill.py`: `--gen_model` output-name tag (no crash on non-Qwen result files).
- `Webthinker/scripts/mis_agreement.py` + `scripts/wulver_mis_agreement.slurm`: per-hop read-only MIS/agreement diagnostic. Logic tested on stubbed data only.
- `scripts/wulver_webthinker_exp.slurm`: 1-GPU experiment job. `scripts/wulver_webthinker_with_server.sh`: server+launcher wrapper (untested).
- Check `git status`: the `run_attack.py` guards and `mis_agreement.py` files may still be uncommitted.

## 4. Status
Done and verified by output pasted from Wulver:
- Converter ran; dev index built (`datasets/browsecomp_plus/e5_index_bcp_dev_corpus`).
- Smoke test (2 queries) ran end to end: 7 searches on query 1, EM 0.5 on 2 queries (too small to mean anything). That smoke file predates the chunk-id change.

Failed:
- 100-query clean run `clean_gptoss` produced **empty outputs for all 100 queries** (`Connection error` to the LLM server; EM 0.0 is not a real score). `mis_agreement_clean.json` was built on it and has 0 hops. **Both files are invalid; delete them.**

Not started: agreement probe (plan section 3, ideal-top-5 variant), bait docs / `concat.py`, PoisonedRAG poisoned corpora, attack runs, full 100k-doc index, MIS defense mode, LLM-judge metric, gpt-oss harmony handling.

## 5. Next steps (in order)
1. Confirm the server is up: `curl http://$(cat ~/vllm_local_server.txt)/v1/models`. If not, resubmit the server job (`MAX_MODEL_LEN=65536`).
2. Re-run the smoke test (`EXTRA_ARGS="--dry_run" TAG=smoke`) and confirm non-empty outputs, >=1 search, and chunk ids like `24653#2` in `step_stats[*].search_documents`.
3. Full clean run: `EXTRA_ARGS="--concurrent_limit 8" TAG=clean_gptoss sbatch --export=ALL scripts/wulver_webthinker_exp.slurm` (needs `GPT_OSS_DIR`, `API_BASE` exported; `API_BASE=http://$(tr -d '[:space:]' < ~/vllm_local_server.txt)/v1`).
4. Cache the NLI model on a login node (`HF_HUB_DISABLE_XET=1`), then run `scripts/wulver_mis_agreement.slurm` (try `EXTRA_ARGS="--max_questions 3"` first).
5. Read the summary: go criterion is mean `n_match_gold` >= 3 and MIS size clearly above 1.28/5. If it fails, see plan section 7 (tighten gold>=3, allow 2 chunks/gold doc, or fall back to MultiHop-RAG inference queries).
6. Then attacks/defenses (plan sections 4.4, 5 items 3, 5, 6, 7, and 6.3 stages E-H).

## 6. Known risks and gotchas
- **gpt-oss raw completions:** the smoke output text started with `analysis` (harmony reasoning channel leaking into the completion). Check answer extraction and prompt contamination before trusting any accuracy number. If bad, add the gpt-oss branch to `generate_response` (plan item 3).
- **EM is the wrong metric** for BrowseComp answers; no LLM-judge yet. Read predictions by hand until it exists.
- **Silent failures:** `generate_response` retries 3 times then returns `""`; errors only show in `logs/bcp_<jobid>.out`. The new preflight/guard catches the all-failed case only.
- **`reasoning_effort` via `extra_body`** in `mis_agreement.py` is unverified on this server. If `n_unknown` is ~5 on every hop, isolated answers are coming back empty.
- **1 chunk per doc** keeps the best-scoring chunk of a ~13-chunk gold doc, which may not contain the answer; the diagnostic will show whether this hurts.
- Compute nodes are offline: cache HF models (E5, NLI, gpt-oss tokenizer) on a login node first; `HF_HUB_DISABLE_XET=1` fixes the stuck Xet downloads.
- The login node kills processes above ~1 GB memory; run heavy things via `srun`/`sbatch`.
- System Python on the cluster is 3.9 with a broken `~/.local` vLLM; the code needs >=3.10 (`X | None` hints). Use the project env, not the system Python.
- Open decisions (plan section 8): agent model (gpt-oss only vs QwQ agent), attacker model (Qwen3-30B vs gpt-oss), headline defense experiment (PoisonedRAG poisoning, KidnapRAG hijacking, or both).

## 7. Key paths
- Queries: `datasets/browsecomp_plus/bcp_select100.json` (+ `_rrag.json`, `selected_qids.txt`, `stats.json`)
- Corpus: `datasets/browsecomp_plus/bcp_dev_corpus.jsonl`; index `.../e5_index_bcp_dev_corpus/`
- Results: `Webthinker/results/browsecomp_plus/`; logs: `logs/`
- MIS reference code: `ReAct/src/reliability_mis.py`, `ReAct/MIS_DIAGNOSTIC_README.md`; HotpotQA baseline `ReAct/results/mis_diagnostic/hotpotqa_qwen25_mis_summary.json`
