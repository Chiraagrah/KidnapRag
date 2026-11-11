# Action Plan: WebThinker KidnapRAG experiment on a high-redundancy dataset (Wulver)

_Prepared 2026-11-11. Covers: dataset choice, conversion to the KidnapRAG/WebThinker formats, code changes, and running on Wulver against your own gpt-oss-120b server (the `wulver_run_with_server.sh` pattern from ReliabilityRAG)._

---

## 0. TL;DR

- **Use BrowseComp-Plus.** Its gold docs each state the final answer on their own, so isolated answering gives several docs that agree. MultiHop-RAG mostly does not. It has the same problem as HotpotQA, and over half of its answers are yes/no or before/after.
- **Don't use all 830 queries.** Pick about 100 queries with **≥ 3 gold docs**. Chunk the documents into passages of about 300 words. Cap retrieval at **one chunk per source doc in the top-k**, so that agreement comes from separate documents and not from five chunks of one page.
- **Run a short "agreement probe" before the expensive runs** (§3). It measures, for each dataset, how many retrieved docs agree on the answer when each doc is answered alone. It should take about an hour on the gpt-oss server, and it turns the dataset choice into a measurement.
- **Code work.** Write one converter script. Make about six small fixes to the WebThinker runner: configurable retriever GPU, per-doc dedup, a gpt-oss model branch, `qid`, the attacker-model tag, and an LLM judge. Add a Slurm wrapper modeled on `wulver_run_with_server.sh`.

---

## 1. The problem, in numbers from your own run

`ReAct/results/mis_diagnostic/hotpotqa_qwen25_mis_summary.json` (clean HotpotQA, Qwen2.5-32B, k = 5, 100 questions), loop 1:

| metric | value |
|---|---|
| mean MIS size (unknown = contradiction) | **1.28 / 5** |
| mean weakness | 0.744 (ceiling 0.8) |
| mean MIS size, NLI-only (unknown = neutral) | 3.79 / 5 |

Most of the 5 retrieved docs give "unknown" when answered alone. HotpotQA has about 2 supporting paragraphs per question, and each one covers a different hop. So no honest majority forms, and a poisoned doc that does answer can win the MIS.

**What the defense needs:** in a search step's top-k, the number of honest docs that each answer the question and agree must be greater than the number of poisoned docs. The new dataset is useful only if it raises the first number.

---

## 2. Dataset decision

### 2.1 What each dataset actually provides

| | **BrowseComp-Plus** | **MultiHop-RAG** |
|---|---|---|
| Queries | 830 | 2,556 |
| Corpus | 100,195 web docs, human-verified (median about 5.2k words) | 609 news articles (Sep–Dec 2023, about 2k tokens each) |
| Positives per query | **6.1 evidence docs, 2.9 gold docs on average** | 2–4 evidence facts (42% have 2); null queries have 0 |
| Does a single positive answer the full question? | **Gold docs: yes.** Each gold doc contains the final answer. | Rarely. Each evidence fact covers one side or one hop. |
| Answer type | Short entity or string (names, titles, numbers) | Inference: entity. Comparison: **yes/no**. Temporal: yes/no/before/after. Null: "insufficient information" |
| Hard negatives | **76 per query**, mined from the web | None annotated; the corpus is tiny |
| Difficulty for the agent | Very hard (deep multi-hop browsing; GPT-4.1 + Qwen3-8B retriever ≈ 35%) | Moderate |
| Fit with WebThinker | Built for deep-research agents and multi-turn search | Built for single-shot RAG with chunk retrieval |
| License / access | MIT. Queries are **obfuscated** on HF and must be decrypted with the repo's `decrypt_dataset.py` | ODC-BY, plain JSON |

On the sample you attached (`temp.json`, query 784): **2 gold docs**, both containing the full answer "Jacqueline Georgette Cantrelle". 4 of 5 evidence docs mention the surname, and **0 of 72** negatives do. The gold docs are short Wikipedia/profile pages (1.5k–3.4k chars). The negatives are long (median about 1.2k words, up to 270k chars).

### 2.2 Why BrowseComp-Plus

1. **Isolated answering actually yields answers.** A gold doc asked alone returns the real answer instead of "unknown". This fixes the specific failure shown in §1.
2. **MultiHop-RAG recreates the HotpotQA problem.** For comparison and temporal queries (56% of the set), each evidence article supports only one side, so isolated answers are again "unknown". Yes/no and before/after answers also make NLI contradiction edges and ASR almost meaningless: a poisoned "yes" against an honest "no" leaves no paraphrase structure for the MIS to use.
3. **Realistic retrieval.** The 100K-doc corpus with 76 hard negatives per query makes poisoned docs compete with plausible distractors. In MultiHop-RAG's 609 articles, an injected doc ranks top-k almost trivially.
4. **It matches WebThinker's purpose** (multi-search deep research).

### 2.3 Caveats to state honestly in the paper or report

- 2.9 gold docs on average is better, **not large**. To get "honest majority > poison" we filter to gold ≥ 3 and keep the poison budget per search at 1–2 docs.
- The questions are very hard, so **clean accuracy with gpt-oss-120b will be low**. ASR and defense effects are only interpretable on queries that are answered correctly when clean. Plan for a clean pass first, then attack only the "clean-correct" subset (or report both).
- Documents are long, so chunking is required (see §4.3). E5-large-v2 truncates at 512 tokens, and WebThinker pastes the **full** `page_info` of all top-k docs into the prompt.

**Fallback:** if the probe in §3 shows BrowseComp-Plus agreement is still too low, use MultiHop-RAG **inference-type queries only** (816, entity answers) as a secondary set. The same converter design applies.

---

## 3. Step 0: agreement probe (≈ 1 h; do this first)

Goal: a number that justifies the choice. Reuse `ReAct/src/reliability_mis.py` (isolated answering + DeBERTa NLI + MIS) directly. No agent is needed.

For 50 sampled queries per dataset:

1. Build the "ideal" top-5: gold/evidence docs first, padded with hard negatives (BrowseComp-Plus) or other corpus chunks (MultiHop-RAG), chunked as in §4.3.
2. Run isolated answering with gpt-oss-120b (the server from `serve_gpt_oss_local.slurm`).
3. Record: number of docs whose isolated answer matches gold (LLM judge), unknown rate, MIS size in both modes (unknown = contradiction, NLI-only).
4. Compare against the HotpotQA baseline numbers in §1.

**Go criterion:** mean agreeing-honest-docs ≥ 3 and MIS size (unknown = contradiction) clearly above 1.28.

---

## 4. Data conversion

### 4.1 Formats the code expects (verified in the repo)

**Query file**, for `--data_path` (JSON list; `scripts/run_attack.py: load_data`):

```json
{
  "id": "784", "qid": "784",
  "question": "...",
  "answer": "Jacqueline Georgette Cantrelle",
  "correct answer": "Jacqueline Georgette Cantrelle",
  "incorrect answer": "<target answer for corpus-poisoning runs>",
  "adv_texts": ["<PoisonedRAG-style poisoned passages>", "..."],
  "gold_docids": ["24653", "50079"],
  "evidence_docids": ["24653", "50079", "7625", "40264", "76395"]
}
```

- `answer` is the ground truth for EM/F1 (`run_attack.py`). `correct answer` is read by the `paradox` attack type.
- `incorrect answer` drives `asr` (`metrics.check_asr`).
- **Set both `id` and `qid`.** `generate_attack_doc_kill.py` reads `item["qid"]` in the `ours` path but `item["id"]` in others, and `run_attack.py` passes input keys through to the results file.

**Corpus file**, for `--corpus_path` (JSONL; `src/retrieval/indexer.py`, `e5_retriever.py`):

```json
{"id": "24653#0", "docid": "24653", "title": "Jacqueline Cantrelle - Wikipedia", "text": "...", "url": "https://..."}
```

- The retriever reads `_id|id`, `title`, and `text|contents`. Extra keys are ignored but kept in the corpus, so `docid` is available for dedup and analysis.
- Poisoned corpora are **separate** JSONL files. `is_poisoned` is set purely by index position (`idx >= num_clean_docs`), so poisoned docs must never be mixed into the clean file.

**ReliabilityRAG format**: emit it too, so the same questions can run through `wulver_benchmark.sh` / `main.py`. See `ReliabilityRAG/data/realtimeqa.json`:

```json
{"id": "784", "question": "...", "correct answer": ["..."], "incorrect answer": "...",
 "incorrect_context": ["<poisoned passages>"], "context": [{"title": "...", "text": "..."}],
 "expanded answer": ["<aliases>"]}
```

### 4.2 Converter: `Webthinker/scripts/prepare_browsecomp_plus.py` (new)

Run it once on a Wulver login node. Downloading needs network; compute nodes may not have it.

1. **Download and decrypt.** Clone `texttron/BrowseComp-Plus` and run `scripts_build_index/decrypt_dataset.py` (needs `huggingface-cli login`). This gives `browsecomp_plus_decrypted.jsonl`, with the same schema as `temp.json`: `query_id, query, answer, gold_docs, evidence_docs, negative_docs`. Download the corpus `Tevatron/browsecomp-plus-corpus`. Keep both under `/project/ss797/dk694/...`, not `$HOME`.
2. **Select queries.** Keep `len(gold_docs) >= 3`. If fewer than 100 qualify, take `>= 2` ordered by gold count. Seeded sample of **N = 100** to match the HotpotQA runs. Save the chosen IDs to `selected_qids.txt` for reproducibility.
3. **Build two corpus variants:**
   - `bcp_dev_corpus.jsonl`: the union of gold + evidence + negative docs for the selected queries (≈ 100 × 85 ≈ 8.5k docs). Fast to index; use it for smoke tests and development.
   - `bcp_full_corpus.jsonl`: all 100,195 docs. Use it for the final reported runs.
4. **Chunk** every doc (§4.3) and write the corpus JSONL above. Parse the `title:` line out of the `---` front-matter block. Strip the front matter from the text, or keep it only in the first chunk.
5. **Write queries** in the KidnapRAG format above. Leave `incorrect answer`/`adv_texts` empty for now (filled in by §4.4).
6. **Write a stats report:** gold/evidence counts, the chunk-length histogram, and the number of chunks per gold doc.

### 4.3 Chunking and context budget

- **Chunk size about 300 words with a 50-word overlap**, on paragraph boundaries where possible. This stays under E5's 512-token limit and keeps WebThinker's prompt size bounded: top-5 × ~400 tokens ≈ 2k tokens per search, × `max_search_limit 10` ≈ 20k, plus reasoning. Also start the gpt-oss server with `MAX_MODEL_LEN=65536`; gpt-oss supports 128k, and the server script reads this from the environment.
- Expected size: full corpus ≈ 1–2M chunks. E5-large fp16 embeddings ≈ 2–4 GB, which fits on one GPU for the dot-product search in `E5_Retriever.search`.
- `SimpleEncoder` already adds the E5 `query: ` / `passage: ` prefixes, so the converter must **not** add them again.

### 4.4 Attack material

- **KidnapRAG bait docs (`ours`, `BM…BCCCCM`)** are generated from the **clean WebThinker trajectory** (`step_stats[*].action/thought`), so they are dataset-agnostic. Flow: clean run → `generate_attack_doc_kill.py --dataset browsecomp_plus` → `concat.py --dataset browsecomp_plus`. The chain-link docs (`datasets/Chain_Link_and_Mal_Ins_DOCS/*.jsonl`) come from the original Drive bundle and are reused unchanged. **Do not chunk bait docs.**
- **Corpus poisoning (PoisonedRAG / ReliabilityRAG "Poison")**: generate `incorrect answer` plus 5 `adv_texts` per query with gpt-oss, using the same prompt `ReAct/results/adv_targeted_results/*.json` was made with. Write them to a separate poisoned JSONL. This is the attack the MIS defense is actually meant to stop, so it should be the headline defense experiment. KidnapRAG bait docs are the second axis.

---

## 5. Code changes in `KidnapRag/Webthinker`

| # | File | Change | Why |
|---|---|---|---|
| 1 | `scripts/run_attack.py`, `models/webthinker*.py` | Add a `--retriever_device` flag (default `cuda:0`) to replace the hard-coded `"cuda:2"` in indexing and the retriever | The LLM runs on the remote gpt-oss server, so the experiment job needs only **1 GPU**. Today it needs 3. |
| 2 | `src/retrieval/e5_retriever.py` | Add `--max_chunks_per_doc` (default 1): over-fetch `k*4`, then keep the best chunk per `docid` | Otherwise 5 chunks of one gold page "agree" and inflate the defense |
| 3 | `models/webthinker.py: generate_response` | Add a gpt-oss branch: if `'gpt-oss' in model_name`, render with the gpt-oss tokenizer's chat template (harmony) and handle `analysis`/`final` channels. Or switch to `chat.completions` with `reasoning_effort` | The current branch applies a chat template only for qwq/deepseek/r1 names. Search-tag stop strings inside the analysis channel need a **smoke test** (see Risks). |
| 4 | `scripts/generate_attack_doc_kill.py: main` | Derive the model tag `m` from a new `--gen_model` argument instead of the `clean_qwq.json` / `clean_deepseek_32b.json` filename check | With a gpt-oss result file, `m` is undefined and the script crashes |
| 5 | `scripts/generate_attack_doc_kill.py` | Optionally point the attacker LLM at the gpt-oss server instead of loading Qwen3-30B in-process (`KIDNAPRAG_ATTACKER_MODEL`) | Avoids a second GPU allocation. Keep Qwen3-30B if you want parity with the paper. |
| 6 | `scripts/run_attack.py` metrics | Add an LLM-judge correctness metric (reuse `ReliabilityRAG/llm_judge.py`) next to EM/F1 | BrowseComp answers are names and strings with variants. EM undercounts; the official benchmark grades with an LLM judge. |
| 7 | `models/webthinker_defense.py` | Add `--defense_method mis`: at each search, run isolated answering + NLI + MIS over the top-k (port from `ReAct/src/reliability_mis.py`) and pass only MIS docs to the model. Keep the read-only diagnostic mode too. | This is the defense the dataset change is for. Today WebThinker has only spotlight, sandwich, and taskshield. |

Keep all changes behind flags so the existing HotpotQA/QwQ scripts behave identically.

---

## 6. Running on Wulver

### 6.1 New wrapper: `KidnapRag/scripts/wulver_webthinker_with_server.sh`

Copy the structure of `ReliabilityRAG/scripts/wulver_run_with_server.sh` and reuse its server instead of starting a new one:

- **Phase 1 (login node):** dry-run check (data and corpus files exist, `check_env`). Then reuse or wait for an existing `gpt-oss-server` job (`squeue -n gpt-oss-server`), or `sbatch ReliabilityRAG/scripts/serve_gpt_oss_local.slurm` with `MAX_MODEL_LEN=65536`. Queue a CPU launcher with `--dependency=after:<server>`.
- **Phase 2 (launcher):** poll `~/vllm_local_server.txt` until `/v1/models` answers. Then `sbatch` the experiment job, exporting `API_BASE=http://$(cat ~/vllm_local_server.txt)/v1`.
- **Experiment job:** 1 GPU (`--gres=gpu:1`, enough for E5 retrieval), 8 CPUs, 64 GB. Use the `KidnapRAG` conda env. Run:

  ```sh
  python3 scripts/run_attack.py --model webthinker \
    --api_base_url "$API_BASE" --model_name openai/gpt-oss-120b --model_path $GPT_OSS_DIR \
    --aux_api_base_url "$API_BASE" --aux_model_name openai/gpt-oss-120b --aux_model_path $GPT_OSS_DIR \
    --data_path datasets/browsecomp_plus/bcp_select100.json \
    --corpus_path datasets/browsecomp_plus_corpus/bcp_dev_corpus.jsonl \
    [--poisoned_corpus_path ...] --retriever_device cuda:0 \
    --max_search_limit 10 --top_k 5 --concurrent_limit 16 --output_path results/browsecomp_plus/<tag>.json
  ```

  Keep `--concurrent_limit` moderate (≈16), since the same server also serves ReliabilityRAG runs.
- Keep the account, partition, and QOS from `ReliabilityRAG/scripts/wulver_run_one.slurm` (`ss797`, `gpu`/`general`, `standard`). Put logs in `KidnapRag/logs/`.
- `STOP_SERVER=1` behaves as in the ReliabilityRAG script.

### 6.2 Index builds as their own job

`run_attack.py` builds the index automatically if it is missing. For the full corpus (1–2M chunks), submit `src/retrieval/build_index.py` as a separate 1-GPU job (≈ 2–4 h estimated; time it on the dev corpus first). That way an experiment job never spends its walltime encoding. Indexes go next to the corpus (`indexer.derive_index_path`).

### 6.3 Run order

| Stage | Job | Output |
|---|---|---|
| A | Converter (login node) | queries + dev/full corpora + stats |
| B | Agreement probe (§3) | go/no-go numbers |
| C | Dev index + **smoke test**: 2 queries, clean, `--dry_run` | proves gpt-oss + WebThinker tags + retriever work end-to-end |
| D | Clean run, 100 queries, dev corpus | `clean_gptoss.json`; pick the clean-correct subset |
| E | Bait docs + `concat.py` (BM…BCCCCM); PoisonedRAG adv_texts | poisoned JSONLs |
| F | Attack runs: {Poison, BM, BCCM, BCCCCM} × {none, MIS, spotlight, sandwich} | results JSONs |
| G | Full-corpus index, then repeat D/F for the final numbers | final results |
| H | `eval_asr.py` (hijack sequence) + LLM-judge accuracy + MIS diagnostic plots | tables/plots |

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| gpt-oss doesn't follow WebThinker's `<\|begin_search_query\|>` protocol through the raw completions API (harmony channels, special-token stripping) | Stage C smoke test first. If it fails, use `chat.completions` with tool-style instructions, or keep QwQ-32B as the agent and use gpt-oss only as the aux model, judge, and attacker. |
| Low clean accuracy leaves few clean-correct queries | Select more than 100 candidate queries (e.g. 200) and filter after the clean run. Also report the oracle setting (gold docs injected into the top-k) as an upper bound. |
| Agreement still too low | Probe first (§3). Then tighten to gold ≥ 3 and allow 2 chunks per doc only for gold docs in the diagnostic, or fall back to MultiHop-RAG inference queries. |
| Full-corpus index cost or walltime | Use the dev corpus for development. Run the full index as a standalone job; it is reusable across all runs. |
| BrowseComp data leakage | Keep the decrypted files out of git (`datasets/` is already ignored) and don't publish the decrypted text. This respects the benchmark's canary. |
| Shared server overload | Use one server for both projects with bounded `concurrent_limit`. Use `STOP_SERVER=1` only on the last run. |

---

## 8. Open decisions for you

1. Who is the agent model: **gpt-oss-120b for everything** (simplest; one server) or **QwQ-32B agent + gpt-oss aux** (closer to the KidnapRAG paper)?
2. What is the attacker model for bait docs: Qwen3-30B-A3B (paper parity) or gpt-oss (no extra GPUs)?
3. Is the headline defense experiment **PoisonedRAG-style corpus poisoning** (what MIS is for), KidnapRAG chain hijacking, or both?

---

### Sources
- BrowseComp-Plus repo: https://github.com/texttron/BrowseComp-Plus
- BrowseComp-Plus HF dataset card (schema, obfuscation, MIT): https://huggingface.co/datasets/Tevatron/browsecomp-plus
- BrowseComp-Plus stats (830 queries, 100,195 docs, 6.1 evidence / 2.9 gold, 76.3 negatives, median 5,179 words, baselines): https://www.emergentmind.com/topics/browsecomp-plus-benchmark
- MultiHop-RAG repo: https://github.com/yixuantt/MultiHop-RAG
- MultiHop-RAG paper (609 articles, 2,556 queries, type and evidence-count breakdown, 256-token chunks): https://arxiv.org/pdf/2401.15391
- Repo files referenced: `KidnapRag/Webthinker/scripts/run_attack.py`, `src/retrieval/{indexer,e5_retriever}.py`, `models/webthinker.py`, `scripts/generate_attack_doc_kill.py`, `concat.py`, `ReAct/MIS_DIAGNOSTIC_README.md`, `ReAct/results/mis_diagnostic/hotpotqa_qwen25_mis_summary.json`, `ReliabilityRAG/scripts/{wulver_run_with_server.sh,serve_gpt_oss_local.slurm,wulver_benchmark.sh}`, `ReliabilityRAG/data/realtimeqa.json`
