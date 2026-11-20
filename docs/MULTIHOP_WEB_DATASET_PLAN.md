# Action Plan: Multihop dataset built with the ReliabilityRAG method, for KidnapRAG ReAct

_Prepared 2026-11-18. Repo state analysed: `KidnapRag` (branch `dataset-builder`, = `main` @ 4dbbc48), `ReliabilityRAG/data/*`, both papers. Companion docs: `webthinker-dataset-migration-plan.md` (BrowseComp-Plus route), `KidnapRag/docs/HANDOFF.md`._

---

## 0. TL;DR

- **The goal:** keep the KidnapRAG multihop questions (MuSiQue, 2Wiki, HotpotQA, 100 each, same qids as the paper). Rebuild their retrieval corpus the way ReliabilityRAG built its data: Google search results (SerpAPI organic snippets), checked for relevance by whether they contain the answer verbatim, with an optional "allrel" cleaned variant and GPT-generated incorrect answer and context.
- **The key change from ReliabilityRAG:** RRAG searches once per question and stores a fixed list of 30–50 snippets. A ReAct agent instead issues its own sub-queries, so a fixed list does not work. **We search Google for every hop**: MuSiQue's gold decomposition, 2Wiki's evidence triples, and an LLM decomposition for HotpotQA. All snippets go into one E5-indexed corpus. Each hop then has about 20 independent web sources, and several of them state that hop's answer. This is the honest majority that HotpotQA lacked (mean MIS 1.28/5).
- **It complements the BrowseComp-Plus route; it does not replace it.** It keeps the paper's datasets and questions, so results stay comparable with the KidnapRAG tables. It also fixes the agreement problem in the same way RRAG's data does.
- **Build:** 8 small scripts in a new `KidnapRag/dataset_builder/` (this branch). Then about 5 targeted ReAct changes: a generic dataset wrapper, dataset paths, per-URL dedup in the retriever, and hop labels in `info`.
- **Three go/no-go gates** before any attack run: (G1) snippet relevance stats, (G2) a static per-hop MIS probe, (G3) clean ReAct step-level agreement.

---

## 1. What the repos actually do (analysis)

### 1.1 ReliabilityRAG dataset method (`ReliabilityRAG/data/`)

| Step | Script | What it does |
|---|---|---|
| Retrieve | `google_search.py` | SerpAPI `engine=google, hl=en, gl=us, num=50` with the **question** as the query. Keeps `organic_results[:N]` as `{title, text=snippet, link}`. The answer box is commented out. No page download and no chunking. |
| Augment | `process.py` | GPT-4o generates: 2 **expanded answers** (paraphrases of the gold answer), 1 **incorrect answer**, and 5 **incorrect_context** passages supporting it (PoisonedRAG-style). It reprompts until the tags parse. |
| Rerank | `sort.py` | mxbai-rerank-large-v2 reorders `context` by (question, snippet). (In our copies, `*_sorted.json` has the same order as the unsorted files, so the runs used Google's order.) |
| Relevance | `check_relevance.py` | A doc is relevant if any gold answer appears **verbatim** (case-insensitive) in its title or text. It reports the per-question fraction and plots a CDF. Paper App. C.5: only 26% (RQA), 31% (NQ) and 19% (TQA) of docs contain the answer. |
| Clean | `make_all_rel.py` (+ GPT-4o rephrase in the paper) | "allrel": each irrelevant doc is replaced, round-robin, by a **rephrased copy** of a relevant doc. Used only for the multi-position analysis. |

Output schema (one item): `id, question, correct answer[list], expanded answer[2], incorrect answer, incorrect_context[5], context[{title,text,link}]×30–50`.

**Important: SerpAPI behaviour has changed since that data was built.** Google stopped honouring `num` in September 2025, and SerpAPI's docs now say the parameter is no longer supported. `num=50` now returns about 10 results. To get 20–30 results you must page with `start=0,10,20`, at **one paid search per page**.

**Security:** `google_search.py` has a **live SerpAPI key hard-coded and committed**. Rotate it and read the key from `SERPAPI_API_KEY` in the new code.

### 1.2 KidnapRAG ReAct pipeline (`KidnapRag/ReAct/`)

- **Agent:** `ReAct/attack_react.py` (and `attack_react_mis.py` for the MIS diagnostic). `E5WikiEnv.search_step` replaces Wikipedia search with `E5_Retriever.search(query, k=5)`. The observation is `title\ncontents` for all 5 docs. Bait/chain docs are flagged `is_poisoned` purely by index position (`idx >= num_clean_docs`).
- **Data wiring is hard-coded per dataset:**
  - Questions come from `ReAct/data/{hotpot,musique,wikimultihopqa}_dev_v1_simplified.json` through `wrappers.{HotPotQA,musiqueA,wikimultihopqa}Wrapper`. These files hold only `question, answer, type`.
  - The run order and attack targets come from `results/adv_targeted_results/{dataset}_qid_to_idx.json` + `{dataset}.json` (`id, question, correct answer, incorrect answer, adv_texts`).
  - The corpus and index come from `datasets/{dataset}/corpus.jsonl` and `datasets/{dataset}/e5_index`. The retriever reads `_id|id`, `title`, `text|contents`.
  - The poisoned corpus is `datasets/{dataset}/REACT_{attack}_{model}.jsonl`, a separate file.
- **Paper corpora:** HotpotQA uses BEIR (5.2M Wikipedia abstracts). MuSiQue uses its 20 provided paragraphs per question, and 2Wiki its 10 context paragraphs. So per hop there are **1–2 supporting paragraphs**. This is the structural reason the MIS has no honest majority.
- **MIS diagnostic:** `src/reliability_mis.py` + `attack_react_mis.py`. Isolated answering per doc → DeBERTa NLI graph → rank-aware MIS, with unknown=contradiction and an NLI-only column. Read-only. This is reused unchanged as the evaluation.

### 1.3 What the new dataset must provide

At a search step, the top-5 should contain **≥ 3 independent docs that each state that step's answer**. Only then do "bait inside/outside the MIS" and weakness-vs-hijack mean anything. ReliabilityRAG's web data gives that kind of redundancy for single-hop questions. The plan below gives it **per hop**.

---

## 2. Design: "MH-Web" (per-hop web search corpus)

### 2.1 Questions and hops

Use the **same 100 qids per dataset** as `results/adv_targeted_results/*_qid_to_idx.json`. Keep them in the same order, so existing clean runs and attack scenarios stay comparable. Load the **raw** dev files, because the simplified ones have dropped the hop information.

| Dataset | Raw file | Hop source | Hop query construction |
|---|---|---|---|
| MuSiQue | `musique_ans_v1.0_dev.jsonl` | `question_decomposition[*]` (sub-question, sub-answer, supporting paragraph) | Substitute `#k` with the previous hop's answer. Rewrite `entity >> relation` forms into a natural search query (gpt-oss, 1 call). Keep the original string too. |
| 2Wiki | `dev.json` | `evidences` triples `[subj, relation, obj]` (+ `supporting_facts`) | Template `relation` → question ("Who directed X?"). Hop answer = `obj`. Comparison questions → one hop per compared entity. |
| HotpotQA | `hotpot_dev_distractor_v1.json` | `supporting_facts` titles + `type` | gpt-oss decomposes into 2 hops given the question, answer, and supporting titles. Each hop's answer must appear in its supporting paragraph (auto-check), otherwise retry/drop. Yes/no comparison questions → hops are the two attribute lookups. |

Output `hops.jsonl` per question: `qid, question, answer, answer_aliases, expanded_answers[2], hops[{hop_idx, query, query_paraphrase, answer, aliases, support_title}]`. Expanded answers are made with the `process.py` prompt on gpt-oss, plus the dataset-native aliases.

### 2.2 Web retrieval (the ReliabilityRAG step, per hop)

For each question, run SerpAPI on these queries:
1. the full question (gives RRAG's original setting and catches pages that answer end to end),
2. each hop query,
3. one paraphrase per hop query (more source diversity; agents phrase queries differently).

Settings: `hl=en, gl=us`, **2 pages (`start=0,10`) ≈ 20 organic results per query**. Parse with RRAG's `parse_google_research_results` (title, snippet, link). **Cache every raw response** (`serp_raw/<sha1(query)>.json`) so the job is resumable and never pays twice.

**Leakage filter (required):** drop results from domains that host the datasets or Q&A dumps: `huggingface.co, github.com, kaggle.com, paperswithcode.com, arxiv.org, aclanthology.org, semanticscholar.org, researchgate.net, quizlet.com`, plus any result whose title or snippet contains the full question with ≥ 0.8 token overlap. Otherwise the full-question search returns dataset viewer pages that answer the whole chain and short-circuit the multihop.

**Estimated call volume (2 pages each):** about 100 Q × (1 + 2.5 hops × 2) × 2 ≈ **1,200 searches per dataset**, ≈ 3,600 for all three, plus ≤ 1,000 for the coverage round (§2.6). Check this against the SerpAPI plan quota before running.

### 2.3 Document format and corpus

Each kept result becomes one doc:

```json
{"_id": "web::<sha1(url)>::<sha1(serp_query)>", "docid": "<sha1(url)>", "title": "...", "text": "<snippet>",
 "url": "...", "source": "serp", "serp_query": "...", "serp_rank": 3,
 "qid": "2hop__67660_81007", "hop_idx": 1, "answer_hits": {"final": false, "hop1": true}}
```

- Google snippets depend on the query, so the same URL gives different snippets for different queries. Keep each one as its own doc, but give them a shared `docid` (= URL hash) so the retriever can dedup to **one doc per URL in the top-k** (port `max_chunks_per_doc` from the WebThinker branch). Otherwise one page could fill the majority on its own.
- **Corpus = web snippets ∪ the dataset's original paragraphs** (MuSiQue 20/q, 2Wiki 10/q; for HotpotQA, its distractor paragraphs, not the 5.2M BEIR corpus, to keep it tractable). Snippets collected for other questions act as natural distractors. The original Wikipedia gold paragraphs stay retrievable, so this corpus is a strict superset of the paper's evidence.
- Metadata (`qid, hop_idx, answer_hits, source`) is ignored by the prompt and used only for analysis.

### 2.4 Snippet length (decision point)

RRAG uses raw snippets (median ~150 chars, about 76% truncated with "..."). For ReAct that is thin but workable: 5 × 150 chars per observation.
- **v1 (default, faithful to RRAG): snippet only.**
- **v2 (optional): snippet-expanded.** Fetch the page (trafilatura) and keep a ~150–250-word window around the snippet's location. Fetching needs network, so do it on a login node or locally. Build v2 only if G3 shows that agents fail because snippets lack the answer.

### 2.5 Relevance labels, filtering, and RRAG variants

- **Relevance:** port `check_relevance.contains_answer` to **per hop**. A doc is "hop-k-bearing" if the hop answer or one of its aliases appears verbatim. Also record whether it bears the final answer. Report per-hop fractions plus CDF plots (same plot code), next to RRAG's 26/31/19% for comparison.
- **Question filter:** keep a question only if **every hop has ≥ 3 answer-bearing docs from ≥ 3 distinct URLs**. This also drops questions where the web now disagrees with the 2019–2020 Wikipedia gold answer (temporal drift). If fewer than 100 of the paper's qids pass, top up from the same dev split (seed 1) and record the replacements in `selected_qids.txt`. Report results on both the original-overlap subset and the full 100.
- **allrel variant** (RRAG cleaning, per hop): for hops with fewer than K=5 answer-bearing docs, add gpt-oss **rephrased copies** of the answer-bearing docs (round-robin) until there are K. Flag them `synthetic_rephrase: true` and give them their own `docid`. Use this only as an ablation: it inflates agreement by construction, which is why RRAG used it only for the multi-position study.
- **Poisoning material** (`process.py` prompts on gpt-oss): an `incorrect answer` plus 5 `incorrect_context` passages, per final answer (and optionally per hop). This gives the PoisonedRAG/corpus-poisoning baseline. KidnapRAG bait/chain docs are still generated from the clean trajectories by `generate_attack_doc_kill.py`; they don't depend on the dataset.

### 2.6 Coverage round (closing the agent-query gap)

The agent's own queries won't match ours exactly. After the first clean ReAct run, collect every agent search query whose top-5 has **fewer than 3 hop-bearing docs**. SerpAPI those exact queries (≤ 1 per step, deduplicated), add the results to the corpus, rebuild the index, and re-run clean. Do **one round only**, so the corpus isn't fitted to a single trajectory. Never expand from attack runs: chain-link queries are the attacker's.

### 2.7 Outputs (all gitignored; never commit raw SERP data)

| File | Consumer |
|---|---|
| `datasets/mhweb_{ds}/corpus.jsonl` (+ `e5_index/`), `corpus_allrel.jsonl` | ReAct retriever (and WebThinker `--corpus_path`) |
| `ReAct/ReAct/data/mhweb_{ds}_dev_simplified.json` | new generic wrapper |
| `results/adv_targeted_results/mhweb_{ds}.json` + `_qid_to_idx.json` | `attack_react.py` main loop, attack targets |
| `datasets/mhweb_{ds}/hops.jsonl`, `hops_rrag.json` (each hop as an RRAG item with `context`) | MIS diagnostic labels; G2 probe through `ReliabilityRAG/main.py` |
| `datasets/mhweb_{ds}/webthinker_queries.json` (`id, qid, question, answer, correct answer, gold/evidence docids`) | WebThinker runner, if you want the same data there |
| `stats.json`, `relevance_cdf_*.png`, `selected_qids.txt` | reporting |

---

## 3. Implementation: `KidnapRag/dataset_builder/` (new, on `dataset-builder`)

| # | Script | Runs on | Notes |
|---|---|---|---|
| 1 | `01_build_hops.py --dataset {musique,2wiki,hotpotqa}` | login/CPU + gpt-oss server | Raw dev file + qid list → `hops.jsonl`. Uses gpt-oss for the MuSiQue `>>` rewrite, the HotpotQA decomposition, and expanded answers. Validates hop answers against the support paragraphs. |
| 2 | `02_serp_collect.py` | node with network | SerpAPI with pagination, raw cache, retries/backoff, leakage filter, `--max_calls` budget guard, `--dry_run` (prints queries + call count). |
| 3 | `03_fetch_expand.py` (v2 only) | network node | trafilatura window extraction. |
| 4 | `04_label_relevance.py` | CPU | per-hop `answer_hits`, stats, CDFs (port of `check_relevance.py`). |
| 5 | `05_build_corpus.py` | CPU | question filter → corpus union + all output files in §2.7. |
| 6 | `06_make_allrel.py` | gpt-oss | RRAG cleaning variant. |
| 7 | `07_poison_material.py` | gpt-oss | incorrect answer/context (`process.py` prompts, tag-parse + reprompt). |
| 8 | `08_expand_from_traces.py` | network | coverage round from a clean results JSON (§2.6). |
| — | `common.py` | — | shared I/O, SerpAPI client (key from env), gpt-oss client (`API_BASE` from `~/vllm_local_server.txt`), alias normalisation. |

**ReAct changes** (behind flags; the existing hotpotqa/musique/2wiki paths stay identical):

| File | Change |
|---|---|
| `ReAct/ReAct/wrappers.py` | Add a generic `JsonQAWrapper(data_file)` (question, answer, qid). |
| `ReAct/ReAct/attack_react.py`, `attack_react_mis.py` | Accept `mhweb_*` datasets → `JsonQAWrapper`. Corpus path from `datasets/{dataset}/corpus{--corpus_variant}.jsonl`. Add `--max_docs_per_url` (default 1). |
| `ReAct/src/e5_retriever.py` | Over-fetch k×4, then keep the top doc per `docid` (poisoned docs have no `docid` and are never merged). Pass `docid, url, qid, hop_idx, answer_hits` through in results, so they reach only `info['retrieved_docs']` and never the observation text. |
| `ReAct/ReAct/attack_react_mis.py` | Per step, log `n_hop_bearing` (docs bearing **any** hop answer of this question) and `n_final_bearing`, beside MIS size. Use these in G3. |
| `ReAct/ReAct/generate_attack_doc_kill.py`, `concat.py` | Accept the new dataset names (paths only). |
| `scripts/` | `wulver_build_index.slurm` (1 GPU, `ReAct/scripts/build_index.py`) and a dataset-builder Slurm/login wrapper. Reuse the gpt-oss server pattern from `ReliabilityRAG/scripts/wulver_run_with_server.sh`. |

---

## 4. Run order with gates

| Stage | Action | Gate / output |
|---|---|---|
| A | Rotate the SerpAPI key; fetch the raw MuSiQue/2Wiki/HotpotQA dev files to `/project/ss797/dk694/...` | — |
| B | `01_build_hops` for **MuSiQue first** (the cleanest gold decomposition); inspect 10 by hand | hop queries make sense; ≥ 95% of hop answers found in their support paragraphs |
| C | `02_serp_collect --dry_run`, then a 10-question pilot (≈ 120 calls) | leakage filter rate; snippets per query |
| D | `04_label_relevance` on the pilot, then the full 100 | **G1:** mean ≥ 5 hop-bearing snippets per hop (of ~20) and ≥ 80% of hops with ≥ 3 distinct-URL bearers. If it fails, add a 3rd page or more paraphrases before building anything. |
| E | `05_build_corpus` → E5 index (Slurm) | corpus/index stats |
| F | **G2 static probe:** for each hop, an "ideal" top-5 = 3 hop-bearing + 2 random corpus docs, and separately the actual E5 top-5 for the hop query. Run `reliability_mis.py` isolated answering + MIS. | MIS size (unknown=contradiction) clearly above **1.28/5**, ideally ≥ 3 on the E5 top-5 |
| G | Clean ReAct run (`attack_react_mis.py`, 100 q) | **G3:** ≥ 60% of search steps with `n_hop_bearing ≥ 3`; mean MIS size; EM / LLM-judge vs the paper's clean numbers |
| H | `08_expand_from_traces` (one round) → re-index → re-run clean if G3 fell short | — |
| I | Bait docs from the clean run (`generate_attack_doc_kill.py`) → `concat.py` → poisoned indices; `07_poison_material` for the PoisonedRAG baseline | `REACT_{BM..BCCCCM}_{model}.jsonl` |
| J | Attack runs BM…BCCCCM × defenses, plus MIS diagnostic on each | ASR, bait-retrieved/taken, bait-in-MIS rate, weakness at bait steps (taken vs not) |
| K | Repeat B–J for 2Wiki, then HotpotQA | three-dataset tables comparable to KidnapRAG Tables 2/6/7 |
| L | Ablations: `allrel` corpus; v2 snippet-expanded (only if needed) | — |

Rough effort: builder scripts 2–3 days; MuSiQue through G3, 2–3 days of wall-clock (mostly the SerpAPI pilot and the clean run); each further dataset about 1 day plus runs.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Web answers disagree with old Wikipedia gold (drift), or hops are too obscure for Google | The per-hop ≥3-bearer filter drops these questions; report how many were dropped and why |
| Full-question search leaks the end-to-end answer, so the agent can skip hops | Leakage domain filter + question-overlap filter. Also report the share of steps whose docs bear the **final** answer, and an ablation with full-question results removed from the corpus |
| Snippets too short or truncated for the agent to extract the answer | G3 shows it (low EM despite high `n_hop_bearing`); then build v2 snippet-expanded |
| Agreement inflated by duplicate pages or syndicated copies | One doc per URL in top-k; also near-duplicate snippet dedup (MinHash ≥ 0.9) within a hop at build time |
| The bait must now out-rank many relevant snippets, so the bait-retrieved rate could drop vs the paper | That is a realistic result in itself. Measure the bait-retrieved rate first (BM × none) before the full grid |
| SerpAPI cost/quota (pagination is now 1 call/page) | Raw cache, `--max_calls` guard, MuSiQue-first pilot. Alternative backend: the Apify RAG web browser actor (already connected on your machine) returns search results + page text, but rankings and snippets differ from Google's, so don't mix backends within one dataset |
| Reproducibility (Google results change over time) | The frozen cache + corpus *is* the dataset. Record the collection date and never re-query for existing qids |
| Publishing scraped text | Keep `serp_raw/` and corpora out of git; release only queries, URLs, and the build scripts |

---

## 6. Decisions needed (recommendations first)

1. **Datasets and order:** MuSiQue → 2Wiki → HotpotQA *(recommended: MuSiQue has gold hop decompositions)*.
2. **Doc text:** v1 snippet-only, as in RRAG *(recommended)*; v2 page-window only if G3 fails.
3. **Question set:** the paper's 100 qids, topped up after filtering *(recommended)*, vs. a fresh seeded sample of 100 that pass the filter.
4. **Search backend and budget:** SerpAPI, 2 pages/query, ≈ 3.6k + 1k calls total *(recommended)* vs. the Apify actor.
5. **Relationship to BrowseComp-Plus:** run MH-Web as the main ReAct dataset and keep BCP for WebThinker, or run both on both.

---

### Sources
- ReliabilityRAG paper (arXiv 2509.23519), App. C.1 (Google Search + mxbai rerank) and C.5 (26/31/19% relevance, round-robin GPT-4o rephrase cleaning)
- KidnapRAG paper (arXiv 2607.00422), App. B (100 q/dataset; BEIR corpus for HotpotQA; MuSiQue 20 paragraphs, 2Wiki 10 paragraphs; e5-large-v2, top-5)
- Repo files: `ReliabilityRAG/data/{google_search,process,sort,check_relevance,make_all_rel}.py`; `KidnapRag/ReAct/ReAct/{attack_react,attack_react_mis,e5_env,wrappers}.py`; `ReAct/src/{e5_retriever,reliability_mis}.py`; `ReAct/results/adv_targeted_results/*`; `docs/HANDOFF.md`
- SerpApi, "Google experiments with restricting results per page": https://serpapi.com/blog/google-experiments-with-restricting-results-per-page/
- Bright Data SERP API parameter docs (`num` deprecated 2025-09-11; use `start` paging): https://docs.brightdata.com/products/serp-api/query-parameters/google
