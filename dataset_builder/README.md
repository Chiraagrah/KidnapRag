# Dataset-builder search collection

## Status (2026-11-20): use `02_keenable_collect.py`

The search backend moved from SerpAPI to **Keenable** (plan doc section 2.2a).
`02_keenable_collect.py` keeps the same input, output, cache, resume and
budget-guard design as `02_serp_collect.py`, plus:

- one call returns up to 50 results (`--max-results`, default 50), no paging
- leakage domains are excluded server-side (`exclude_domains`), with the
  client-side filter as a backstop
- `--snippet-max-length` (default 2000), `--mode pro|realtime`, `--query-time`
- adds `published_at` / `acquired_at` to each result
- failed searches are retried on rerun, and the run aborts after 3 failures in
  a row (`--max-consecutive-failures`)

### API key

The key lives in `dataset_builder/.env` (gitignored, never commit it):

```
KEENABLE_API_KEY=keen_...
```

The script loads `.env` from the current directory or from its own folder. A
real shell `export KEENABLE_API_KEY=...` overrides the file. Without a key it
uses the keyless `/v1/search/public` endpoint (1000 requests/hour, 10/sec per
IP). On Wulver, copy `.env` next to the script with `chmod 600 .env`.

### Run

```sh
cd dataset_builder
python3 02_keenable_collect.py --input queries_sample.json --out searches_sample.json --dry-run
python3 02_keenable_collect.py --input queries_sample.json --out searches_sample.json --max-calls 2
python3 02_keenable_collect.py --input queries_sample.json --out searches_sample.json
python3 test_02_keenable_collect.py     # offline tests, no network or key needed
```

Needs Python >= 3.10 and outbound HTTPS to `api.keenable.ai` (run it on a
Wulver login node, not a compute node). Never commit `keenable_raw/` or
`searches*.json`.

---

## Earlier status: `02_serp_collect.py` (SerpAPI), not the chunked Claude-agent collector

Two approaches were tried for the per-hop web search collection step of the
MH-Web dataset build (see `claude/multihop-web-dataset-plan.md` in the RAG
Security project):

1. **`02_serp_collect.py` (recommended, verified working)** -- plain SerpAPI
   calls via `requests`-style HTTP, matching the original ReliabilityRAG
   method (`ReliabilityRAG/data/google_search.py`). Returns Google's own
   pre-rendered organic snippets directly. Fully unit-tested (caching,
   leakage filter, resume, budget guard all verified to work -- see "Testing
   notes" below).

2. **`driver.py` + `agent_prompt_template.txt` (abandoned for this step)** --
   spawns headless `claude -p` subprocesses to do the searching with
   Claude's own WebSearch/WebFetch tools. **Smoke-tested on 17 real searches
   and found to have a ~7% usable-snippet rate**: WebSearch's own snippet
   field is unreliable/often empty, so the prompt had the agent WebFetch
   each top result for a real excerpt -- but most fetches came back HTTP
   403/404/405 (site-level bot-blocking), and only the top 3-4 of each
   search's 10 results were ever attempted at all (the rest stayed
   `"fetch_error":"not_in_top_3"` by design, to bound cost). Net: out of
   ~170 result slots in the smoke test, about 12 had real text. This also
   repeatedly tried to write its output via a generated Python script it
   then couldn't execute (no Bash/PowerShell permission), burning 2 of 3
   attempts and real usage-window cost before falling back to writing the
   JSON directly. Kept in this folder for reference / in case WebFetch
   reliability improves, but **don't use it for the real 700-search run.**

See `KNOWN_ISSUES.md` for the full failure trail if you want the details.

## Files

- `02_serp_collect.py` — **use this one.** Standalone script, SerpAPI only,
  no `claude` subprocess dependency. Per-hop collection, raw-response cache
  (resumable, never pays twice), leakage-domain filter, `--max-calls` budget
  guard, `--dry-run`.
- `make_sample_input.py` — builds a tiny `queries_sample.json` (3 questions)
  from the schema below, so you can smoke-test the pipeline end to end before
  running it on all 100 questions.
- `driver.py`, `agent_prompt_template.txt`, `merge.py` — the abandoned
  chunked Claude-agent collector (see status note above). Kept for
  reference only.

## Input format (`queries.json`)

One entry per question, produced by `01_build_hops.py` (plan §2.1). If that
script doesn't exist yet, hand-write a few entries in this shape to test:

```json
[
  {
    "qid": "2hop__67660_81007",
    "question": "Who does the narrator on How I Met Your Mother end up with?",
    "hops": [
      {"hop_idx": 0, "query": "Who narrates How I Met Your Mother?",
       "paraphrase": "How I Met Your Mother narrator identity"},
      {"hop_idx": 1, "query": "Who does Ted Mosby end up with?",
       "paraphrase": "Ted Mosby's eventual wife"}
    ]
  }
]
```

`hops` may have 1-3 entries. The driver always issues one additional
"full_question" search per qid (the question text itself), matching the
ReliabilityRAG method.

## Output format (`results/chunk_<i>.json`)

A JSON array, one object per search, appended as the agent works:

```json
{
  "qid": "2hop__67660_81007",
  "search_type": "full_question",
  "hop_idx": null,
  "query_text": "Who does the narrator on How I Met Your Mother end up with?",
  "results": [
    {"title": "...", "url": "https://...", "snippet": "..."}
  ],
  "timestamp": "2026-11-19T22:04:11Z"
}
```

`search_type` is one of `full_question`, `hop`, `hop_paraphrase`.

## Running `02_serp_collect.py`

```sh
# 0. one-time: get a SerpAPI key and rotate the one hard-coded in
#    ReliabilityRAG/data/google_search.py (that key is live and committed
#    to git -- rotate it regardless of whether you use this script).
export SERPAPI_API_KEY="your-key-here"     # never hard-code it in the script

# 1. smoke test on 3 questions (17 searches, ~17 SerpAPI calls at 1 page each)
python3 make_sample_input.py
python3 02_serp_collect.py --input queries_sample.json --out searches_sample.json --dry-run
python3 02_serp_collect.py --input queries_sample.json --out searches_sample.json

# 2. inspect searches_sample.json -- real Google snippets, no WebFetch/403
#    wall. Then run the full 700-search job:
python3 02_serp_collect.py --input queries.json --out searches.json
```

Resumable by design: rerun the same command and it picks up exactly where
it left off (reads the existing `--out` file, skips anything already
there, and the raw-response cache in `serp_raw/` means even a search
that's missing from `--out` won't re-pay for a page it already fetched).
Use `--max-calls N` to cap a single run's new API spend while testing.

`--pages` controls how many 10-result pages to pull per search (default 1 =
10 results, matching the scope you specified). Google stopped honoring the
old `num=50` parameter in Sept 2025 — each page is its own billed call now,
so raising `--pages` multiplies cost roughly linearly (see
`claude/multihop-web-dataset-plan.md` for the pricing math).

**Never commit `serp_raw/` or any `searches*.json`** — they contain scraped
web content. Add them to `.gitignore` alongside the rest of `datasets/`.

## Testing notes

`02_serp_collect.py`'s pure logic (leakage-domain filter, `parse_organic_results`,
`expected_searches`, the `fetch_search` cache, and `main()`'s budget-guard +
resume behavior) was verified with a monkeypatched `call_serpapi` standing in
for the real network call — no SerpAPI key was available in the environment
this was written in. Confirmed: cache hits avoid re-calling; `--max-calls`
stops mid-run cleanly; a second run resumes and completes the rest; the
leakage filter drops `huggingface.co`/`github.com`/etc. and keeps everything
else. **Not yet exercised against the real SerpAPI endpoint** — run the
3-question smoke test first and sanity-check the actual response shape
before trusting it on the full 700-search job (SerpAPI's JSON schema is
assumed from `ReliabilityRAG/data/google_search.py`'s usage of it, which may
have drifted since that script was last run).
