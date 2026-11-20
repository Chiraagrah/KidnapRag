#!/usr/bin/env python3
"""Per-hop SerpAPI collection (ReliabilityRAG method, applied per hop).

Replaces the Claude-agent WebSearch+WebFetch collector: that approach hit a
~7% usable-snippet rate in testing (most WebFetch calls were blocked with
HTTP 403/404/405, and most results were never fetched at all because of the
per-search fetch cap). SerpAPI returns Google's own pre-rendered organic
snippets directly -- no page fetch, no bot-blocking wall, and it's what the
ReliabilityRAG dataset itself was built with (see
ReliabilityRAG/data/google_search.py).

Reads queries.json (see README.md / make_sample_input.py for the schema:
one entry per qid with "question" and a "hops" list of {hop_idx, query,
paraphrase}). For each entry, issues one SerpAPI call per: the full
question, each hop's query, and each hop's paraphrase -- same search plan
the chunked-agent collector used, so output is drop-in comparable.

IMPORTANT: Google stopped honoring the `num` parameter in Sept 2025 (see
plan doc "multihop-web-dataset-plan.md" sources). Each call now returns
~10 results regardless of `num`; to get N results per query, page with
the `start` parameter (start=0, 10, 20, ...) across N/10 calls. This script
defaults to 1 page (10 results) per query to match the 10-results-per-search
scope you specified; raise --pages to pull more via extra paid calls.

Caches every raw response to serp_raw/<sha1(query)>.json so the job is
resumable and never pays twice for the same query.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

# Leakage filter: domains likely to host the dataset itself or Q&A dumps
# that would hand the agent the final answer outright (plan doc section
# "Leakage filter (required)").
LEAKAGE_DOMAINS = {
    "huggingface.co", "github.com", "kaggle.com", "paperswithcode.com",
    "arxiv.org", "aclanthology.org", "semanticscholar.org",
    "researchgate.net", "quizlet.com",
}


def query_hash(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()


def is_leakage_domain(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    host = host.removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in LEAKAGE_DOMAINS)


def call_serpapi(query: str, api_key: str, start: int, num_retries: int = 3) -> dict:
    params = {
        "q": query,
        "engine": "google",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": api_key,
        "start": start,
    }
    url = SERPAPI_ENDPOINT + "?" + urllib.parse.urlencode(params)
    last_exc = None
    for attempt in range(num_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 -- want to retry on any transient failure
            last_exc = e
            wait = 2 ** attempt
            print(f"  SerpAPI call failed ({e}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"SerpAPI call failed after {num_retries} attempts: {last_exc}")


def parse_organic_results(raw: dict, apply_leakage_filter: bool) -> list[dict]:
    """Mirrors ReliabilityRAG/data/google_search.py's parse_google_research_results,
    plus the leakage-domain filter the plan calls for."""
    results = []
    for item in raw.get("organic_results", []):
        url = item.get("link", "")
        if apply_leakage_filter and url and is_leakage_domain(url):
            continue
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("snippet", ""),
        })
    return results


def fetch_search(
    query: str,
    api_key: str,
    cache_dir: Path,
    pages: int,
    apply_leakage_filter: bool,
) -> tuple[list[dict], int]:
    """Returns (results, api_calls_made). Caches per (query, start) page."""
    all_results = []
    calls_made = 0
    for page_idx in range(pages):
        start = page_idx * 10
        cache_key = f"{query_hash(query)}_start{start}"
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = call_serpapi(query, api_key, start)
            cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            calls_made += 1
            time.sleep(1.0)  # be polite to the API / rate limit
        page_results = parse_organic_results(raw, apply_leakage_filter)
        all_results.extend(page_results)
        if not raw.get("organic_results"):
            break  # no more pages available
    return all_results, calls_made


def expected_searches(entry: dict) -> list[tuple]:
    """(search_type, hop_idx, query_text) tuples for one queries.json entry --
    same search plan as the chunked-agent collector's agent_prompt_template.txt."""
    out = [("full_question", None, entry["question"])]
    for hop in entry["hops"]:
        out.append(("hop", hop["hop_idx"], hop["query"]))
        out.append(("hop_paraphrase", hop["hop_idx"], hop["paraphrase"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="queries.json")
    ap.add_argument("--out", required=True, type=Path, help="output searches.json path")
    ap.add_argument("--cache-dir", type=Path, default=Path("serp_raw"),
                     help="raw-response cache dir (never commit this; see .gitignore)")
    ap.add_argument("--pages", type=int, default=1,
                     help="result pages per query, 10 results each (default 1 = 10 results, "
                          "matching the 10-results-per-search scope)")
    ap.add_argument("--no-leakage-filter", action="store_true",
                     help="disable dropping dataset-hosting/Q&A-dump domains (debugging only)")
    ap.add_argument("--max-calls", type=int, default=None,
                     help="stop after this many *new* SerpAPI calls (budget guard)")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the query plan and estimated call count, make no calls")
    args = ap.parse_args()

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("Set SERPAPI_API_KEY in the environment (never hard-code it -- the "
                 "original ReliabilityRAG/data/google_search.py had a live key committed "
                 "to git; rotate that key before using this script).")

    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    plan = []  # (qid, search_type, hop_idx, query_text)
    for entry in queries:
        for search_type, hop_idx, query_text in expected_searches(entry):
            plan.append((entry["qid"], search_type, hop_idx, query_text))

    est_calls = len(plan) * args.pages
    print(f"{len(queries)} questions -> {len(plan)} searches -> up to {est_calls} "
          f"SerpAPI calls at {args.pages} page(s)/search (cache may reduce this on rerun)")

    if args.dry_run:
        for qid, search_type, hop_idx, query_text in plan[:20]:
            print(f"  [{qid}] {search_type} hop_idx={hop_idx}: {query_text!r}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing output, skip what's already there.
    done = {}
    if args.out.exists():
        for obj in json.loads(args.out.read_text(encoding="utf-8")):
            done[(obj["qid"], obj["search_type"], obj["hop_idx"])] = obj

    total_calls = 0
    for i, (qid, search_type, hop_idx, query_text) in enumerate(plan):
        key = (qid, search_type, hop_idx)
        if key in done:
            continue
        if args.max_calls is not None and total_calls >= args.max_calls:
            print(f"hit --max-calls {args.max_calls}, stopping early "
                  f"({len(plan) - i} searches left undone)")
            break

        try:
            results, calls_made = fetch_search(
                query_text, api_key, args.cache_dir, args.pages,
                apply_leakage_filter=not args.no_leakage_filter,
            )
            obj = {
                "qid": qid, "search_type": search_type, "hop_idx": hop_idx,
                "query_text": query_text, "results": results[:10 * args.pages],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:  # noqa: BLE001
            obj = {
                "qid": qid, "search_type": search_type, "hop_idx": hop_idx,
                "query_text": query_text, "results": [], "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            calls_made = 0
            print(f"  FAILED: [{qid}] {search_type} hop_idx={hop_idx}: {e}", file=sys.stderr)

        done[key] = obj
        total_calls += calls_made

        # Write after every search, not just at the end -- same resumability
        # principle as the chunked-agent collector.
        args.out.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False),
                             encoding="utf-8")

        n_results = len(obj["results"])
        print(f"[{i+1}/{len(plan)}] {qid} {search_type} hop_idx={hop_idx}: "
              f"{n_results} results ({calls_made} new API call(s))")

    print(f"\ndone. {len(done)}/{len(plan)} searches recorded, "
          f"{total_calls} new SerpAPI calls made this run.")
    missing = [p for p in plan if (p[0], p[1], p[2]) not in done]
    if missing:
        print(f"{len(missing)} searches still missing (rerun this script to resume):")
        for qid, search_type, hop_idx, _ in missing[:10]:
            print(f"  [{qid}] {search_type} hop_idx={hop_idx}")


if __name__ == "__main__":
    main()
