#!/usr/bin/env python3
"""Per-hop Keenable collection (ReliabilityRAG method, applied per hop).

Replaces 02_serp_collect.py (SerpAPI) per the 2026-11-20 revision of
claude/multihop-web-dataset-plan.md (section 2.2a: "Why Keenable instead of
SerpAPI"). Keenable (api.keenable.ai) is an independent web search/fetch API
built for agent use:

  - /v1/search takes max_results directly (1-50, default 10) in ONE call --
    no pagination parameter, unlike SerpAPI post-Sept-2025 (num capped at
    ~10/call, requiring one paid call per page).
  - exclude_domains (up to 1000 hosts) drops leakage domains server-side,
    so the client-side leakage filter below is only a backstop for domains
    not on that list (e.g. question-overlap leakage from unlisted hosts).
  - snippet_max_length (180-10000) lets us request long, complete snippets
    directly instead of a fixed short truncation.
  - Works keyless by default via POST /v1/search/public (rate-limited per
    IP -- keyless traffic shares one pool of 1000 requests/hour and 10/sec
    per IP across search+fetch combined, confirmed against
    docs.keenable.ai/authentication; no cost). KEENABLE_API_KEY, when set,
    switches to POST /v1/search with X-API-Key and raises the rate limit
    (billed as search.pro/search.realtime SKUs). Never hard-code the key --
    read it from the environment only.

Reads queries.json (see README.md / make_sample_input.py for the schema:
one entry per qid with "question" and a "hops" list of {hop_idx, query,
paraphrase}). For each entry, issues one Keenable /v1/search call per: the
full question, each hop's query, and each hop's paraphrase -- same search
plan 02_serp_collect.py used, so output is drop-in comparable.

Caches every raw response to keenable_raw/<sha1(query+params)>.json so the
job is resumable and never pays twice for the same query.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

KEENABLE_SEARCH_ENDPOINT = "https://api.keenable.ai/v1/search"
KEENABLE_SEARCH_PUBLIC_ENDPOINT = "https://api.keenable.ai/v1/search/public"


def load_dotenv(path: Path = Path(".env")) -> None:
    """Tiny .env loader (no python-dotenv dependency, so this works on Wulver's
    bare Python too). Only sets a variable if it isn't already in os.environ,
    so a real shell export always wins over the file. Looks in the current
    directory and in this script's own directory, in that order."""
    candidates = [path, Path(__file__).resolve().parent / path.name]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break  # first match wins

# Leakage filter: domains likely to host the dataset itself or Q&A dumps
# that would hand the agent the final answer outright (plan doc section
# 2.2c "Leakage filter"). exclude_domains in the request does most of this
# work server-side; this client-side check is a backstop (and is also
# applied when --no-server-exclude is set, e.g. to measure its effect).
LEAKAGE_DOMAINS = {
    "huggingface.co", "github.com", "kaggle.com", "paperswithcode.com",
    "arxiv.org", "aclanthology.org", "semanticscholar.org",
    "researchgate.net", "quizlet.com",
}


def query_hash(q: str, params: dict) -> str:
    key = q + "|" + json.dumps(params, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def is_leakage_domain(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    host = host.removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in LEAKAGE_DOMAINS)


def call_keenable_search(
    query: str,
    api_key: str | None,
    max_results: int,
    snippet_max_length: int,
    mode: str,
    exclude_domains: list[str],
    app_title: str,
    query_time: str | None,
    num_retries: int = 5,
) -> dict:
    """POST /v1/search (or the keyless /public variant). Retries on 429/5xx,
    honoring Retry-After when present -- the API reference publishes no
    numeric rate limit, so backoff-on-429 is the only available signal."""
    body = {
        "query": query,
        "mode": mode,
        "max_results": max_results,
        "snippet_max_length": snippet_max_length,
    }
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    if query_time:
        body["query_time"] = query_time

    endpoint = KEENABLE_SEARCH_ENDPOINT if api_key else KEENABLE_SEARCH_PUBLIC_ENDPOINT
    headers = {"Content-Type": "application/json", "X-Keenable-Title": app_title}
    if api_key:
        headers["X-API-Key"] = api_key

    data = json.dumps(body).encode("utf-8")
    last_exc = None
    for attempt in range(num_retries):
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After") if e.headers else None
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                err_msg = err_body.get("message", str(e))
            except Exception:
                err_msg = str(e)
            if e.code == 402:
                # Out of credits -- the docs say don't retry this one.
                raise RuntimeError(f"Keenable 402 (out of credits): {err_msg}") from e
            if e.code == 400:
                # Bad request -- a parameter problem, not transient. Don't retry.
                raise RuntimeError(f"Keenable 400 (invalid request): {err_msg}") from e
            last_exc = RuntimeError(f"Keenable HTTP {e.code}: {err_msg}")
            wait = float(retry_after) if retry_after else (2 ** attempt)
            print(f"  Keenable call failed ({last_exc}); retrying in {wait}s...",
                  file=sys.stderr)
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001 -- retry on any other transient failure
            last_exc = e
            wait = 2 ** attempt
            print(f"  Keenable call failed ({e}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Keenable call failed after {num_retries} attempts: {last_exc}")


def parse_search_results(raw: dict, apply_leakage_filter: bool) -> list[dict]:
    """Maps Keenable's /v1/search response (results[].{title,url,snippet,
    published_at,acquired_at}) into the same {title,url,snippet} shape
    02_serp_collect.py produced, plus the extra date fields Keenable gives
    us for free (plan doc 2.2b)."""
    results = []
    for item in raw.get("results", []):
        url = item.get("url", "")
        if apply_leakage_filter and url and is_leakage_domain(url):
            continue
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("snippet", ""),
            "published_at": item.get("published_at"),
            "acquired_at": item.get("acquired_at"),
        })
    return results


def fetch_search(
    query: str,
    api_key: str | None,
    cache_dir: Path,
    max_results: int,
    snippet_max_length: int,
    mode: str,
    exclude_domains: list[str],
    app_title: str,
    query_time: str | None,
    apply_leakage_filter: bool,
) -> tuple[list[dict], int]:
    """Returns (results, api_calls_made). One call gets up to max_results
    results directly -- no paging loop needed (plan doc 2.2a)."""
    params = {
        "max_results": max_results,
        "snippet_max_length": snippet_max_length,
        "mode": mode,
        "exclude_domains": sorted(exclude_domains),
        "query_time": query_time,
    }
    cache_path = cache_dir / f"{query_hash(query, params)}.json"
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        calls_made = 0
    else:
        raw = call_keenable_search(
            query, api_key, max_results, snippet_max_length, mode,
            exclude_domains, app_title, query_time,
        )
        cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        calls_made = 1
        time.sleep(0.5)  # be polite / stay under the per-IP rate limit
    results = parse_search_results(raw, apply_leakage_filter)
    return results, calls_made


def expected_searches(entry: dict) -> list[tuple]:
    """(search_type, hop_idx, query_text) tuples for one queries.json entry --
    same search plan 02_serp_collect.py used."""
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
    ap.add_argument("--cache-dir", type=Path, default=Path("keenable_raw"),
                     help="raw-response cache dir (never commit this; see .gitignore)")
    ap.add_argument("--max-results", type=int, default=50,
                     help="results per search, 1-50 (default 50 -- one Keenable call "
                          "replaces what used to be multiple SerpAPI pages)")
    ap.add_argument("--snippet-max-length", type=int, default=2000,
                     help="cap on snippet length in chars, 180-10000 (default 2000; "
                          "plan doc 2.4)")
    ap.add_argument("--mode", choices=["pro", "realtime"], default="pro",
                     help="pro = deeper retrieval/ranking (default); realtime = faster, "
                          "for the coverage round (plan doc 2.6)")
    ap.add_argument("--query-time", default=None,
                     help="ISO 8601 timestamp: search the index as it stood at that time "
                          "(plan doc 2.2, temporal-drift control). Omit to search the "
                          "current index.")
    ap.add_argument("--app-title", default="KidnapRAG-dataset-builder",
                     help="sent as X-Keenable-Title; required by the keyless /public "
                          "endpoint (400 'Missing app identifier' if absent)")
    ap.add_argument("--no-server-exclude", action="store_true",
                     help="don't pass exclude_domains to Keenable (debugging / to compare "
                          "server-side vs client-side filtering only)")
    ap.add_argument("--no-leakage-filter", action="store_true",
                     help="disable the client-side leakage backstop entirely (debugging only)")
    ap.add_argument("--max-calls", type=int, default=None,
                     help="stop after this many *new* Keenable calls (budget guard)")
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                     help="abort the run after this many searches fail in a row "
                          "(default 3), instead of grinding through the whole plan")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the query plan and estimated call count, make no calls")
    args = ap.parse_args()

    load_dotenv()  # picks up .env next to this script or in cwd, if present
    api_key = os.environ.get("KEENABLE_API_KEY")
    if not api_key:
        print("KEENABLE_API_KEY not set -- using Keenable's keyless public endpoint "
              "(rate-limited per IP, no cost). Set KEENABLE_API_KEY to raise the rate "
              "limit for the full run once the pilot confirms keyless is binding "
              "(plan doc stage C).", file=sys.stderr)

    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    plan = []  # (qid, search_type, hop_idx, query_text)
    for entry in queries:
        for search_type, hop_idx, query_text in expected_searches(entry):
            plan.append((entry["qid"], search_type, hop_idx, query_text))

    print(f"{len(queries)} questions -> {len(plan)} searches -> up to {len(plan)} "
          f"Keenable calls (1 call/search, max_results={args.max_results}; "
          f"cache may reduce this on rerun)")

    if args.dry_run:
        for qid, search_type, hop_idx, query_text in plan[:20]:
            print(f"  [{qid}] {search_type} hop_idx={hop_idx}: {query_text!r}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    exclude_domains = [] if args.no_server_exclude else sorted(LEAKAGE_DOMAINS)

    # Resume support: load existing output, skip what's already there.
    # Entries that failed (have an "error" key) are NOT treated as done, so a
    # rerun retries them instead of silently keeping an empty result.
    done = {}
    if args.out.exists():
        for obj in json.loads(args.out.read_text(encoding="utf-8")):
            done[(obj["qid"], obj["search_type"], obj["hop_idx"])] = obj

    total_calls = 0
    consecutive_failures = 0
    for i, (qid, search_type, hop_idx, query_text) in enumerate(plan):
        key = (qid, search_type, hop_idx)
        if key in done and "error" not in done[key]:
            continue
        if args.max_calls is not None and total_calls >= args.max_calls:
            print(f"hit --max-calls {args.max_calls}, stopping early "
                  f"({len(plan) - i} searches left undone)")
            break
        if consecutive_failures >= args.max_consecutive_failures:
            print(f"aborting: {consecutive_failures} searches failed in a row "
                  f"(network, key or rate-limit problem -- see errors above). "
                  f"Fix it and rerun; failed searches will be retried.", file=sys.stderr)
            break

        try:
            results, calls_made = fetch_search(
                query_text, api_key, args.cache_dir,
                args.max_results, args.snippet_max_length, args.mode,
                exclude_domains, args.app_title, args.query_time,
                apply_leakage_filter=not args.no_leakage_filter,
            )
            obj = {
                "qid": qid, "search_type": search_type, "hop_idx": hop_idx,
                "query_text": query_text, "results": results,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except Exception as e:  # noqa: BLE001
            obj = {
                "qid": qid, "search_type": search_type, "hop_idx": hop_idx,
                "query_text": query_text, "results": [], "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            calls_made = 1  # a failed attempt still counts toward --max-calls
            consecutive_failures += 1
            print(f"  FAILED: [{qid}] {search_type} hop_idx={hop_idx}: {e}", file=sys.stderr)
        else:
            consecutive_failures = 0

        done[key] = obj
        total_calls += calls_made

        # Write after every search, not just at the end -- same resumability
        # principle as 02_serp_collect.py.
        args.out.write_text(json.dumps(list(done.values()), indent=2, ensure_ascii=False),
                             encoding="utf-8")

        n_results = len(obj["results"])
        print(f"[{i+1}/{len(plan)}] {qid} {search_type} hop_idx={hop_idx}: "
              f"{n_results} results ({calls_made} new API call(s))")

    print(f"\ndone. {len(done)}/{len(plan)} searches recorded, "
          f"{total_calls} new Keenable calls made this run.")
    missing = [p for p in plan
               if (p[0], p[1], p[2]) not in done or "error" in done[(p[0], p[1], p[2])]]
    if missing:
        print(f"{len(missing)} searches still missing (rerun this script to resume):")
        for qid, search_type, hop_idx, _ in missing[:10]:
            print(f"  [{qid}] {search_type} hop_idx={hop_idx}")


if __name__ == "__main__":
    main()
