#!/usr/bin/env python3
"""Chunked Claude-agent web search collector.

Splits a queries.json file (see README.md for the schema) into chunks,
spawns one headless `claude -p` subprocess per chunk to run the web searches
with Claude Code's own WebSearch tool, and validates/retries each chunk's
output against the set of searches it was expected to produce.

NOT TESTED against a live `claude` binary (the environment this was written
in has `claude` disabled). Smoke-test with make_sample_input.py before
running on the full dataset. See README.md "Known risks" before trusting
this unmodified.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPT_TEMPLATE = Path(__file__).parent / "agent_prompt_template.txt"


def load_queries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        if "qid" not in item or "question" not in item or "hops" not in item:
            raise ValueError(f"queries[{i}] missing qid/question/hops: {item!r}")
    return data


def chunk_list(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def expected_keys_for_entry(entry: dict) -> set[tuple]:
    """(qid, search_type, hop_idx) tuples a fully-processed entry must have."""
    keys = {(entry["qid"], "full_question", None)}
    for hop in entry["hops"]:
        hop_idx = hop["hop_idx"]
        keys.add((entry["qid"], "hop", hop_idx))
        keys.add((entry["qid"], "hop_paraphrase", hop_idx))
    return keys


def present_keys(output_path: Path) -> set[tuple]:
    if not output_path.exists():
        return set()
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  WARNING: {output_path} is not valid JSON yet; treating as empty", file=sys.stderr)
        return set()
    keys = set()
    for obj in data:
        keys.add((obj.get("qid"), obj.get("search_type"), obj.get("hop_idx")))
    return keys


def run_chunk(
    chunk: list[dict],
    chunk_idx: int,
    work_dir: Path,
    out_dir: Path,
    prompt_template: str,
    claude_bin: str,
    raw_output: bool,
    timeout_s: int,
    max_fetch_per_search: int,
    attempt_n: int = 0,
) -> bool:
    """Returns True if the chunk's output now covers every expected key."""
    chunk_input_path = work_dir / f"chunk_{chunk_idx:02d}_input.json"
    chunk_output_path = out_dir / f"chunk_{chunk_idx:02d}.json"
    chunk_input_path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")

    if not chunk_output_path.exists():
        chunk_output_path.write_text("[]", encoding="utf-8")

    expected = set()
    for entry in chunk:
        expected |= expected_keys_for_entry(entry)

    have = present_keys(chunk_output_path)
    if expected <= have:
        print(f"chunk {chunk_idx}: already complete ({len(have)} searches), skipping")
        return True

    prompt = prompt_template.format(
        CHUNK_INPUT_PATH=str(chunk_input_path.resolve()),
        CHUNK_OUTPUT_PATH=str(chunk_output_path.resolve()),
        MAX_FETCH_PER_SEARCH=max_fetch_per_search,
    )

    cmd = [claude_bin, "-p", prompt, "--allowedTools", "WebSearch WebFetch Read Write"]
    if not raw_output:
        cmd += ["--output-format", "json"]

    print(f"chunk {chunk_idx}: launching claude -p ({len(chunk)} questions, "
          f"{len(expected)} searches expected, {len(have)} already present)")
    t0 = time.time()
    try:
        # On Windows, subprocess.run(text=True) decodes with the console's
        # legacy codepage (cp1252) by default, which chokes on non-ASCII
        # bytes in claude's output (emoji, smart quotes, non-English page
        # text from WebFetch) and silently turns stdout/stderr into None.
        # Force UTF-8 and replace undecodable bytes instead of crashing.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        print(f"chunk {chunk_idx}: TIMED OUT after {timeout_s}s", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"chunk {chunk_idx}: could not exec '{claude_bin}' -- is it on PATH?",
              file=sys.stderr)
        raise

    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"chunk {chunk_idx}: claude exited {proc.returncode} after {dt:.0f}s", file=sys.stderr)
        print(f"  stderr (tail): {(proc.stderr or '')[-2000:]}", file=sys.stderr)

    # Always dump the raw stdout for this attempt to a debug file -- the
    # terminal scrolls, this doesn't. Keep every attempt (don't overwrite),
    # since a retry's output is exactly what you need when debugging why the
    # previous attempt stalled.
    debug_dir = out_dir / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"chunk_{chunk_idx:02d}_attempt_{attempt_n}.json"
    debug_path.write_text(proc.stdout or "", encoding="utf-8")
    if proc.stderr and proc.stderr.strip():
        (debug_dir / f"chunk_{chunk_idx:02d}_attempt_{attempt_n}.stderr.txt").write_text(
            proc.stderr, encoding="utf-8"
        )

    # returncode 0 does NOT mean the chunk succeeded: claude -p can exit cleanly
    # while every tool call it tried was silently permission-denied. When
    # --output-format json is in play, parse the result and surface this
    # explicitly instead of only trusting returncode/stderr.
    if not raw_output and proc.stdout and proc.stdout.strip():
        try:
            result_obj = json.loads(proc.stdout)
        except json.JSONDecodeError:
            result_obj = None
            print(f"chunk {chunk_idx}: stdout was not valid JSON -- see {debug_path}", file=sys.stderr)
        if result_obj:
            denials = result_obj.get("permission_denials") or []
            if denials:
                tools = sorted({d.get("tool_name") for d in denials})
                print(f"chunk {chunk_idx}: {len(denials)} PERMISSION DENIALS for tool(s) "
                      f"{tools} -- check --allowedTools covers what the prompt needs",
                      file=sys.stderr)
            # Always show what the agent actually said at the end -- this is
            # the single most useful line for diagnosing a stall, whether or
            # not is_error is set (a "stopped early, here's why" message is
            # not flagged as an error by claude itself).
            result_text = str(result_obj.get("result") or "")
            print(f"chunk {chunk_idx}: result text: {result_text[:800]}"
                  + (" [...truncated, see debug file]" if len(result_text) > 800 else ""))
            print(f"chunk {chunk_idx}: full output saved to {debug_path}")
            # Running under a Claude Pro/Max login: total_cost_usd is only an
            # informational cost-equivalent estimate, not a real charge --
            # actual consumption is against the plan's usage-window limits.
            # Print token/search usage too so you can gauge how much of that
            # window one chunk costs and extrapolate before running them all.
            usage = result_obj.get("usage") or {}
            cost = result_obj.get("total_cost_usd")
            web_searches = (usage.get("server_tool_use") or {}).get("web_search_requests")
            print(f"chunk {chunk_idx}: cost-equivalent ${cost:.4f} "
                  f"(Pro/Max plan: usage-window consumption, not a real charge) | "
                  f"input={usage.get('input_tokens')} output={usage.get('output_tokens')} "
                  f"cache_read={usage.get('cache_read_input_tokens')} "
                  f"web_searches_reported={web_searches} num_turns={result_obj.get('num_turns')}")

    have = present_keys(chunk_output_path)
    missing = expected - have
    if missing:
        print(f"chunk {chunk_idx}: incomplete after {dt:.0f}s -- "
              f"{len(missing)}/{len(expected)} searches missing", file=sys.stderr)
        return False

    print(f"chunk {chunk_idx}: complete after {dt:.0f}s ({len(expected)} searches)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="queries.json")
    ap.add_argument("--out-dir", required=True, type=Path, help="where chunk_NN.json land")
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=2,
                     help="retries per chunk before giving up on it")
    ap.add_argument("--timeout", type=int, default=3600,
                     help="seconds to wait for one claude -p invocation "
                          "(raised default: now includes WebFetch calls, not just WebSearch)")
    ap.add_argument("--max-fetch-per-search", type=int, default=4,
                     help="how many of each search's top results to WebFetch for a real "
                          "excerpt (the rest keep title/url only, snippet empty)")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE)
    ap.add_argument("--raw-output", action="store_true",
                     help="omit --output-format json (use if your claude build rejects it)")
    ap.add_argument("--parallel", type=int, default=1,
                     help="chunks to run concurrently (default 1 = sequential)")
    ap.add_argument("--only-chunks", type=str, default=None,
                     help="comma-separated chunk indices to (re)run, e.g. '3,7'")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "_chunk_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.input)
    chunks = chunk_list(queries, args.chunk_size)
    print(f"{len(queries)} questions -> {len(chunks)} chunks of up to {args.chunk_size}")

    prompt_template = args.prompt_template.read_text(encoding="utf-8")

    indices = range(len(chunks))
    if args.only_chunks:
        indices = [int(x) for x in args.only_chunks.split(",")]

    if args.parallel > 1:
        import concurrent.futures as cf

        def attempt(i):
            for attempt_n in range(args.max_retries + 1):
                ok = run_chunk(
                    chunks[i], i, work_dir, args.out_dir, prompt_template,
                    args.claude_bin, args.raw_output, args.timeout,
                    args.max_fetch_per_search, attempt_n,
                )
                if ok:
                    return i, True
                print(f"chunk {i}: retry {attempt_n + 1}/{args.max_retries}")
            return i, False

        results = {}
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            for i, ok in ex.map(attempt, indices):
                results[i] = ok
    else:
        results = {}
        for i in indices:
            ok = False
            for attempt_n in range(args.max_retries + 1):
                ok = run_chunk(
                    chunks[i], i, work_dir, args.out_dir, prompt_template,
                    args.claude_bin, args.raw_output, args.timeout,
                    args.max_fetch_per_search, attempt_n,
                )
                if ok:
                    break
                print(f"chunk {i}: retry {attempt_n + 1}/{args.max_retries}")
            results[i] = ok

    failed = [i for i, ok in results.items() if not ok]
    print("\n--- summary ---")
    print(f"{len(results) - len(failed)}/{len(results)} chunks complete")
    if failed:
        print(f"FAILED chunks (rerun with --only-chunks {','.join(map(str, failed))}): {failed}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
