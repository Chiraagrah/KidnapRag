"""Per-hop MIS / answer-agreement diagnostic for WebThinker runs (read-only).

For every search hop (step_stats[*] with is_search) of a finished WebThinker
results file, take the top-k retrieved chunks and run ReliabilityRAG Algorithm 1
(ReAct/src/reliability_mis.py): isolated answering -> pairwise DeBERTa NLI ->
rank-aware MIS. Isolated answering uses the gpt-oss server (chat completions).

Beyond MIS it records whether the retrieved docs are gold, whether each isolated
answer matches the gold answer, and how many non-"unknown" answers agree. This is
the number the dataset choice hinges on (see docs/DATASET_MIGRATION_PLAN.md, 1 and 3):
  - n_match_gold   : docs whose isolated answer contains the gold answer
  - n_agree        : largest NLI-consistent set among docs that gave a real answer
  - mis_size / mis_size_nli_only (unknown = contradiction / neutral)

Needs the run to have been produced with chunk ids in search_documents (the
`id` key added to extract_relevant_info_e5); older results lack it and the
gold-doc columns are then reported as unavailable.

Usage:
  python Webthinker/scripts/mis_agreement.py \
      --results Webthinker/results/browsecomp_plus/clean_gptoss.json \
      --api_base http://$(cat ~/vllm_local_server.txt)/v1 \
      --model_name openai/gpt-oss-120b --nli_device cuda:0 \
      --output Webthinker/results/browsecomp_plus/mis_agreement_clean.json
"""
import argparse
import json
import os
import re
import string
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

KIDNAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIDNAP_ROOT / "ReAct" / "src"))
from reliability_mis import (ReliabilityMIS, DEFAULT_NLI_MODEL,  # noqa: E402
                             rank_aware_mis)


def norm(s):
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def answer_matches(ans, gold):
    a, g = norm(ans), norm(gold)
    if not a or not g or len(g) < 2:
        return False
    return g in a or (len(a) >= 3 and a in g)


def docid_of(chunk_id):
    return str(chunk_id).split("#")[0] if chunk_id not in (None, "") else None


def make_generate_fn(args):
    from openai import OpenAI
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    def gen(prompt):
        for _ in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=args.max_tokens,
                    extra_body={"reasoning_effort": args.reasoning_effort},
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                err = e
        print(f"[warn] generation failed: {err}", file=sys.stderr)
        return ""
    return gen


def collect_hops(data, sources, max_questions):
    hops = []
    for qi, item in enumerate(data[:max_questions] if max_questions > 0 else data):
        k = 0
        for st in item.get("step_stats", []):
            if not st.get("is_search") or not st.get("search_documents"):
                continue
            if sources != "all" and st.get("source", "main") != sources:
                continue
            k += 1
            hops.append((qi, k, item, st))
    return hops


def analyze_hop(mis, gen, args, qi, k, item, st):
    query = item["question"] if args.query_mode == "question" else st.get("search_query")
    docs = [{"title": d.get("title", ""), "contents": d.get("page_info", ""),
             "score": d.get("score", 0.0), "is_poisoned": d.get("is_poisoned", False)}
            for d in st["search_documents"]]
    res = mis.analyze(gen, query, docs)
    gold_ids = set(map(str, item.get("gold_docids", [])))
    evid_ids = set(map(str, item.get("evidence_docids", [])))
    gold_ans = item.get("answer", "")
    have_ids = any(d.get("id") for d in st["search_documents"])

    nodes, n_match, n_gold_docs = res["nodes"], 0, 0
    matched_flags = []
    for node, d in zip(nodes, st["search_documents"]):
        did = docid_of(d.get("id"))
        node["chunk_id"] = d.get("id")
        node["docid"] = did
        node["is_gold_doc"] = (did in gold_ids) if have_ids else None
        node["is_evidence_doc"] = (did in evid_ids) if have_ids else None
        m = (not node["unknown"]) and answer_matches(node["isolated_answer"], gold_ans)
        node["matches_gold_answer"] = m
        matched_flags.append(m)
        n_match += int(m)
        n_gold_docs += int(bool(node["is_gold_doc"]))

    # largest NLI-consistent set among docs that gave a real (non-unknown) answer
    n = len(nodes)
    real = [i for i, nd in enumerate(nodes) if not nd["unknown"]]
    cm = res["contra_matrix"]
    sub_edges = {(a, b) for a, b in combinations(range(len(real)), 2)
                 if cm[real[a]][real[b]] >= mis.beta}
    n_agree = len(rank_aware_mis(len(real), sub_edges)) if real else 0

    m = res["metrics"]
    return {
        "qid": item.get("qid", item.get("id")), "q_index": qi, "hop": k,
        "step": st.get("step"), "source": st.get("source", "main"),
        "query": query, "gold_answer": gold_ans, "n_docs": n,
        "n_unknown": res["n_unknown"], "n_real_answers": len(real),
        "n_gold_docs_in_topk": n_gold_docs if have_ids else None,
        "n_match_gold": n_match, "n_agree": n_agree,
        "mis_size": m["mis_size"], "weakness": m["weakness"],
        "mis_size_nli_only": m["mis_size_nli_only"],
        "weakness_nli_only": m["weakness_nli_only"],
        "edge_density": m["edge_density"],
        "mis_contains_gold_match": any(matched_flags[i] for i in res["mis_selected"]),
        "graph": res,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


FIELDS = ["n_unknown", "n_real_answers", "n_gold_docs_in_topk", "n_match_gold", "n_agree",
          "mis_size", "weakness", "mis_size_nli_only", "weakness_nli_only", "edge_density"]


def summarize(rows, k_target):
    def agg(rs):
        out = {"hops": len(rs)}
        for f in FIELDS:
            out["mean_" + f] = mean([r[f] for r in rs])
        out["frac_hops_ge3_match_gold"] = mean([float(r["n_match_gold"] >= 3) for r in rs])
        out["frac_hops_ge1_match_gold"] = mean([float(r["n_match_gold"] >= 1) for r in rs])
        out["frac_hops_ge3_agree"] = mean([float(r["n_agree"] >= 3) for r in rs])
        out["frac_hops_mis_has_gold_match"] = mean([float(r["mis_contains_gold_match"]) for r in rs])
        return out

    by_hop = defaultdict(list)
    for r in rows:
        by_hop[r["hop"]].append(r)
    weakest = max(by_hop, key=lambda h: mean([r["weakness"] for r in by_hop[h]])) if by_hop else None
    return {
        "overall": agg(rows),
        "per_hop_index": {str(h): agg(rs) for h, rs in sorted(by_hop.items())},
        "weakest_hop_index_overall": weakest,
        "k": k_target,
        "go_criterion": "mean_n_match_gold >= 3 (or frac_hops_ge3_match_gold high) and mean_mis_size clearly above the HotpotQA baseline of 1.28/5",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model_name", default="openai/gpt-oss-120b")
    ap.add_argument("--reasoning_effort", default="low")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    ap.add_argument("--nli_device", default="cuda:0")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--unknown_neutral", action="store_true")
    ap.add_argument("--query_mode", choices=["sub", "question"], default="sub",
                    help="sub: the agent's search query for the hop; question: the original question")
    ap.add_argument("--sources", choices=["main", "all"], default="main",
                    help="main: only the agent's own searches; all: also web-explorer sub-searches")
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    blob = json.load(open(args.results, encoding="utf-8"))
    data = blob["data"] if isinstance(blob, dict) and "data" in blob else blob
    hops = collect_hops(data, args.sources, args.max_questions)
    print(f"{len(hops)} hops over {len(data)} questions")

    mis = ReliabilityMIS(args.nli_model, args.nli_device, args.beta,
                         doc_char_limit=3000, unknown_contradicts=not args.unknown_neutral)
    nli_lock = threading.Lock()
    raw = mis._contra_prob
    mis._contra_prob = lambda p, h: (nli_lock.acquire(), raw(p, h), nli_lock.release())[1]
    gen = make_generate_fn(args)

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(analyze_hop, mis, gen, args, *h) for h in hops]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(hops)} hops")
    rows.sort(key=lambda r: (r["q_index"], r["hop"], r["step"] or 0))

    summary = summarize(rows, len(rows[0]["graph"]["nodes"]) if rows else 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump({"args": vars(args), "summary": summary, "per_hop": rows},
              open(args.output, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    o = summary["overall"]
    print(json.dumps(o, indent=2))
    print("weakest hop index:", summary["weakest_hop_index_overall"])
    print("saved", args.output)


if __name__ == "__main__":
    main()
