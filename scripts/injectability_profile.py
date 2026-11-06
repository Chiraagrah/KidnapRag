#!/usr/bin/env python
"""Per-hop INJECTABILITY profile for the weakest-link study (MuSiQue + e5).

For every hop of every question, run the e5 retriever on the hop's (placeholder-
resolved) sub-query and measure how hard it is to inject a poison doc there:

  gold_rank          rank of the hop's gold doc (0 = top; -1 = not in --depth)
  gold_score         e5 dot-product score of the gold doc
  top_score          score of the rank-0 doc
  best_nongold_score best score among non-gold docs
  margin             gold_score - best_nongold_score
                       > 0  gold leads  -> poison must exceed gold_score to win
                       <=0  gold already loses -> trivially injectable

Low margin / low gold_score = EASY to inject = weak link (injectability half of
weakness = leverage x 1/margin). No LLM, no attack; runs fine on CPU.

The MuSiQue decomposition queries are compact "entity >> relation" with "#N"
referring to the answer of hop N (1-indexed); both are normalized/resolved here.

Lives in repo-root scripts/; locates ReAct/src relative to --repo.

Run (CPU node is enough):
  python scripts/injectability_profile.py \
      --corpus_path ReAct/datasets/musique/corpus.jsonl \
      --index_dir   ReAct/datasets/musique/e5_index \
      --support_map ReAct/datasets/musique/support_map.json \
      --qid_to_idx  ReAct/results/adv_targeted_results/musique_qid_to_idx.json \
      --out ReAct/results/mis_diagnostic/musique_injectability.json
"""
import argparse, json, os, re, sys, statistics as st, collections


def resolve_query(text, sub_answers):
    """Resolve MuSiQue '#N' bridge refs (1-indexed) and strip '>>' markup."""
    def sub(m):
        n = int(m.group(1))
        return sub_answers[n - 1] if 1 <= n <= len(sub_answers) else m.group(0)
    text = re.sub(r"#(\d+)", sub, text or "")
    text = text.replace(">>", " ")
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repo root (default: parent of scripts/)")
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--support_map", required=True)
    ap.add_argument("--qid_to_idx", default=None,
                    help="restrict to these qids (the subset actually run); default = all")
    ap.add_argument("--depth", type=int, default=100, help="retrieval depth to locate gold")
    ap.add_argument("--k", type=int, default=5, help="agent's top-k cutoff")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "ReAct", "src"))
    from e5_retriever import E5_Retriever  # noqa: E402

    kw = {} if args.device is None else {"device": args.device}
    retriever = E5_Retriever(index_dir=args.index_dir, corpus_path=args.corpus_path, **kw)

    support = json.load(open(args.support_map, encoding="utf-8"))
    if args.qid_to_idx:
        subset = set(json.load(open(args.qid_to_idx, encoding="utf-8")).keys())
        support = {q: s for q, s in support.items() if q in subset}
    print(f"profiling {len(support)} questions")

    records = []
    for qi, (qid, steps) in enumerate(support.items(), 1):
        sub_answers = [s.get("sub_answer", "") for s in steps]
        for s in steps:
            q = resolve_query(s.get("sub_question", ""), sub_answers)
            gold_id = s.get("gold_doc_id")
            res = retriever.search(q, k=args.depth)
            ids = [r["id"] for r in res]
            scores = [r["score"] for r in res]
            gold_rank = ids.index(gold_id) if gold_id in ids else -1
            found = gold_rank >= 0
            gold_score = scores[gold_rank] if found else None
            top_score = scores[0] if scores else None
            nongold = [sc for i, sc in enumerate(scores) if i != gold_rank]
            best_nongold = max(nongold) if nongold else None
            margin = (gold_score - best_nongold) if (found and best_nongold is not None) else None
            records.append({
                "qid": qid, "hop": s["hop"], "n_hops": len(steps),
                "query": q, "gold_doc_id": gold_id,
                "gold_rank": gold_rank,
                "gold_in_topk": bool(found and gold_rank < args.k),
                "gold_score": round(gold_score, 4) if found else None,
                "top_score": round(top_score, 4) if top_score is not None else None,
                "best_nongold_score": round(best_nongold, 4) if best_nongold is not None else None,
                "margin": round(margin, 4) if margin is not None else None,
            })
        if qi % 20 == 0:
            print(f"  {qi}/{len(support)} questions")

    # ---------------- aggregate ----------------
    n = len(records)
    found = [r for r in records if r["gold_rank"] >= 0]
    margins = [r["margin"] for r in found if r["margin"] is not None]
    rank1 = sum(1 for r in found if r["gold_rank"] == 0)
    intopk = sum(1 for r in records if r["gold_in_topk"])
    neg = sum(1 for m in margins if m <= 0)

    # per-question weakest hop = smallest margin (gold already-lost counts as very weak)
    def hop_weakness(r):
        if r["gold_rank"] < 0:
            return (-2, )          # gold not retrieved at all -> weakest
        return (r["margin"] if r["margin"] is not None else 1e9, )
    byq = collections.defaultdict(list)
    for r in records:
        byq[r["qid"]].append(r)
    weakest_hop_idx = collections.Counter()
    for qid, rs in byq.items():
        w = min(rs, key=hop_weakness)
        weakest_hop_idx[w["hop"]] += 1

    summary = {
        "n_hops": n, "n_questions": len(byq),
        "gold_found_rate": round(len(found) / n, 4),
        "gold_in_topk_rate": round(intopk / n, 4),
        "gold_is_rank1_rate": round(rank1 / len(found), 4) if found else None,
        "margin_already_lost_rate": round(neg / len(margins), 4) if margins else None,
        "margin_stats": {
            "min": round(min(margins), 4), "median": round(st.median(margins), 4),
            "mean": round(st.mean(margins), 4), "max": round(max(margins), 4),
        } if margins else None,
        "weakest_hop_index_distribution": {str(k): v for k, v in sorted(weakest_hop_idx.items())},
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "records": records},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== INJECTABILITY SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
