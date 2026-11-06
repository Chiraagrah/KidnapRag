#!/usr/bin/env python
"""Per-hop INJECTABILITY profile for the weakest-link study (MuSiQue + e5).

For every hop of every question, run the e5 retriever on the hop's (placeholder-
resolved) sub-query and measure how hard it is to inject a poison doc there:

  gold_rank          rank of the hop's gold passage (0 = top; -1 = not in --depth)
  gold_score         e5 dot-product score of the gold passage
  top_score          score of the rank-0 doc
  best_nongold_score best score among non-gold docs
  margin             gold_score - best_nongold_score
                       > 0  gold leads  -> poison must exceed gold_score to win
                       <=0  gold already loses -> trivially injectable

Low margin / low gold_score = EASY to inject = weak link. No LLM, no attack; CPU is fine.

Reads EVERYTHING from the official MuSiQue dev file (--dev):
  - the per-hop sub-query, from question_decomposition (resolving "#N" bridge refs
    to the answer of step N, and stripping ">>" markup);
  - the gold passage TEXT, from paragraphs[paragraph_support_idx].
The gold is then matched against retrieved docs BY TEXT, so this is robust to
whatever _id scheme the indexed corpus uses (mq_N, question-scoped, etc.).

Lives in repo-root scripts/; locates ReAct/src relative to --repo.

Run (CPU node is enough):
  python scripts/injectability_profile.py \
      --corpus_path ReAct/datasets/musique/corpus.jsonl \
      --index_dir   ReAct/datasets/musique/e5_index \
      --dev         ReAct/datasets/musique_data_v1.0/data/musique_ans_v1.0_dev.jsonl \
      --qid_to_idx  ReAct/results/adv_targeted_results/musique_qid_to_idx.json \
      --device cpu --out ReAct/results/mis_diagnostic/musique_injectability.json
"""
import argparse, json, os, re, sys, statistics as st, collections


def norm_text(s):
    return " ".join((s or "").split())


def resolve_query(text, answers_by_id):
    """Resolve MuSiQue '#N' bridge refs to step N's answer; strip '>>' markup."""
    def sub(m):
        n = int(m.group(1))
        return answers_by_id.get(n, m.group(0))
    text = re.sub(r"#(\d+)", sub, text or "")
    return " ".join(text.replace(">>", " ").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--dev", required=True, help="official musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--qid_to_idx", default=None, help="restrict to these qids (the subset run)")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "ReAct", "src"))
    from e5_retriever import E5_Retriever  # noqa: E402

    subset = None
    if args.qid_to_idx:
        subset = set(json.load(open(args.qid_to_idx, encoding="utf-8")).keys())

    # load dev questions we need
    questions = {}
    with open(args.dev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if subset is None or q["id"] in subset:
                questions[q["id"]] = q
    print(f"profiling {len(questions)} questions"
          + (f" (of {len(subset)} in subset)" if subset else ""))

    kw = {} if args.device is None else {"device": args.device}
    retriever = E5_Retriever(index_dir=args.index_dir, corpus_path=args.corpus_path, **kw)

    records = []
    for qi, (qid, q) in enumerate(questions.items(), 1):
        decomp = q.get("question_decomposition", [])
        # MuSiQue '#N' refers to the N-th decomposition POSITION (1-indexed),
        # NOT the step's source 'id' field -- key by position.
        answers_by_id = {i + 1: step.get("answer", "")
                         for i, step in enumerate(decomp)}
        paras = q.get("paragraphs", [])
        for hop, step in enumerate(decomp):
            query = resolve_query(step.get("question", ""), answers_by_id)
            gidx = step.get("paragraph_support_idx")
            gold_text = norm_text(paras[gidx]["paragraph_text"]) if (
                gidx is not None and 0 <= gidx < len(paras)) else None

            res = retriever.search(query, k=args.depth)
            texts = [norm_text(r.get("contents", "")) for r in res]
            scores = [r["score"] for r in res]
            gold_rank = texts.index(gold_text) if (gold_text and gold_text in texts) else -1
            found = gold_rank >= 0
            gold_score = scores[gold_rank] if found else None
            top_score = scores[0] if scores else None
            nongold = [sc for i, sc in enumerate(scores) if i != gold_rank]
            best_nongold = max(nongold) if nongold else None
            margin = (gold_score - best_nongold) if (found and best_nongold is not None) else None
            records.append({
                "qid": qid, "hop": hop, "n_hops": len(decomp), "query": query,
                "gold_title": paras[gidx]["title"] if (gidx is not None and 0 <= gidx < len(paras)) else None,
                "gold_rank": gold_rank,
                "gold_in_topk": bool(found and gold_rank < args.k),
                "gold_score": round(gold_score, 4) if found else None,
                "top_score": round(top_score, 4) if top_score is not None else None,
                "best_nongold_score": round(best_nongold, 4) if best_nongold is not None else None,
                "margin": round(margin, 4) if margin is not None else None,
            })
        if qi % 20 == 0:
            print(f"  {qi}/{len(questions)} questions")

    # ---------------- aggregate ----------------
    n = len(records)
    found = [r for r in records if r["gold_rank"] >= 0]
    margins = [r["margin"] for r in found if r["margin"] is not None]
    rank1 = sum(1 for r in found if r["gold_rank"] == 0)
    intopk = sum(1 for r in records if r["gold_in_topk"])
    neg = sum(1 for m in margins if m <= 0)

    def hop_weakness(r):
        if r["gold_rank"] < 0:
            return -2.0
        return r["margin"] if r["margin"] is not None else 1e9
    byq = collections.defaultdict(list)
    for r in records:
        byq[r["qid"]].append(r)
    weakest_hop_idx = collections.Counter(min(rs, key=hop_weakness)["hop"] for rs in byq.values())

    summary = {
        "n_hops": n, "n_questions": len(byq),
        "gold_found_rate": round(len(found) / n, 4) if n else 0,
        "gold_in_topk_rate": round(intopk / n, 4) if n else 0,
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
