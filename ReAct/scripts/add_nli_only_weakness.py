#!/usr/bin/env python
"""Augment an existing MIS diagnostic run with an NLI-only weakness column.

The main run uses the "unknown = contradiction" rule, which on clean HotpotQA
saturates weakness near the k=5 ceiling (0.8). This recomputes, per loop, the
weakness you would get under the *original* ReliabilityRAG rule: contradiction
edges from the DeBERTa NLI probabilities only (contra_prob >= beta), with
"unknown" answers treated as neutral. All inputs come from the stored
`contra_matrix`, so no model re-run is needed.

Adds, alongside the existing fields (non-destructive):
  per_question[*].steps[*].mis.metrics.weakness_nli_only
                                        .mis_size_nli_only
                                        .excluded_fraction_nli_only
                                        .edge_density_nli_only
  summary.aggregation_A_per_loop_index[i].mean_weakness_nli_only  (+ mis/excl/density)
  summary.weakest_loop_index_overall_nli_only
  summary.aggregation_B_per_question_weakest_loop_distribution_nli_only

Usage (from ReAct/):
  python scripts/add_nli_only_weakness.py \
      --per_question results/mis_diagnostic/hotpotqa_qwen25_mis_per_question.json \
      --summary      results/mis_diagnostic/hotpotqa_qwen25_mis_summary.json \
      --beta 0.5
Originals are backed up to *.bak before being overwritten.
"""
import argparse, json, shutil, collections
from itertools import combinations


def mis_size(adj, n):
    """Largest independent set size by brute force (n is small, k=5)."""
    for r in range(n, 0, -1):
        for c in combinations(range(n), r):
            if all(adj[i][j] == 0 for i, j in combinations(c, 2)):
                return r
    return 1


def nli_only_metrics(contra_matrix, beta):
    n = len(contra_matrix)
    if n == 0:
        return None
    adj = [[1 if (i != j and contra_matrix[i][j] >= beta) else 0
            for j in range(n)] for i in range(n)]
    edge_count = sum(adj[i][j] for i, j in combinations(range(n), 2))
    max_edges = n * (n - 1) // 2
    ms = mis_size(adj, n)
    return {
        "mis_size_nli_only": ms,
        "excluded_fraction_nli_only": round(1 - ms / n, 4),
        "edge_density_nli_only": round(edge_count / max_edges, 4) if max_edges else 0.0,
        "weakness_nli_only": round(1 - ms / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_question", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    pq = json.load(open(args.per_question, encoding="utf-8"))

    # per-loop NLI-only metrics, grouped by loop index for aggregation A
    by_loop = collections.defaultdict(list)          # loop_idx -> [weakness_nli,...]
    by_loop_full = collections.defaultdict(list)      # loop_idx -> [metrics dict]
    # per-question weakest loop under NLI-only (lexicographic: excl desc, density desc, earliest step)
    weakest_dist_nli = collections.Counter()

    for q in pq:
        q_loops = []  # (step, metrics_nli)
        for stp in q.get("steps", []):
            if not stp.get("is_search"):
                continue
            m = stp.get("mis")
            if not m or "contra_matrix" not in m:
                continue
            met_nli = nli_only_metrics(m["contra_matrix"], args.beta)
            if met_nli is None:
                continue
            m.setdefault("metrics", {}).update(met_nli)
            step_idx = stp.get("step")
            by_loop[step_idx].append(met_nli["weakness_nli_only"])
            by_loop_full[step_idx].append(met_nli)
            q_loops.append((step_idx, met_nli))
        if q_loops:
            # weakest: highest excluded_fraction, then edge_density, then earliest step
            worst = max(q_loops, key=lambda t: (t[1]["excluded_fraction_nli_only"],
                                                t[1]["edge_density_nli_only"], -t[0]))
            weakest_dist_nli[worst[0]] += 1

    # aggregation A additions
    summary = json.load(open(args.summary, encoding="utf-8"))
    aggA = summary.get("aggregation_A_per_loop_index", {})
    best_idx, best_val = None, -1.0
    for idx_str, row in aggA.items():
        idx = int(idx_str)
        mets = by_loop_full.get(idx, [])
        if not mets:
            continue
        n = len(mets)
        mw = sum(x["weakness_nli_only"] for x in mets) / n
        row["mean_weakness_nli_only"] = round(mw, 4)
        row["mean_excluded_fraction_nli_only"] = round(
            sum(x["excluded_fraction_nli_only"] for x in mets) / n, 4)
        row["mean_edge_density_nli_only"] = round(
            sum(x["edge_density_nli_only"] for x in mets) / n, 4)
        row["mean_mis_size_nli_only"] = round(
            sum(x["mis_size_nli_only"] for x in mets) / n, 4)
        # overall weakest by mean (require >=3 questions, mirroring aggregation A)
        if row.get("n_questions", n) >= 3 and mw > best_val:
            best_val, best_idx = mw, idx_str

    summary["weakest_loop_index_overall_nli_only"] = best_idx
    summary["aggregation_B_per_question_weakest_loop_distribution_nli_only"] = \
        {str(k): v for k, v in sorted(weakest_dist_nli.items())}
    summary.setdefault("config", {})["nli_only_beta"] = args.beta

    for path, obj in [(args.per_question, pq), (args.summary, summary)]:
        shutil.copyfile(path, path + ".bak")
        json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)
    print("Done. Backups written to *.bak")
    print("weakest_loop_index_overall_nli_only =", best_idx)
    print("aggregation_B (NLI-only) =",
          summary["aggregation_B_per_question_weakest_loop_distribution_nli_only"])


if __name__ == "__main__":
    main()
