#!/usr/bin/env python3
"""Clean, de-duplicate and relevance-label the collected web results
(plan doc sections 2.2b, 2.2c, 2.5; fixes for the 2026-11-20 pilot findings).

Input:  searches.json from 02_keenable_collect.py + queries_<ds>.json from
        01_build_hops.py.
Output: labeled_docs.jsonl  -- one row per distinct (page, snippet) per qid
        relevance_stats.json -- per-qid / per-hop redundancy and the G1 gate

Per question, every search result goes through:
  1. clean_snippet(): strips markdown links, images, citation markers
     ("[[1]](./Ted_Mosby#cite_note-1)"), emphasis and table rules.
     Results shorter than --min-chars after cleaning are dropped.
  2. Question-quote filter, applied to FULL-QUESTION results only: drop a
     result that reproduces >= --quote-threshold of the question's word
     trigrams (quiz / dataset-viewer pages on hosts not in exclude_domains).
     Hop searches are not filtered: their queries are short, so a real
     topical page overlaps them heavily (6 false hits on Wikipedia/IMDb in
     the pilot with a bag-of-words rule).
  3. canonical_url(): folds mirrors (wikipedia.com, wikiwand.com, m. hosts,
     tracking params) onto one address.
  4. Exact duplicates (same canonical URL + same normalised text) are merged,
     keeping every (search_type, hop_idx, rank) they came from.
  5. Source clusters: results are grouped when they share a canonical URL or
     their texts overlap >= --dup-threshold (5-word shingle overlap
     coefficient, catches syndicated copies). `docid` = the cluster id, so the
     retriever's one-doc-per-docid rule (plan section 3) cannot let copies of
     one source form a fake agreeing majority.
  6. Labels: `final` (final answer or alias present) and `hops` (indices of
     hop answers present), verbatim and case/accent-insensitive, as in
     ReliabilityRAG/data/check_relevance.py.

Redundancy is counted in distinct SOURCE CLUSTERS, not results. A hop passes
G1 when >= --min-sources clusters from that question's hop searches bear its
answer. Full-question results are labelled but kept apart (pool
"full_question"), because in the pilot 22/50 of them already stated the final
answer; 05_build_corpus.py leaves them out unless --include-full-question.
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (UnionFind, canonical_url, clean_snippet, contains_answer, host_of,  # noqa: E402
                    normalize, overlap_coefficient, question_trigram_overlap, read_json,
                    sha1, shingles, write_json, write_jsonl)


def process_question(entry, searches, args):
    qid, question = entry["qid"], entry["question"]
    hop_aliases = {h["hop_idx"]: h["aliases"] for h in entry["hops"]}
    drops = defaultdict(int)
    docs = {}  # (canon, normtext) -> doc
    for s in searches:
        if s.get("error"):
            drops["search_error"] += 1
            continue
        pool = "full_question" if s["search_type"] == "full_question" else "hop"
        for rank, r in enumerate(s["results"]):
            text = clean_snippet(r.get("snippet", ""))
            title = clean_snippet(r.get("title", ""))
            if len(text) < args.min_chars:
                drops["too_short"] += 1
                continue
            if pool == "full_question" and \
                    question_trigram_overlap(question, title + " " + text) >= args.quote_threshold:
                drops["quotes_question"] += 1
                continue
            canon = canonical_url(r.get("url", ""))
            key = (canon, normalize(text))
            prov = {"search_type": s["search_type"], "hop_idx": s["hop_idx"], "rank": rank,
                    "query": s["query_text"]}
            if key in docs:
                docs[key]["provenance"].append(prov)
                drops["exact_duplicate"] += 1
                continue
            docs[key] = {
                "qid": qid, "title": title, "text": text, "url": r.get("url", ""),
                "canonical_url": canon, "host": host_of(r.get("url", "")),
                "published_at": r.get("published_at"), "acquired_at": r.get("acquired_at"),
                "provenance": [prov],
            }

    rows = list(docs.values())
    for d in rows:
        d["pools"] = sorted({"full_question" if p["search_type"] == "full_question" else "hop"
                             for p in d["provenance"]})
        d["best_rank"] = min(p["rank"] for p in d["provenance"])

    # Source clusters: same canonical URL, or near-duplicate text.
    uf = UnionFind(len(rows))
    by_canon = {}
    for i, d in enumerate(rows):
        if d["canonical_url"] in by_canon:
            uf.union(i, by_canon[d["canonical_url"]])
        else:
            by_canon[d["canonical_url"]] = i
    sh = [shingles(d["title"] + " " + d["text"]) for d in rows]
    near = 0
    for i in range(len(rows)):
        if len(sh[i]) < args.min_shingles:
            continue
        for j in range(i + 1, len(rows)):
            if len(sh[j]) >= args.min_shingles and uf.find(i) != uf.find(j) and \
                    overlap_coefficient(sh[i], sh[j]) >= args.dup_threshold:
                uf.union(i, j)
                near += 1
    members = defaultdict(list)
    for i in range(len(rows)):
        members[uf.find(i)].append(i)
    for root, idxs in members.items():
        cid = sha1(min(rows[i]["canonical_url"] for i in idxs))[:16]
        for i in idxs:
            rows[i]["docid"] = cid
            rows[i]["cluster_size"] = len(idxs)

    for d in rows:
        blob = d["title"] + " " + d["text"]
        d["labels"] = {"final": contains_answer(blob, entry["answer_aliases"]),
                       "hops": [k for k, al in hop_aliases.items() if contains_answer(blob, al)]}
        d["_id"] = f"web::{sha1(d['canonical_url'])[:16]}::{sha1(normalize(d['text']))[:12]}"

    # Redundancy per hop, in distinct source clusters.
    def clusters(pool, pred):
        return len({d["docid"] for d in rows if pool in d["pools"] and pred(d)})

    hop_stats = []
    for h in entry["hops"]:
        k = h["hop_idx"]
        own = [d for d in rows if any(p["hop_idx"] == k and p["search_type"] != "full_question"
                                      for p in d["provenance"])]
        hop_stats.append({
            "hop_idx": k, "answer": h["answer"],
            "results_from_own_searches": len(own),
            "bearing_results_own": sum(k in d["labels"]["hops"] for d in own),
            "bearing_sources_own": len({d["docid"] for d in own if k in d["labels"]["hops"]}),
            "bearing_sources_hop_pool": clusters("hop", lambda d, k=k: k in d["labels"]["hops"]),
            "bearing_sources_with_fq": len({d["docid"] for d in rows if k in d["labels"]["hops"]}),
        })
    passes = all(hs["bearing_sources_hop_pool"] >= args.min_sources for hs in hop_stats)
    passes_fq = all(hs["bearing_sources_with_fq"] >= args.min_sources for hs in hop_stats)
    q_stats = {
        "qid": qid, "reserve": entry["reserve"], "orig_idx": entry["orig_idx"],
        "hop_type": entry["hop_type"], "n_searches": len(searches),
        "n_search_errors": drops["search_error"], "n_docs": len(rows),
        "n_sources": len(members), "near_duplicate_merges": near,
        "drops": dict(drops),
        "final_bearing_sources_hop_pool": clusters("hop", lambda d: d["labels"]["final"]),
        "final_bearing_sources_fq_pool": clusters("full_question", lambda d: d["labels"]["final"]),
        "hops": hop_stats, "passes_G1": passes, "passes_G1_with_fq": passes_fq,
        "complete": len(searches) == 1 + 2 * len(entry["hops"]) and drops["search_error"] == 0,
    }
    return rows, q_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--searches", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--quote-threshold", type=float, default=0.6)
    ap.add_argument("--dup-threshold", type=float, default=0.8)
    ap.add_argument("--min-shingles", type=int, default=8)
    ap.add_argument("--min-sources", type=int, default=3,
                    help="G1: answer-bearing source clusters needed per hop (default 3)")
    args = ap.parse_args()

    queries = read_json(args.queries)
    by_qid = defaultdict(list)
    for s in read_json(args.searches):
        by_qid[s["qid"]].append(s)

    all_rows, stats = [], []
    for entry in queries:
        if entry["qid"] not in by_qid:
            continue
        rows, st = process_question(entry, by_qid[entry["qid"]], args)
        all_rows.extend(rows)
        stats.append(st)
    if not stats:
        sys.exit("no queries matched any searches -- wrong --searches file?")

    write_jsonl(args.out_dir / "labeled_docs.jsonl", all_rows)

    hops = [h for s in stats for h in s["hops"]]
    targets = [s for s in stats if not s["reserve"]]
    summary = {
        "n_questions_with_searches": len(stats),
        "n_complete": sum(s["complete"] for s in stats),
        "n_docs": len(all_rows),
        "drops": {k: sum(s["drops"].get(k, 0) for s in stats)
                  for k in ("too_short", "quotes_question", "exact_duplicate", "search_error")},
        "near_duplicate_merges": sum(s["near_duplicate_merges"] for s in stats),
        "per_hop_bearing_sources_hop_pool": {
            "mean": round(statistics.mean(h["bearing_sources_hop_pool"] for h in hops), 2),
            "median": statistics.median(h["bearing_sources_hop_pool"] for h in hops),
            "share_ge_min_sources": round(sum(h["bearing_sources_hop_pool"] >= args.min_sources
                                              for h in hops) / len(hops), 3),
            "share_zero": round(sum(h["bearing_sources_hop_pool"] == 0 for h in hops) / len(hops), 3),
        },
        "mean_bearing_results_own_search_per_hop": round(statistics.mean(
            h["bearing_results_own"] for h in hops), 2),
        "G1_pass_targets": f"{sum(s['passes_G1'] for s in targets)}/{len(targets)}",
        "G1_pass_reserves": f"{sum(s['passes_G1'] for s in stats if s['reserve'])}/"
                            f"{sum(s['reserve'] for s in stats)}",
        "G1_pass_targets_with_fq": f"{sum(s['passes_G1_with_fq'] for s in targets)}/{len(targets)}",
        "final_answer_in_fq_pool_share": round(sum(s["final_bearing_sources_fq_pool"] > 0
                                                   for s in stats) / len(stats), 3),
        "params": vars(args) | {"queries": str(args.queries), "searches": str(args.searches),
                                "out_dir": str(args.out_dir)},
    }
    write_json(args.out_dir / "relevance_stats.json", {"summary": summary, "questions": stats})

    print(f"{len(stats)} questions, {len(all_rows)} docs -> {args.out_dir}")
    for k, v in summary.items():
        if k != "params":
            print(f"  {k}: {v}")
    failing = [(s["qid"], [h["bearing_sources_hop_pool"] for h in s["hops"]])
               for s in targets if not s["passes_G1"]]
    if failing:
        print(f"  targets failing G1 (bearing sources per hop): {failing[:8]}"
              f"{' ...' if len(failing) > 8 else ''}")


if __name__ == "__main__":
    main()
