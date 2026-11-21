#!/usr/bin/env python3
"""Select questions and write the MH-Web corpus plus the ReAct data files
(plan doc sections 2.3, 2.5, 2.7).

Inputs: queries_<ds>.json (01), labeled_docs.jsonl + relevance_stats.json
(04), the raw MuSiQue dev file (for the original 20 paragraphs/question) and
the paper's attack-target file (results/adv_targeted_results/musique.json).

Question selection: a question is kept when its searches are complete and
every hop has >= --min-sources answer-bearing source clusters in the corpus
pool being built. The paper's 100 targets are taken first, in their
original order; failing ones are replaced by passing reserves, in order,
until --n questions are kept.

Corpus = web snippets from ALL collected questions (the non-selected ones
act as natural distractors) + the original MuSiQue paragraphs of the
selected questions. Full-question search results are excluded unless
--include-full-question (pilot: 22/50 of them already state the final
answer, which lets the agent skip hops). Every row has:
  _id, docid, title, text, url, source, qids, labels, published_at, host
`docid` is a source cluster id that is consistent across questions: rows
sharing a canonical URL, or merged as near-duplicates under any question,
get the same docid. The retriever's one-doc-per-docid rule uses it.

Writes under --repo-root (KidnapRag/):
  ReAct/datasets/<name>/corpus.jsonl            retriever corpus (gitignored)
  ReAct/datasets/<name>/hops.jsonl              selected questions + hop stats
  ReAct/datasets/<name>/hops_rrag.json          each hop as a ReliabilityRAG item (G2 probe)
  ReAct/datasets/<name>/build_stats.json, selected_qids.txt
  ReAct/ReAct/data/<name>_dev_simplified.json   questions for the ReAct wrapper
  ReAct/results/adv_targeted_results/<name>.json and <name>_qid_to_idx.json
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (contains_answer, read_json, read_jsonl, sha1, write_json,  # noqa: E402
                    write_jsonl)


class StrUnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--labeled-dir", required=True, type=Path, help="04_label_relevance.py --out-dir")
    ap.add_argument("--raw", required=True, type=Path, help="musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--adv-targets", required=True, type=Path,
                    help="ReAct/results/adv_targeted_results/musique.json")
    ap.add_argument("--repo-root", required=True, type=Path, help="KidnapRag/")
    ap.add_argument("--name", default="mhweb_musique")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min-sources", type=int, default=3)
    ap.add_argument("--include-full-question", action="store_true")
    ap.add_argument("--rrag-context", type=int, default=30,
                    help="snippets per hop in hops_rrag.json (default 30)")
    args = ap.parse_args()

    # Select questions by the hop-search pool in BOTH builds, so the default
    # corpus and the --include-full-question ablation cover the exact same
    # questions; the flag only changes which docs go into the corpus.
    pool_key = "bearing_sources_hop_pool"
    queries = {e["qid"]: e for e in read_json(args.queries)}
    stats = {s["qid"]: s for s in read_json(args.labeled_dir / "relevance_stats.json")["questions"]}
    adv = read_json(args.adv_targets)
    default_incorrect = collections.Counter(v["incorrect answer"] for v in adv.values()).most_common(1)[0][0]

    # ---- question selection -------------------------------------------
    def ok(qid):
        s = stats.get(qid)
        return bool(s) and s["complete"] and all(h[pool_key] >= args.min_sources for h in s["hops"])

    targets = sorted((e for e in queries.values() if not e["reserve"]), key=lambda e: e["orig_idx"])
    reserves = [e for e in queries.values() if e["reserve"]]
    kept_t = [e for e in targets if ok(e["qid"])]
    kept_r = [e for e in reserves if ok(e["qid"])]
    selected = (kept_t + kept_r)[:args.n]
    sel_ids = {e["qid"] for e in selected}
    missing = [e["qid"] for e in targets if e["qid"] not in stats or not stats[e["qid"]]["complete"]]
    if missing:
        print(f"WARNING: {len(missing)} target questions have incomplete searches (rerun "
              f"02_keenable_collect.py first): {missing[:5]}", file=sys.stderr)
    if len(selected) < args.n:
        print(f"WARNING: only {len(selected)} questions pass (asked for {args.n}); "
              f"add reserves with 01_build_hops.py --reserve and collect them", file=sys.stderr)

    # ---- web docs: global dedup + consistent docids ---------------------
    uf = StrUnionFind()
    web = {}
    for r in read_jsonl(args.labeled_dir / "labeled_docs.jsonl"):
        if not args.include_full_question and "hop" not in r["pools"]:
            continue
        uf.union("D:" + r["docid"], "U:" + r["canonical_url"])
        lab = {"final": r["labels"]["final"], "hops": r["labels"]["hops"]}
        if r["_id"] in web:
            w = web[r["_id"]]
            if r["qid"] not in w["qids"]:
                w["qids"].append(r["qid"])
                w["labels"][r["qid"]] = lab
            w["_docids"].add(r["docid"])
            continue
        web[r["_id"]] = {
            "_id": r["_id"], "title": r["title"], "text": r["text"], "url": r["url"],
            "source": "keenable", "qids": [r["qid"]], "labels": {r["qid"]: lab},
            "published_at": r.get("published_at"), "host": r["host"],
            "_canon": r["canonical_url"], "_docids": {r["docid"]},
        }
    for w in web.values():
        for d in w["_docids"]:
            uf.union("D:" + d, "U:" + w["_canon"])
    comp_min = {}
    for w in web.values():
        root = uf.find("U:" + w["_canon"])
        comp_min[root] = min(comp_min.get(root, w["_canon"]), w["_canon"])
    for w in web.values():
        w["docid"] = sha1(comp_min[uf.find("U:" + w.pop("_canon"))])[:16]
        w.pop("_docids")

    # ---- original MuSiQue paragraphs for the selected questions ---------
    raw = {r["id"]: r for r in read_jsonl(args.raw) if r["id"] in sel_ids}
    paras = {}
    for qid in [e["qid"] for e in selected]:
        e = queries[qid]
        sup_idx = {h["paragraph_support_idx"] for h in raw[qid]["question_decomposition"]}
        for i, p in enumerate(raw[qid]["paragraphs"]):
            pid = "mq::" + sha1(p["title"] + "\n" + p["paragraph_text"])[:16]
            blob = p["title"] + " " + p["paragraph_text"]
            lab = {"final": contains_answer(blob, e["answer_aliases"]),
                   "hops": [h["hop_idx"] for h in e["hops"] if contains_answer(blob, h["aliases"])],
                   "supporting": i in sup_idx}
            if pid in paras:
                paras[pid]["qids"].append(qid)
                paras[pid]["labels"][qid] = lab
                continue
            paras[pid] = {"_id": pid, "docid": pid, "title": p["title"], "text": p["paragraph_text"],
                          "url": None, "source": "musique_paragraph", "qids": [qid],
                          "labels": {qid: lab}, "published_at": None, "host": None}

    corpus = list(web.values()) + list(paras.values())
    out = args.repo_root / "ReAct" / "datasets" / args.name
    write_jsonl(out / "corpus.jsonl", corpus)

    # ---- per-question files ---------------------------------------------
    hop_rows, rrag = [], []
    by_qid_web = collections.defaultdict(list)
    for w in web.values():
        for q in w["qids"]:
            by_qid_web[q].append(w)
    for e in selected:
        hop_rows.append(e | {"stats": stats[e["qid"]]})
        for h in e["hops"]:
            ctx = [w for w in by_qid_web[e["qid"]]]
            ctx.sort(key=lambda w: (h["hop_idx"] not in w["labels"][e["qid"]]["hops"], w["docid"]))
            seen, items = set(), []
            for w in ctx:  # one snippet per source, answer-bearing first
                if w["docid"] in seen:
                    continue
                seen.add(w["docid"])
                items.append({"title": w["title"], "text": w["text"], "link": w["url"]})
            rrag.append({"id": f"{e['qid']}#hop{h['hop_idx']}", "qid": e["qid"], "hop_idx": h["hop_idx"],
                         "question": h["query"], "correct answer": h["aliases"],
                         "context": items[:args.rrag_context]})
    write_jsonl(out / "hops.jsonl", hop_rows)
    write_json(out / "hops_rrag.json", rrag)
    (out / "selected_qids.txt").write_text(
        "".join(f"{e['qid']}\t{'reserve' if e['reserve'] else 'target'}\n" for e in selected), encoding="utf-8")

    simplified = [{"question": e["question"], "answer": e["answer"], "type": e["hop_type"], "id": e["qid"]}
                  for e in selected]
    write_json(args.repo_root / "ReAct" / "ReAct" / "data" / f"{args.name}_dev_simplified.json", simplified)
    adv_out = {}
    for e in selected:
        adv_out[e["qid"]] = adv.get(e["qid"]) or {
            "id": e["qid"], "question": e["question"], "correct answer": e["answer"],
            "incorrect answer": default_incorrect, "adv_texts": []}
    res = args.repo_root / "ReAct" / "results" / "adv_targeted_results"
    write_json(res / f"{args.name}.json", adv_out, indent=4)
    write_json(res / f"{args.name}_qid_to_idx.json", {e["qid"]: i for i, e in enumerate(selected)}, indent=4)

    n_src = len({w["docid"] for w in web.values()})
    build = {
        "name": args.name, "include_full_question": args.include_full_question,
        "min_sources": args.min_sources,
        "selected": len(selected), "targets_kept": len([e for e in selected if not e["reserve"]]),
        "reserves_used": len([e for e in selected if e["reserve"]]),
        "targets_failing": [e["qid"] for e in targets if e["qid"] not in sel_ids],
        "corpus_rows": len(corpus), "web_rows": len(web), "web_source_clusters": n_src,
        "musique_paragraph_rows": len(paras),
        "hosts_top20": collections.Counter(w["host"] for w in web.values()).most_common(20),
    }
    write_json(out / "build_stats.json", build)
    print(f"selected {len(selected)} questions ({build['targets_kept']} paper targets + "
          f"{build['reserves_used']} reserves); {len(build['targets_failing'])} targets dropped")
    print(f"corpus: {len(corpus)} rows = {len(web)} web snippets ({n_src} source clusters) + "
          f"{len(paras)} MuSiQue paragraphs -> {out / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
