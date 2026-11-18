#!/usr/bin/env python
"""Merge injectability + leverage into the per-question weak-link target list.

target hop = lowest injection cost (margin; gold-already-losing/absent = 0 = free),
tie-break by higher leverage, then earliest hop. Writes musique_weaklink_targets.json.

  python scripts/build_weaklink_targets.py \
      --injectability ReAct/results/mis_diagnostic/musique_injectability.json \
      --leverage      ReAct/results/mis_diagnostic/musique_leverage.json \
      --out           ReAct/results/mis_diagnostic/musique_weaklink_targets.json
"""
import argparse, json, collections

ap = argparse.ArgumentParser()
ap.add_argument("--injectability", required=True)
ap.add_argument("--leverage", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

inj = json.load(open(a.injectability, encoding="utf-8"))["records"]
lev = {(r["qid"], r["hop"]): r for r in json.load(open(a.leverage, encoding="utf-8"))["records"]}

byq = collections.defaultdict(list)
for r in inj:
    found = r["gold_rank"] >= 0
    margin = r["margin"]
    cost = 0.0 if (not found or margin is None or margin <= 0) else margin
    byq[r["qid"]].append({"qid": r["qid"], "hop": r["hop"], "n_hops": r["n_hops"],
                          "margin": margin, "injection_cost": cost,
                          "leverage": (lev.get((r["qid"], r["hop"])) or {}).get("leverage")})

targets = [min(hs, key=lambda h: (h["injection_cost"], -(h["leverage"] or 0), h["hop"]))
           for hs in byq.values()]
json.dump({"targets": targets}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"wrote {len(targets)} targets -> {a.out}")
print("target-hop distribution:", dict(sorted(collections.Counter(t["hop"] for t in targets).items())))
