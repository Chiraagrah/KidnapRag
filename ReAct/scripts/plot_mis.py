"""
Plot ReliabilityRAG MIS diagnostics produced by attack_react_mis.py.

Two modes:
  1) Aggregate weakness-by-loop-index bar chart (from *_mis_summary.json):
       python plot_mis.py --summary .../hotpotqa_qwen25_mis_summary.json

  2) A single loop's contradiction graph (from *_mis_per_question.json):
       python plot_mis.py --per_question .../hotpotqa_qwen25_mis_per_question.json \
                          --qid <QID> --step <N>

  3) The adjacency / contradiction-probability matrix of a single loop as text
     (no matplotlib needed):
       python plot_mis.py --per_question .../hotpotqa_qwen25_mis_per_question.json \
                          --qid <QID> --step <N> --print_matrix

The plotting modes need matplotlib (imported lazily, so --print_matrix works
without it); no networkx (a manual circular layout is used).
"""
import os
import json
import math
import argparse


def _find_loop(per_question_path, qid, step):
    with open(per_question_path, encoding="utf-8") as f:
        data = json.load(f)
    q = next((d for d in data if str(d["id"]) == str(qid)), None)
    if q is None:
        raise SystemExit(f"qid {qid} not found")
    rec = next((s for s in q["steps"] if s["step"] == int(step) and s.get("mis")), None)
    if rec is None:
        raise SystemExit(f"step {step} for qid {qid} has no MIS analysis")
    return rec["mis"]


def print_matrix(per_question_path, qid, step):
    """Print the contradiction graph as adjacency + probability matrices."""
    mis = _find_loop(per_question_path, qid, step)
    nodes = mis["nodes"]
    n = len(nodes)
    adj = mis["adjacency"]
    prob = mis["contra_matrix"]
    met = mis["metrics"]

    print(f"\nqid {qid} | loop {step} | query: {mis['query']}")
    print(f"MIS={met['mis_size']}/{met['n_docs']} selected={mis['mis_selected']} "
          f"edges={met['edge_count']} weakness={met['weakness']} "
          f"(edge iff contradiction prob >= beta)")

    print("\nnodes:")
    for nd in nodes:
        tag = []
        if nd["in_mis"]:
            tag.append("MIS")
        if nd["is_poisoned"]:
            tag.append("POISONED")
        print(f"  x{nd['rank']} [{','.join(tag) or '-':>8}] score={nd['score']:.3f} "
              f"| {nd['title'][:40]!r} -> {nd['isolated_answer'][:60]!r}")

    hdr = "     " + " ".join(f"x{j}" for j in range(n))
    print("\nadjacency (1 = contradiction edge):")
    print(hdr)
    for i in range(n):
        print(f"  x{i} " + "  ".join(str(adj[i][j]) for j in range(n)))

    print("\ncontradiction probabilities:")
    print("      " + "    ".join(f"x{j}" for j in range(n)))
    for i in range(n):
        print(f"  x{i} " + " ".join(f"{prob[i][j]:.2f}" for j in range(n)))


def plot_aggregate(summary_path, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    per_index = summary["aggregation_A_per_loop_index"]
    idxs = sorted(per_index, key=lambda s: int(s))
    weakness = [per_index[i]["mean_weakness"] for i in idxs]
    excluded = [per_index[i]["mean_excluded_fraction"] for i in idxs]
    density = [per_index[i]["mean_edge_density"] for i in idxs]
    support = [per_index[i]["n_questions"] for i in idxs]

    x = range(len(idxs))
    fig, ax = plt.subplots(figsize=(max(7, len(idxs) * 0.8), 4.5))
    w = 0.27
    ax.bar([i - w for i in x], weakness, w, label="weakness (1 - MIS/k)")
    ax.bar(list(x), excluded, w, label="excluded fraction")
    ax.bar([i + w for i in x], density, w, label="edge density")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{i}\n(n={n})" for i, n in zip(idxs, support)])
    ax.set_xlabel("ReAct loop index")
    ax.set_ylabel("mean score")
    ax.set_title(f"Per-loop MIS weakness — {summary['config']['dataset']} / "
                 f"{summary['config'].get('scenario', '')}\n"
                 f"weakest loop overall: {summary.get('weakest_loop_index_overall')}")
    ax.legend()
    fig.tight_layout()
    out_path = out_path or os.path.splitext(summary_path)[0] + "_weakness.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def _circular_layout(n, radius=1.0):
    return [(radius * math.cos(2 * math.pi * k / n),
             radius * math.sin(2 * math.pi * k / n)) for k in range(n)]


def plot_graph(per_question_path, qid, step, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mis = _find_loop(per_question_path, qid, step)
    nodes, edges = mis["nodes"], mis["edges"]
    n = len(nodes)
    pos = _circular_layout(n)

    fig, ax = plt.subplots(figsize=(7, 7))
    for e in edges:                       # contradiction edges
        (x1, y1), (x2, y2) = pos[e["i"]], pos[e["j"]]
        ax.plot([x1, x2], [y1, y2], color="#c0392b", lw=1.5, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{e['contra_prob']:.2f}",
                color="#c0392b", fontsize=8, ha="center", zorder=3)
    for k, nd in enumerate(nodes):        # document nodes
        x, y = pos[k]
        in_mis = nd["in_mis"]
        poisoned = nd["is_poisoned"]
        face = "#f39c12" if poisoned else ("#2ecc71" if in_mis else "#bdc3c7")
        edge_c = "#1e8449" if in_mis else "#7f8c8d"
        ax.scatter([x], [y], s=1400, c=face, edgecolors=edge_c, linewidths=2.5, zorder=2)
        ax.text(x, y, f"x{k}\n(rank {nd['rank']})", ha="center", va="center",
                fontsize=9, zorder=4)
        ax.text(x * 1.35, y * 1.35, (nd["isolated_answer"] or "")[:40],
                ha="center", va="center", fontsize=7, color="#34495e")

    met = mis["metrics"]
    ax.set_title(f"Contradiction graph — qid {qid}, loop {step}\n"
                 f"query: {mis['query'][:70]}\n"
                 f"MIS={met['mis_size']}/{met['n_docs']} (green), "
                 f"edges={met['edge_count']}, weakness={met['weakness']}")
    ax.set_axis_off()
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    fig.tight_layout()
    out_path = out_path or os.path.join(
        os.path.dirname(per_question_path), f"graph_{qid}_step{step}.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", help="*_mis_summary.json -> aggregate bar chart")
    ap.add_argument("--per_question", help="*_mis_per_question.json -> one graph")
    ap.add_argument("--qid")
    ap.add_argument("--step", type=int)
    ap.add_argument("--print_matrix", action="store_true",
                    help="print adjacency + probability matrices as text (no matplotlib)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.summary:
        plot_aggregate(a.summary, a.out)
    elif a.per_question and a.qid is not None and a.step is not None:
        if a.print_matrix:
            print_matrix(a.per_question, a.qid, a.step)
        else:
            plot_graph(a.per_question, a.qid, a.step, a.out)
    else:
        ap.error("pass --summary, or --per_question with --qid and --step")
