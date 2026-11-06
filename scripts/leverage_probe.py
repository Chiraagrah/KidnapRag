#!/usr/bin/env python
"""Per-hop LEVERAGE probe for the weakest-link study (MuSiQue + Qwen).

Leverage(s) = does the agent's FINAL answer actually depend on hop s?
Measured by a counterfactual chain-completion that PROPAGATES the corruption:
  1. Corrupt hop s's sub-answer to a plausible, type-matched WRONG value A_s'.
  2. Re-answer every downstream hop IN ORDER with Qwen, feeding each corrupted
     answer into the next hop's query (so the error propagates through the chain).
  3. leverage(s) = 1 if the resulting final answer differs from the gold final
     answer, else 0.

A hop whose corruption the chain "heals" or that Qwen answers from parametric
memory yields leverage 0 -> poison-proof -> NOT a useful target. The terminal hop
yields leverage 1 by construction. Combine with injectability margins:
weak_score(s) = leverage(s) / margin(s).

MuSiQue '#N' refers to the N-th decomposition POSITION (1-indexed), not step['id'].

Self-contained: only needs vllm + the MuSiQue dev file. Lives in repo-root scripts/.
Run via scripts/run_leverage_musique.sh (needs a full A100 for Qwen-32B).
"""
import argparse, json, re, string, collections


def norm(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def resolve(text, ans_by_pos):
    """Resolve '#N' to the answer at decomposition position N; strip '>>'."""
    text = re.sub(r"#(\d+)", lambda m: str(ans_by_pos.get(int(m.group(1)), m.group(0))), text or "")
    return " ".join(text.replace(">>", " ").split())


WRONG_TMPL = (
    "Give one plausible but INCORRECT answer to the question below - a different "
    "entity of the same type as the correct answer. Respond with ONLY the wrong answer.\n"
    "Question: {q}\nCorrect answer: {a}\nWrong answer:"
)
ANSWER_TMPL = "Answer the question in a few words.\nQuestion: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--qid_to_idx", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=1,
              trust_remote_code=True, max_model_len=30000, gpu_memory_utilization=0.9)

    def gen(prompts, max_tokens):
        if not prompts:
            return []
        sp = SamplingParams(temperature=0, top_p=1, max_tokens=max_tokens,
                            seed=args.seed, stop=["\n"])
        return [o.outputs[0].text.strip() for o in llm.generate(prompts, sp, use_tqdm=True)]

    subset = set(json.load(open(args.qid_to_idx, encoding="utf-8")).keys()) if args.qid_to_idx else None
    questions = {}
    for line in open(args.dev, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        if subset is None or q["id"] in subset:
            questions[q["id"]] = q
    print(f"leverage probe on {len(questions)} questions")

    # one task per hop
    tasks = []
    for qid, q in questions.items():
        steps = q["question_decomposition"]
        H = len(steps)
        true_ans = {i + 1: steps[i]["answer"] for i in range(H)}
        for s in range(H):
            tasks.append({
                "qid": qid, "hop": s, "n_hops": H, "gold_final": q["answer"],
                "As": steps[s]["answer"], "steps": steps, "true_ans": true_ans,
                # q_s resolved for the counterfactual-answer prompt
                "q_s": resolve(steps[s]["question"], true_ans),
            })

    # batch 1: a plausible wrong counterfactual answer per hop
    wrong = gen([WRONG_TMPL.format(q=t["q_s"], a=t["As"]) for t in tasks], max_tokens=24)
    for t, w in zip(tasks, wrong):
        t["As_prime"] = w.split("\n")[0].strip().strip(".\"' ")
        # working answer map: truth, with hop s (position s+1) corrupted
        t["a"] = dict(t["true_ans"]); t["a"][t["hop"] + 1] = t["As_prime"]

    # rounds: re-answer each downstream hop in order, propagating the corruption
    max_round = max((t["n_hops"] for t in tasks), default=0)
    for r in range(1, max_round):
        batch = []
        for t in tasks:
            tgt = t["hop"] + r          # 0-indexed downstream hop to answer this round
            if tgt < t["n_hops"]:
                qi = resolve(t["steps"][tgt]["question"], t["a"])  # refs all < tgt, set
                t["_pend"] = tgt + 1    # position to fill
                batch.append((t, ANSWER_TMPL.format(q=qi)))
            else:
                t["_pend"] = None
        outs = gen([p for _, p in batch], max_tokens=24)
        for (t, _), o in zip(batch, outs):
            t["a"][t["_pend"]] = o.split("\n")[0].strip()

    records = []
    for t in tasks:
        new_final = t["a"][t["n_hops"]]      # position H = last hop's (propagated) answer
        valid = bool(t["As_prime"]) and norm(t["As_prime"]) != norm(t["As"])
        g, nf = norm(t["gold_final"]), norm(new_final)
        unchanged = (g in nf) or (nf in g) if (g and nf) else (g == nf)
        records.append({
            "qid": t["qid"], "hop": t["hop"], "n_hops": t["n_hops"],
            "sub_answer": t["As"], "counterfactual": t["As_prime"],
            "gold_final": t["gold_final"], "new_final": new_final,
            "corruption_valid": valid,
            "leverage": (None if not valid else int(not unchanged)),
        })

    valid = [r for r in records if r["leverage"] is not None]
    rate = lambda rs: round(sum(r["leverage"] for r in rs) / len(rs), 4) if rs else None
    byhop, bynh = collections.defaultdict(list), collections.defaultdict(list)
    for r in valid:
        byhop[r["hop"]].append(r); bynh[r["n_hops"]].append(r)
    summary = {
        "n_hops_total": len(records), "n_valid": len(valid),
        "overall_leverage": rate(valid),
        "leverage_by_hop": {str(h): {"n": len(byhop[h]), "leverage": rate(byhop[h])} for h in sorted(byhop)},
        "leverage_by_nhops": {str(n): {"n": len(bynh[n]), "leverage": rate(bynh[n])} for n in sorted(bynh)},
    }
    json.dump({"summary": summary, "records": records},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=== LEVERAGE SUMMARY ===")
    print(json.dumps(summary, indent=1))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
