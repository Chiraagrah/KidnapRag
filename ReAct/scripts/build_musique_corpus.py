#!/usr/bin/env python
"""Build a deduplicated MuSiQue retrieval corpus (+ gold support map).

Two input modes:

  --source  <musique_ans_v1.0_dev.jsonl>   (official MuSiQue download)
      Extracts every paragraph across all questions, dedups by (title, text),
      writes corpus.jsonl, and ALSO writes support_map.json: for each question,
      its decomposition (sub-questions, sub-answers, and the unique doc id that
      supports each hop). The support map is the ground truth the weakest-link
      targeting needs later.

  --from_corpus <corpus.jsonl>             (the KidnapRAG-shipped pooled corpus)
      Just dedups by text (identical passages differ only by question-scoped
      _id). Produces the same unique passage set as --source, but no support map.

Output corpus rows: {"_id","title","text"} (text = what E5_Retriever reads).

Examples
  python build_musique_corpus.py --source data/musique_ans_v1.0_dev.jsonl \
      --out_corpus datasets/musique/corpus.jsonl \
      --out_supportmap datasets/musique/support_map.json
  python build_musique_corpus.py --from_corpus datasets/musique/corpus_raw.jsonl \
      --out_corpus datasets/musique/corpus.jsonl
"""
import argparse, json, collections


def norm(s):
    return (s or "").strip()


def build_from_source(path):
    seen = {}            # (title,text) -> _id
    corpus = []          # unique docs in first-seen order
    support_map = {}     # qid -> [{hop, sub_question, sub_answer, gold_doc_id, gold_title}]
    n_para = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q.get("id")
            # map this question's local paragraph idx -> global unique doc id
            local = {}
            for p in q.get("paragraphs", []):
                n_para += 1
                title, text = norm(p.get("title")), norm(p.get("paragraph_text"))
                key = (title, text)
                if key not in seen:
                    did = f"mq_{len(seen)}"
                    seen[key] = did
                    corpus.append({"_id": did, "title": title, "text": text})
                local[p.get("idx")] = seen[key]
            # decomposition -> gold doc per hop
            steps = []
            for h, step in enumerate(q.get("question_decomposition", [])):
                gidx = step.get("paragraph_support_idx")
                gold_id = local.get(gidx)
                steps.append({
                    "hop": h,
                    "sub_question": step.get("question"),
                    "sub_answer": step.get("answer"),
                    "gold_doc_id": gold_id,
                    "gold_title": next((c["title"] for c in corpus if c["_id"] == gold_id), None),
                })
            support_map[qid] = steps
    return corpus, support_map, n_para


def build_from_corpus(path):
    seen = {}
    corpus = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n += 1
            title = norm(o.get("title"))
            text = norm(o.get("text", o.get("contents", "")))
            key = (title, text)
            if key not in seen:
                did = o.get("_id", o.get("id", f"mq_{len(seen)}"))
                seen[key] = did
                corpus.append({"_id": did, "title": title, "text": text})
    return corpus, None, n


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="official MuSiQue jsonl (has paragraphs + decomposition)")
    g.add_argument("--from_corpus", help="existing pooled corpus.jsonl to dedup by text")
    ap.add_argument("--out_corpus", required=True)
    ap.add_argument("--out_supportmap", default=None)
    args = ap.parse_args()

    if args.source:
        corpus, support_map, n_in = build_from_source(args.source)
    else:
        corpus, support_map, n_in = build_from_corpus(args.from_corpus)

    with open(args.out_corpus, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if support_map is not None and args.out_supportmap:
        with open(args.out_supportmap, "w", encoding="utf-8") as f:
            json.dump(support_map, f, ensure_ascii=False, indent=2)

    print(f"input passages:       {n_in}")
    print(f"unique docs written:  {len(corpus)}  ({n_in - len(corpus)} duplicates removed)")
    print(f"corpus -> {args.out_corpus}")
    if support_map is not None and args.out_supportmap:
        hops = collections.Counter(len(v) for v in support_map.values())
        print(f"support_map -> {args.out_supportmap}  ({len(support_map)} questions; hop-count dist {dict(sorted(hops.items()))})")


if __name__ == "__main__":
    main()
