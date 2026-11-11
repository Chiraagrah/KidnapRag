"""Convert BrowseComp-Plus into KidnapRAG/WebThinker (and ReliabilityRAG) formats.

Inputs (login node, after running BrowseComp-Plus's decrypt_dataset.py):
  --queries  browsecomp_plus_decrypted.jsonl  (query_id, query, answer, gold_docs, evidence_docs, negative_docs)
  --corpus   Tevatron/browsecomp-plus-corpus  (a .jsonl/.parquet with docid, url, text) or HF dataset name

Outputs under --out_dir:
  bcp_select<N>.json            KidnapRAG query file
  bcp_select<N>_rrag.json       ReliabilityRAG-style file
  bcp_dev_corpus.jsonl          chunked gold+evidence+negative docs of selected queries
  bcp_full_corpus.jsonl         chunked full corpus (only with --full)
  selected_qids.txt, stats.json
"""
import argparse, json, os, random, re, statistics


def docid_of(x):
    return str(x["docid"] if isinstance(x, dict) else x)


def load_queries(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def iter_corpus(path):
    if os.path.exists(path):
        if path.endswith(".parquet"):
            import pandas as pd
            for r in pd.read_parquet(path).to_dict("records"):
                yield r
        else:
            with open(path, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        yield json.loads(l)
    else:  # HF dataset name
        from datasets import load_dataset
        for r in load_dataset(path, split="train"):
            yield r


FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def split_front_matter(text):
    m = FM.match(text)
    title = ""
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else ""
        text = text[m.end():]
    return title, text


def chunk_text(text, size=300, overlap=50):
    """~size-word chunks on paragraph boundaries where possible, with word overlap."""
    words_per_par = [(p, len(p.split())) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, n = [], [], 0
    for p, w in words_per_par:
        if w > size:  # hard-split long paragraph
            ws = p.split()
            step = size - overlap
            for i in range(0, len(ws), step):
                piece = ws[i:i + size]
                if cur:
                    chunks.append(" ".join(cur)); cur, n = [], 0
                chunks.append(" ".join(piece))
                if i + size >= len(ws):
                    break
            continue
        if n + w > size and cur:
            chunks.append(" ".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, n = ([tail] if tail else []), len(tail.split())
        cur.append(p.strip()); n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text.strip()]


def doc_chunks(rec, size, overlap):
    docid = str(rec["docid"])
    title, body = split_front_matter(rec.get("text", ""))
    title = title or rec.get("title", "") or ""
    for i, c in enumerate(chunk_text(body, size, overlap)):
        yield {"id": f"{docid}#{i}", "docid": docid, "title": title, "text": c, "url": rec.get("url", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--min_gold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk_words", type=int, default=300)
    ap.add_argument("--overlap_words", type=int, default=50)
    ap.add_argument("--full", action="store_true", help="also write the full chunked corpus")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    qs = load_queries(a.queries)
    for q in qs:
        q["_gold"] = [docid_of(d) for d in q.get("gold_docs", [])]
        q["_evid"] = [docid_of(d) for d in q.get("evidence_docs", [])]
        q["_neg"] = [docid_of(d) for d in q.get("negative_docs", [])]
    elig = [q for q in qs if len(q["_gold"]) >= a.min_gold]
    if len(elig) < a.n:
        print(f"only {len(elig)} queries with >= {a.min_gold} gold; relaxing to >= 2")
        elig = sorted([q for q in qs if len(q["_gold"]) >= 2], key=lambda q: -len(q["_gold"]))
    random.Random(a.seed).shuffle(elig)
    sel = elig[:a.n]
    with open(os.path.join(a.out_dir, "selected_qids.txt"), "w") as f:
        f.write("\n".join(str(q["query_id"]) for q in sel))

    keep = set()
    for q in sel:
        keep.update(q["_gold"], q["_evid"], q["_neg"])

    n_chunks_dev = n_chunks_full = 0
    chunks_per_gold, lens = {}, []
    gold_all = {d for q in sel for d in q["_gold"]}
    dev_f = open(os.path.join(a.out_dir, "bcp_dev_corpus.jsonl"), "w", encoding="utf-8")
    full_f = open(os.path.join(a.out_dir, "bcp_full_corpus.jsonl"), "w", encoding="utf-8") if a.full else None
    for rec in iter_corpus(a.corpus):
        did = str(rec["docid"])
        in_dev = did in keep
        if not (in_dev or full_f):
            continue
        n = 0
        for ch in doc_chunks(rec, a.chunk_words, a.overlap_words):
            line = json.dumps(ch, ensure_ascii=False) + "\n"
            n += 1
            if in_dev:
                dev_f.write(line); n_chunks_dev += 1
                lens.append(len(ch["text"].split()))
            if full_f:
                full_f.write(line); n_chunks_full += 1
        if did in gold_all:
            chunks_per_gold[did] = n
    dev_f.close()
    if full_f: full_f.close()

    out, rrag = [], []
    for q in sel:
        qid = str(q["query_id"])
        out.append({
            "id": qid, "qid": qid, "question": q["query"], "answer": q["answer"],
            "correct answer": q["answer"], "incorrect answer": "", "adv_texts": [],
            "gold_docids": q["_gold"], "evidence_docids": q["_evid"],
        })
        rrag.append({"id": qid, "question": q["query"], "correct answer": [q["answer"]],
                     "incorrect answer": "", "incorrect_context": [], "context": [],
                     "expanded answer": []})
    json.dump(out, open(os.path.join(a.out_dir, f"bcp_select{a.n}.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(rrag, open(os.path.join(a.out_dir, f"bcp_select{a.n}_rrag.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    stats = {
        "queries_total": len(qs), "queries_selected": len(sel),
        "gold_per_query_mean": statistics.mean(len(q["_gold"]) for q in sel),
        "evidence_per_query_mean": statistics.mean(len(q["_evid"]) for q in sel),
        "dev_docs": len(keep), "dev_chunks": n_chunks_dev, "full_chunks": n_chunks_full,
        "chunk_words_mean": statistics.mean(lens) if lens else 0,
        "chunk_words_max": max(lens) if lens else 0,
        "chunks_per_gold_doc_mean": statistics.mean(chunks_per_gold.values()) if chunks_per_gold else 0,
    }
    json.dump(stats, open(os.path.join(a.out_dir, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
