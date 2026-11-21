#!/usr/bin/env python3
"""Build per-hop search queries for MuSiQue (plan doc section 2.1).

Reads the raw MuSiQue dev file (it keeps `question_decomposition`, which the
simplified ReAct files dropped) and the paper's 100 target qids, and writes a
queries file in the format 02_keenable_collect.py reads:

  {qid, question, answer, answer_aliases, hop_type, reserve, orig_idx,
   hops: [{hop_idx, query, paraphrase, raw_question, answer, aliases,
           support_title, answer_in_support}]}

How hop queries are made, with no LLM call (deterministic, reproducible):
  * "#k" placeholders are replaced by the gold answer of hop k. These are
    "oracle" queries: they are only used to gather the corpus, never shown
    to the agent, which writes its own queries at run time.
  * "Entity >> relation" hops (Wikidata-style, 59 of the 100 targets) are
    turned into a natural question by a relation template, e.g.
    "James Finch >> place of birth" -> "Where was James Finch born?".
    Unknown relations fall back to "What is the <relation> of <entity>?".
  * paraphrase = a keyword form ("James Finch place of birth" or the
    question minus its leading wh-word), for a second, differently-phrased
    search per hop.

Reserves (--reserve N): N extra answerable dev questions, sampled with a
fixed seed, collected alongside the targets so that 05_build_corpus.py can
top the set back up to 100 if some targets fail the redundancy filter
(plan doc section 2.5).

Growing the reserve pool later (--extend-from): pass an existing
queries_musique.json and a larger --reserve N. This keeps every qid already
in that file (so already-collected Keenable searches for them stay valid --
no need to re-collect) and samples only the additional reserves needed to
reach N, from the dev pool minus every qid already used (targets + existing
reserves). Without --extend-from, re-running with a bigger --reserve on the
same --seed is NOT guaranteed to keep the same reserve qids (Python's
random.sample() over a larger k is not a stable extension of a smaller k),
so it would silently invalidate prior collection.
"""
import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import contains_answer, read_json, read_jsonl, write_json  # noqa: E402

# Wikidata relation -> question template ({e} = entity). Covers the
# relations that appear >= 5 times in the MuSiQue dev set.
RELATION_TEMPLATES = {
    "located in the administrative territorial entity": "In which administrative region is {e} located?",
    "performer": "Who is the performer of {e}?",
    "place of birth": "Where was {e} born?",
    "country": "In which country is {e}?",
    "headquarters location": "Where is the headquarters of {e}?",
    "place of death": "Where did {e} die?",
    "shares border with": "What place shares a border with {e}?",
    "spouse": "Who is the spouse of {e}?",
    "member of sports team": "Which sports team did {e} play for?",
    "author": "Who is the author of {e}?",
    "operator": "Who operates {e}?",
    "record label": "What record label is {e} signed to?",
    "child": "Who is the child of {e}?",
    "creator": "Who created {e}?",
    "cast member": "Who is a cast member of {e}?",
    "country of citizenship": "What is the nationality of {e}?",
    "educated at": "Where was {e} educated?",
    "owned by": "Who owns {e}?",
    "father": "Who is the father of {e}?",
    "mother": "Who is the mother of {e}?",
    "platform": "What platform is {e} available on?",
    "member of": "What organization is {e} a member of?",
    "contains administrative territorial entity": "What administrative region is part of {e}?",
    "employer": "Who is the employer of {e}?",
    "manufacturer": "Who manufactures {e}?",
    "capital": "What is the capital of {e}?",
    "capital of": "What is {e} the capital of?",
    "winner": "Who won {e}?",
    "composer": "Who composed {e}?",
    "location of formation": "Where was {e} formed?",
    "sibling": "Who is the sibling of {e}?",
    "instance of": "What kind of thing is {e}?",
    "location": "Where is {e} located?",
    "producer": "Who produced {e}?",
    "part of": "What is {e} part of?",
    "has part": "What are the parts of {e}?",
    "work location": "Where did {e} work?",
    "religion": "What is the religion of {e}?",
    "mouth of the watercourse": "Where does the {e} river flow into?",
    "located on terrain feature": "On what landform or island is {e} located?",
    "licensed to broadcast to": "What area is {e} licensed to broadcast to?",
    "continent": "On which continent is {e}?",
    "named after": "Who or what is {e} named after?",
    "director": "Who directed {e}?",
    "followed by": "What came after {e}?",
    "follows": "What came before {e}?",
    "member of political party": "What political party is {e} a member of?",
    "screenwriter": "Who wrote the screenplay of {e}?",
    "part of the series": "What series is {e} part of?",
    "located in or next to body of water": "What body of water is {e} located on or next to?",
    "original language of film or TV show": "What is the original language of {e}?",
    "notable work": "What is a notable work of {e}?",
    "award received": "What award did {e} receive?",
    "publisher": "Who published {e}?",
    "participant in": "What event did {e} take part in?",
    "participant": "Who took part in {e}?",
    "founded by": "Who founded {e}?",
    "production company": "What production company made {e}?",
    "conflict": "What war or conflict was {e} involved in?",
    "movement": "What artistic or political movement is {e} associated with?",
    "date of birth": "When was {e} born?",
    "date of death": "When did {e} die?",
    "genre": "What genre is {e}?",
    "language of work or name": "What language is {e} in?",
    "position held": "What position did {e} hold?",
    "occupation": "What is the occupation of {e}?",
    "inception": "When was {e} founded?",
    "publication date": "When was {e} published?",
    "official language": "What is the official language of {e}?",
    "head of government": "Who is the head of government of {e}?",
    "head of state": "Who is the head of state of {e}?",
    "military branch": "What military branch did {e} serve in?",
    "original broadcaster": "What network originally aired {e}?",
    "developer": "Who developed {e}?",
    "distributed by": "Who distributed {e}?",
    "sport": "What sport does {e} play?",
    "league": "What league does {e} play in?",
    "crosses": "What does {e} cross?",
    "drafted by": "Which team drafted {e}?",
    "languages spoken, written or signed": "What language does {e} speak?",
    "main subject": "What is the main subject of {e}?",
}
# Lookups lowercase the relation, so normalize the keys the same way
# ("original language of film or TV show" never matched before).
RELATION_TEMPLATES = {k.lower(): v for k, v in RELATION_TEMPLATES.items()}

_LEADING = re.compile(r"^(?:(?:who|what|when|where|which|whom|whose|why|how many|how much|"
                      r"in what|in which|on what|at what)\s+)?(?:(?:is|was|are|were|did|does|"
                      r"do|has|have|had)\s+)?(?:the\s+)?", re.I)


def substitute(q: str, answers: list) -> str:
    def rep(m):
        k = int(m.group(1)) - 1
        return answers[k] if 0 <= k < len(answers) and answers[k] else m.group(0)
    q = re.sub(r"#(\d+)", rep, q)
    q = re.sub(r"\b(the|a|an)\s+(the|a|an)\b", r"\2", q, flags=re.I)  # "the the USSR"
    q = re.sub(r"\s+('s|\?|,)", r"\1", q)                             # "CBC 's", "X ?"
    return " ".join(q.split())


def hop_queries(raw_q: str, prev_answers: list) -> tuple:
    q = substitute(raw_q, prev_answers).strip()
    if ">>" in q:
        ent, _, rel = (s.strip() for s in q.partition(">>"))
        tmpl = RELATION_TEMPLATES.get(rel.lower())
        query = tmpl.format(e=ent) if tmpl else f"What is the {rel} of {ent}?"
        return query, f"{ent} {rel}", bool(tmpl)
    query = q[0].upper() + q[1:] if q else q
    if not query.endswith("?"):
        query += "?"
    para = _LEADING.sub("", q.rstrip("?")).strip()
    if not para or para.lower() == q.rstrip("?").lower() or len(para.split()) < 2:
        para = q.rstrip("?")
    return query, para, True


def build_entry(rec: dict, orig_idx, reserve: bool) -> dict:
    decomp = rec["question_decomposition"]
    gold = [h["answer"] for h in decomp]
    final_aliases = [rec["answer"]] + list(rec.get("answer_aliases") or [])
    hops, templated_ok = [], True
    for i, h in enumerate(decomp):
        query, para, known = hop_queries(h["question"], gold)
        templated_ok &= known
        sup = rec["paragraphs"][h["paragraph_support_idx"]] if h.get("paragraph_support_idx") is not None else None
        aliases = [h["answer"]] + (final_aliases[1:] if i == len(decomp) - 1 else [])
        hops.append({
            "hop_idx": i, "query": query, "paraphrase": para, "raw_question": h["question"],
            "answer": h["answer"], "aliases": aliases,
            "support_title": sup["title"] if sup else None,
            "answer_in_support": bool(sup) and contains_answer(
                sup["title"] + " " + sup["paragraph_text"], aliases),
        })
    return {
        "qid": rec["id"], "question": rec["question"], "answer": rec["answer"],
        "answer_aliases": final_aliases, "hop_type": rec["id"].split("__")[0],
        "reserve": reserve, "orig_idx": orig_idx, "all_relations_templated": templated_ok,
        "hops": hops,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--qid-map", required=True, type=Path,
                    help="results/adv_targeted_results/musique_qid_to_idx.json (paper's 100 qids)")
    ap.add_argument("--out", required=True, type=Path, help="queries_musique.json")
    ap.add_argument("--reserve", type=int, default=20, help="extra reserve questions (default 20)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extend-from", type=Path, default=None,
                    help="existing queries_musique.json to grow: keeps all its qids "
                         "(targets + reserves) so prior Keenable collection stays valid, "
                         "and samples only the extra reserves needed to reach --reserve")
    args = ap.parse_args()

    targets = read_json(args.qid_map)  # {qid: idx}
    raw = {r["id"]: r for r in read_jsonl(args.raw)}
    missing = [q for q in targets if q not in raw]
    if missing:
        sys.exit(f"{len(missing)} target qids not in {args.raw}: {missing[:5]}")

    if args.extend_from is not None:
        existing = read_json(args.extend_from)
        existing_qids = {e["qid"] for e in existing}
        n_existing_reserves = sum(1 for e in existing if e["reserve"])
        entries = list(existing)  # keep every already-built entry byte-identical
        pool = sorted(q for q, r in raw.items()
                      if q not in targets and q not in existing_qids and r.get("answerable", True))
        n_new = max(0, args.reserve - n_existing_reserves)
        rng = random.Random(args.seed)
        for q in rng.sample(pool, min(n_new, len(pool))):
            entries.append(build_entry(raw[q], None, True))
        print(f"extending {args.extend_from}: kept {len(existing)} existing entries "
              f"({n_existing_reserves} reserves), added {min(n_new, len(pool))} new reserves")
    else:
        entries = [build_entry(raw[q], idx, False) for q, idx in sorted(targets.items(), key=lambda kv: kv[1])]
        pool = sorted(q for q, r in raw.items() if q not in targets and r.get("answerable", True))
        rng = random.Random(args.seed)
        for q in rng.sample(pool, min(args.reserve, len(pool))):
            entries.append(build_entry(raw[q], None, True))
    write_json(args.out, entries)

    n_hops = sum(len(e["hops"]) for e in entries)
    bad = [(e["qid"], h["hop_idx"]) for e in entries for h in e["hops"] if not h["answer_in_support"]]
    untmpl = sorted({h["raw_question"].split(">>")[1].strip() for e in entries for h in e["hops"]
                     if ">>" in h["raw_question"]
                     and h["raw_question"].split(">>")[1].strip().lower() not in RELATION_TEMPLATES})
    n_search = sum(1 + 2 * len(e["hops"]) for e in entries)
    print(f"wrote {args.out}: {len(entries)} questions ({len(targets)} targets + "
          f"{len(entries) - len(targets)} reserves), {n_hops} hops, {n_search} searches planned")
    print(f"hop answer found in its support paragraph: {n_hops - len(bad)}/{n_hops}")
    if bad:
        print(f"  not found (check by hand): {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if untmpl:
        print(f"relations with no template (generic fallback used): {untmpl}")


if __name__ == "__main__":
    main()
