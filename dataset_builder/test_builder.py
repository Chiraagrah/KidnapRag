#!/usr/bin/env python3
"""Offline tests for common.py, 01_build_hops.py, 04/05 (no network, no key).
Run: python3 test_builder.py"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (canonical_url, clean_snippet, contains_answer,  # noqa: E402
                    overlap_coefficient, question_trigram_overlap, shingles)


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), HERE / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_clean_snippet():
    raw = ("**Theodore Evelyn Mosby[[1]](./Ted_Mosby#cite_note-1)** is a [fictional character]"
           "(./Character_(arts)) and the [protagonist](./Protagonist) in[2] the show.")
    assert clean_snippet(raw) == "Theodore Evelyn Mosby is a fictional character and the protagonist in the show."
    assert clean_snippet("Artifact](/pmwiki/x.php/Main/A): After Lily") == "Artifact: After Lily"
    assert clean_snippet("[[Mosby]](/wiki/Ted) said") == "Mosby said"
    assert clean_snippet("![i](x.png)Born 1966 (USA).[citation needed]") == "Born 1966 (USA)."
    assert clean_snippet("Plain prose, (m. 2020–24).") == "Plain prose, (m. 2020–24)."
    print("OK clean_snippet")


def test_canonical_url():
    w = "en.wikipedia.org/wiki/Ted_Mosby"
    for u in ["https://wikiwand.com/en/articles/Ted_Mosby", "https://wikipedia.com/wiki/Ted_Mosby",
              "https://en.m.wikipedia.org/wiki/ted%20Mosby#History", "http://www.wikipedia.org/wiki/Ted_Mosby",
              "https://en.wikipedia.org/w/index.php?title=Ted_Mosby&oldid=5"]:
        assert canonical_url(u) == w, (u, canonical_url(u))
    assert canonical_url("https://www.imdb.com/title/tt1/?ref_=nv&utm_source=x") == "imdb.com/title/tt1"
    assert canonical_url("https://ex.com/a?id=3") == "ex.com/a?id=3"
    print("OK canonical_url")


def test_matching_and_overlap():
    assert contains_answer("Directed by Barbet Schroeder.", ["barbet schroeder"])
    assert contains_answer("Cahiers du cinéma critic", ["Cahiers du cinema"])
    assert not contains_answer("Tracyville", ["Tracy"])         # whole words only
    assert not contains_answer("anything", ["", "   "])
    q = "Who does the narrator on How I Met Your Mother end up with?"
    assert question_trigram_overlap(q, "Quiz: who does the narrator on how i met your mother end up with") > 0.9
    assert question_trigram_overlap(q, "How I Met Your Mother is a sitcom narrated by Bob Saget.") < 0.3
    a = shingles("Edward Davis Wood Jr. was an American filmmaker, actor, and author known for low budget films")
    b = shingles("Edward Davis Wood Jr. was an American filmmaker, actor, and author known for low budget films ...")
    assert overlap_coefficient(a, b) >= 0.8
    print("OK matching / overlap")


def test_hop_queries():
    m = load("01_build_hops.py")
    q, p, known = m.hop_queries("#1 >> place of birth", ["James Finch"])
    assert (q, p, known) == ("Where was James Finch born?", "James Finch place of birth", True)
    q, p, known = m.hop_queries("Foo >> invented relation", [])
    assert q == "What is the invented relation of Foo?" and not known
    q, _, _ = m.hop_queries("Who argued that the #1 had become imperialist?", ["the USSR"])
    assert q == "Who argued that the USSR had become imperialist?", q
    q, _, _ = m.hop_queries("How many of #1 's transmitters were updated", ["CBC"])
    assert q == "How many of CBC's transmitters were updated?", q
    print("OK hop queries")


def _fixture(tmp):
    queries = [{
        "qid": "q1", "question": "Who does the narrator of Show X end up with?", "answer": "Tracy",
        "answer_aliases": ["Tracy", "The Mother"], "hop_type": "2hop", "reserve": False, "orig_idx": 0,
        "hops": [{"hop_idx": 0, "query": "Who narrates Show X?", "paraphrase": "Show X narrator",
                  "answer": "Ted", "aliases": ["Ted"]},
                 {"hop_idx": 1, "query": "Who does Ted end up with?", "paraphrase": "Ted wife",
                  "answer": "Tracy", "aliases": ["Tracy", "The Mother"]}]}]
    long = " and this sentence pads the snippet so that it is long enough to be kept by the filter."
    res = lambda url, text: {"title": "t", "url": url, "snippet": text + long}  # noqa: E731
    searches = [
        {"qid": "q1", "search_type": "full_question", "hop_idx": None, "query_text": queries[0]["question"],
         "results": [res("https://quiz.com/1", "Who does the narrator of Show X end up with? Tracy"),
                     res("https://fq.com/a", "In the end Tracy marries Ted")]},
        {"qid": "q1", "search_type": "hop", "hop_idx": 0, "query_text": "Who narrates Show X?",
         "results": [res("https://en.wikipedia.org/wiki/Ted", "Ted is the narrator of Show X"),
                     res("https://wikiwand.com/en/articles/Ted", "Ted narrates the series Show X"),
                     res("https://mirror1.org/Ted", "Ted is the narrator of Show X"),
                     res("https://b.com/x", "The narrator is Ted, voiced by Bob"),
                     res("https://c.com/y", "Show X's storyteller Ted recounts the tale")]},
        {"qid": "q1", "search_type": "hop_paraphrase", "hop_idx": 0, "query_text": "Show X narrator",
         "results": [res("https://d.com/z", "Narration by Ted throughout Show X")]},
        {"qid": "q1", "search_type": "hop", "hop_idx": 1, "query_text": "Who does Ted end up with?",
         "results": [res("https://e.com/1", "Ted ends up with Tracy"),
                     res("https://f.com/2", "Ted marries The Mother in 2020"),
                     res("https://g.com/3", "Tracy and Ted wed")]},
        {"qid": "q1", "search_type": "hop_paraphrase", "hop_idx": 1, "query_text": "Ted wife",
         "results": [res("https://h.com/4", "His wife Tracy")]},
    ]
    raw = [{"id": "q1", "question_decomposition": [{"paragraph_support_idx": 0}, {"paragraph_support_idx": 1}],
            "paragraphs": [{"title": "Show X", "paragraph_text": "Narrated by Ted."},
                           {"title": "Tracy", "paragraph_text": "Tracy is the mother."}]}]
    tmp = Path(tmp)
    (tmp / "q.json").write_text(json.dumps(queries))
    (tmp / "s.json").write_text(json.dumps(searches))
    (tmp / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    (tmp / "adv.json").write_text(json.dumps({"q1": {"id": "q1", "question": "x", "correct answer": "Tracy",
                                                     "incorrect answer": "BAIT", "adv_texts": []}}))
    return tmp


def test_label_and_build():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = _fixture(tmp)
        lab, bld = load("04_label_relevance.py"), load("05_build_corpus.py")
        sys.argv = ["04", "--queries", str(tmp / "q.json"), "--searches", str(tmp / "s.json"),
                    "--out-dir", str(tmp / "lab")]
        lab.main()
        st = json.loads((tmp / "lab" / "relevance_stats.json").read_text())
        q = st["questions"][0]
        assert q["drops"].get("quotes_question") == 1, q["drops"]          # quiz page dropped
        h0 = q["hops"][0]
        # 6 hop-0 results, but wikipedia+wikiwand fold to one URL and mirror1 is a near-duplicate:
        assert h0["bearing_results_own"] == 6, h0
        assert h0["bearing_sources_own"] == 4, h0
        assert q["passes_G1"], q
        rows = [json.loads(l) for l in (tmp / "lab" / "labeled_docs.jsonl").read_text().splitlines()]
        wiki = {r["docid"] for r in rows if "Ted" in r["url"] and r["labels"]["hops"] == [0]}
        assert len(wiki) == 1, wiki

        sys.argv = ["05", "--queries", str(tmp / "q.json"), "--labeled-dir", str(tmp / "lab"),
                    "--raw", str(tmp / "raw.jsonl"), "--adv-targets", str(tmp / "adv.json"),
                    "--repo-root", str(tmp / "repo"), "--n", "1"]
        bld.main()
        out = tmp / "repo" / "ReAct" / "datasets" / "mhweb_musique"
        corpus = [json.loads(l) for l in (out / "corpus.jsonl").read_text().splitlines()]
        assert not any("fq.com" in (r["url"] or "") for r in corpus), "full-question results must be excluded"
        assert sum(r["source"] == "musique_paragraph" for r in corpus) == 2
        sup = [r["labels"]["q1"]["supporting"] for r in corpus if r["source"] == "musique_paragraph"]
        assert sup == [True, True]
        adv = json.loads((tmp / "repo/ReAct/results/adv_targeted_results/mhweb_musique.json").read_text())
        assert adv["q1"]["incorrect answer"] == "BAIT"

        sys.argv = sys.argv + ["--include-full-question", "--name", "mhweb_fq"]
        bld.main()
        corpus = (tmp / "repo/ReAct/datasets/mhweb_fq/corpus.jsonl").read_text()
        assert "fq.com" in corpus and "quiz.com" not in corpus
    print("OK label + build (dedup, quote filter, fq exclusion, paragraphs, attack targets)")


def test_selection_with_reserves():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = _fixture(tmp)
        q = json.loads((tmp / "q.json").read_text())
        bad = json.loads(json.dumps(q[0])) | {"qid": "q0", "orig_idx": 0}
        q[0]["reserve"], q[0]["orig_idx"] = True, None   # q1 becomes a reserve
        (tmp / "q.json").write_text(json.dumps([bad] + q))  # q0 has no searches -> incomplete
        raw = (tmp / "raw.jsonl").read_text()
        (tmp / "raw.jsonl").write_text(raw + raw.replace('"q1"', '"q0"'))
        lab, bld = load("04_label_relevance.py"), load("05_build_corpus.py")
        sys.argv = ["04", "--queries", str(tmp / "q.json"), "--searches", str(tmp / "s.json"),
                    "--out-dir", str(tmp / "lab")]
        lab.main()
        sys.argv = ["05", "--queries", str(tmp / "q.json"), "--labeled-dir", str(tmp / "lab"),
                    "--raw", str(tmp / "raw.jsonl"), "--adv-targets", str(tmp / "adv.json"),
                    "--repo-root", str(tmp / "repo"), "--n", "1"]
        bld.main()
        sel = (tmp / "repo/ReAct/datasets/mhweb_musique/selected_qids.txt").read_text()
        assert sel == "q1\treserve\n", sel
    print("OK failing target replaced by a passing reserve")


if __name__ == "__main__":
    test_clean_snippet()
    test_canonical_url()
    test_matching_and_overlap()
    test_hop_queries()
    test_label_and_build()
    test_selection_with_reserves()
    print("\nALL BUILDER TESTS PASSED")
