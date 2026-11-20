#!/usr/bin/env python3
"""Writes a 3-question queries_sample.json for smoke-testing the pipeline
before running it on the full 100-question dataset."""
import json
from pathlib import Path

SAMPLE = [
    {
        "qid": "2hop__67660_81007",
        "question": "Who does the narrator on How I Met Your Mother end up with?",
        "hops": [
            {"hop_idx": 0, "query": "Who narrates How I Met Your Mother?",
             "paraphrase": "How I Met Your Mother narrator identity"},
            {"hop_idx": 1, "query": "Who does Ted Mosby end up with?",
             "paraphrase": "Ted Mosby's eventual wife"},
        ],
    },
    {
        "qid": "sample_2hop_b",
        "question": "Where does the founder of Les Films Du Losange work at?",
        "hops": [
            {"hop_idx": 0, "query": "Who founded Les Films Du Losange?",
             "paraphrase": "Les Films Du Losange founder name"},
            {"hop_idx": 1, "query": "Where does Barbet Schroeder work?",
             "paraphrase": "Barbet Schroeder current workplace"},
        ],
    },
    {
        "qid": "sample_3hop_c",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "hops": [
            {"hop_idx": 0, "query": "What nationality is Scott Derrickson?",
             "paraphrase": "Scott Derrickson nationality director"},
            {"hop_idx": 1, "query": "What nationality was Ed Wood?",
             "paraphrase": "Ed Wood director nationality"},
            {"hop_idx": 2, "query": "Scott Derrickson Ed Wood same nationality comparison",
             "paraphrase": "compare nationalities Scott Derrickson Ed Wood"},
        ],
    },
]

if __name__ == "__main__":
    out = Path(__file__).parent / "queries_sample.json"
    out.write_text(json.dumps(SAMPLE, indent=2), encoding="utf-8")
    n_searches = sum(1 + 2 * len(e["hops"]) for e in SAMPLE)
    print(f"wrote {out} ({len(SAMPLE)} questions, {n_searches} searches total)")
