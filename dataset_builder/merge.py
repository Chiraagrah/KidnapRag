#!/usr/bin/env python3
"""Merge chunk_*.json into one searches.json and report any missing searches."""
import argparse
import json
from pathlib import Path

from driver import expected_keys_for_entry, load_queries  # reuse the same logic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--expected-input", required=True, type=Path, help="queries.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    merged = []
    seen = set()
    for chunk_file in sorted(args.results_dir.glob("chunk_*.json")):
        data = json.loads(chunk_file.read_text(encoding="utf-8"))
        for obj in data:
            key = (obj.get("qid"), obj.get("search_type"), obj.get("hop_idx"))
            if key in seen:
                continue  # dedup if a rerun double-wrote something
            seen.add(key)
            merged.append(obj)

    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(merged)} search records to {args.out}")

    queries = load_queries(args.expected_input)
    expected = set()
    for entry in queries:
        expected |= expected_keys_for_entry(entry)

    missing = expected - seen
    errored = {(o.get("qid"), o.get("search_type"), o.get("hop_idx"))
               for o in merged if o.get("error")}

    if missing:
        print(f"\n{len(missing)} expected searches are MISSING entirely:")
        for k in sorted(missing, key=lambda x: (str(x[0]), str(x[1]), x[2] or -1)):
            print(f"  qid={k[0]} type={k[1]} hop_idx={k[2]}")
    if errored:
        print(f"\n{len(errored)} searches are present but flagged with an error "
              f"(empty results, worth a manual rerun):")
        for k in sorted(errored, key=lambda x: (str(x[0]), str(x[1]), x[2] or -1)):
            print(f"  qid={k[0]} type={k[1]} hop_idx={k[2]}")
    if not missing and not errored:
        print("\nAll expected searches present and error-free.")


if __name__ == "__main__":
    main()
