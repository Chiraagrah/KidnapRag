"""Monkeypatch-based exercise of 02_keenable_collect.py's logic, mirroring
how 02_serp_collect.py was verified (no real Keenable key in this env)."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("kc", str(__import__("pathlib").Path(__file__).resolve().parent / "02_keenable_collect.py"))
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)

# Tests must not pick up a real key from a .env next to the script.
kc.load_dotenv = lambda *a, **k: None
import os as _os
_os.environ.pop("KEENABLE_API_KEY", None)

CALLS = []

def fake_call(query, api_key, max_results, snippet_max_length, mode,
              exclude_domains, app_title, query_time, num_retries=5):
    CALLS.append(query)
    # Simulate a realistic mixed-domain result set, including leakage domains
    # that should be dropped both server-side (exclude_domains passed) and
    # by the client-side backstop.
    return {
        "query": query,
        "mode": mode,
        "results": [
            {"title": f"Result A for {query}", "url": "https://en.wikipedia.org/wiki/Example",
             "snippet": f"Snippet A relevant to {query}", "published_at": "2018-01-01T00:00:00Z",
             "acquired_at": "2026-11-01T00:00:00Z"},
            {"title": f"HF dataset viewer: {query}", "url": "https://huggingface.co/datasets/foo",
             "snippet": "leaked answer here", "published_at": None, "acquired_at": None},
            {"title": f"GitHub issue about {query}", "url": "https://github.com/foo/bar/issues/1",
             "snippet": "leaked answer here too", "published_at": None, "acquired_at": None},
            {"title": f"Result B for {query}", "url": "https://www.imdb.com/title/tt0460649/",
             "snippet": f"Snippet B relevant to {query}", "published_at": "2019-05-01T00:00:00Z",
             "acquired_at": "2026-11-01T00:00:00Z"},
        ],
        "usage": {"sku": "search.pro", "amount": 1, "credits": 0.0, "paid": False},
    }

kc.call_keenable_search = fake_call

import subprocess, shutil, tempfile, os

def run(tmpdir, extra_args):
    out_path = Path(tmpdir) / "searches.json"
    cache_dir = Path(tmpdir) / "keenable_raw"
    args = ["--input", "queries_sample.json", "--out", str(out_path),
            "--cache-dir", str(cache_dir)] + extra_args
    sys.argv = ["02_keenable_collect.py"] + args
    kc.main()
    return out_path, cache_dir


def test_leakage_filter_and_full_run():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        out_path, cache_dir = run(tmp, [])
        data = json.loads(out_path.read_text())
        assert len(data) == 17, f"expected 17 searches recorded, got {len(data)}"
        assert len(CALLS) == 17, f"expected 17 new calls, got {len(CALLS)}"
        for obj in data:
            urls = [r["url"] for r in obj["results"]]
            assert all("huggingface.co" not in u and "github.com" not in u for u in urls), \
                f"leakage domain survived in {obj['qid']}/{obj['search_type']}: {urls}"
            assert any("wikipedia.org" in u for u in urls), "expected clean result to survive"
            assert any("imdb.com" in u for u in urls), "expected clean result to survive"
        # published_at / acquired_at passed through
        sample = data[0]["results"][0]
        assert sample["published_at"] == "2018-01-01T00:00:00Z"
        assert sample["acquired_at"] == "2026-11-01T00:00:00Z"
        print("OK: leakage filter + full run (17/17 searches, 17 new calls, dates passed through)")


def test_cache_hit_on_rerun():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        run(tmp, [])
        n_calls_first = len(CALLS)
        CALLS.clear()
        # Rerun with the same output file present -> should skip everything via `done`,
        # making zero new calls (not even cache calls, since resume skips before fetch_search).
        out_path, cache_dir = run(tmp, [])
        assert len(CALLS) == 0, f"expected 0 new calls on full resume, got {len(CALLS)}"
        print(f"OK: resume skips all {n_calls_first} already-done searches, 0 new calls")


def test_cache_dir_hit_when_output_missing():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        out_path, cache_dir = run(tmp, [])
        out_path.unlink()  # simulate losing the output file but keeping the raw cache
        CALLS.clear()
        out_path2, cache_dir2 = run(tmp, [])
        assert len(CALLS) == 0, f"expected cache hits (0 new network calls), got {len(CALLS)}"
        data = json.loads(out_path2.read_text())
        assert len(data) == 17
        print("OK: raw cache survives even when --out is lost (0 new calls, output rebuilt)")


def test_max_calls_budget_guard():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        out_path, cache_dir = run(tmp, ["--max-calls", "5"])
        data = json.loads(out_path.read_text())
        assert len(data) == 5, f"expected exactly 5 searches done, got {len(data)}"
        assert len(CALLS) == 5
        # Resume: should pick up the remaining 12
        CALLS.clear()
        out_path2, _ = run(tmp, [])
        data2 = json.loads(out_path2.read_text())
        assert len(data2) == 17, f"expected all 17 after resume, got {len(data2)}"
        assert len(CALLS) == 12, f"expected 12 new calls to finish, got {len(CALLS)}"
        print("OK: --max-calls 5 stops early, resume finishes the remaining 12")


def test_no_leakage_filter_flag():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        out_path, _ = run(tmp, ["--no-leakage-filter", "--max-calls", "1"])
        data = json.loads(out_path.read_text())
        urls = [r["url"] for r in data[0]["results"]]
        assert any("huggingface.co" in u for u in urls), \
            "expected leakage domain to survive with --no-leakage-filter"
        print("OK: --no-leakage-filter lets leakage domains through (debug flag works)")


def test_exclude_domains_passed_when_not_disabled():
    seen_exclude = []
    def capturing_call(query, api_key, max_results, snippet_max_length, mode,
                        exclude_domains, app_title, query_time, num_retries=5):
        seen_exclude.append(exclude_domains)
        return fake_call(query, api_key, max_results, snippet_max_length, mode,
                          exclude_domains, app_title, query_time, num_retries)
    kc.call_keenable_search = capturing_call
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp, ["--max-calls", "1"])
        assert seen_exclude[0] == sorted(kc.LEAKAGE_DOMAINS), seen_exclude[0]
        print("OK: exclude_domains sent to Keenable by default (server-side filter active)")

        with tempfile.TemporaryDirectory() as tmp:
            seen_exclude.clear()
            run(tmp, ["--no-server-exclude", "--max-calls", "1"])
        assert seen_exclude[0] == [], seen_exclude[0]
        print("OK: --no-server-exclude sends an empty exclude_domains list")
    finally:
        kc.call_keenable_search = fake_call


def test_keyless_warning_and_header_logic():
    # app_title / X-Keenable-Title only matters inside the real call_keenable_search,
    # which we've replaced -- verify it's still threaded through correctly.
    with tempfile.TemporaryDirectory() as tmp:
        captured = {}
        def capture_title(query, api_key, max_results, snippet_max_length, mode,
                           exclude_domains, app_title, query_time, num_retries=5):
            captured["app_title"] = app_title
            captured["api_key"] = api_key
            return fake_call(query, api_key, max_results, snippet_max_length, mode,
                              exclude_domains, app_title, query_time, num_retries)
        kc.call_keenable_search = capture_title
        try:
            run(tmp, ["--app-title", "unit-test-app", "--max-calls", "1"])
            assert captured["app_title"] == "unit-test-app"
            assert captured["api_key"] is None  # KEENABLE_API_KEY not set in this test env
            print("OK: --app-title threaded through; keyless (api_key=None) when env var unset")
        finally:
            kc.call_keenable_search = fake_call


if __name__ == "__main__":
    test_leakage_filter_and_full_run()
    test_cache_hit_on_rerun()
    test_cache_dir_hit_when_output_missing()
    test_max_calls_budget_guard()
    test_no_leakage_filter_flag()
    test_exclude_domains_passed_when_not_disabled()
    test_keyless_warning_and_header_logic()
    print("\nALL TESTS PASSED")


def _failing_call(*a, **k):
    raise RuntimeError("Tunnel connection failed: 403 host is not on the allow list")


def test_failures_count_toward_budget_and_abort():
    kc.call_keenable_search = _failing_call
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_path, _ = run(tmp, ["--max-calls", "2", "--max-consecutive-failures", "99"])
            data = json.loads(out_path.read_text())
            assert len(data) == 2, f"--max-calls 2 should stop after 2 failed attempts, got {len(data)}"
        with tempfile.TemporaryDirectory() as tmp:
            out_path, _ = run(tmp, [])
            data = json.loads(out_path.read_text())
            assert len(data) == 3, f"should abort after 3 consecutive failures, got {len(data)}"
        print("OK: failed calls count toward --max-calls; aborts after 3 failures in a row")
    finally:
        kc.call_keenable_search = fake_call


def test_errored_searches_retried_on_resume():
    with tempfile.TemporaryDirectory() as tmp:
        kc.call_keenable_search = _failing_call
        run(tmp, [])  # 3 errored entries, then abort
        kc.call_keenable_search = fake_call
        CALLS.clear()
        out_path, _ = run(tmp, [])
        data = json.loads(out_path.read_text())
        assert len(data) == 17 and not any("error" in o for o in data), \
            "errored searches should be retried and replaced on resume"
        assert len(CALLS) == 17, f"expected 17 calls (3 retried + 14 new), got {len(CALLS)}"
        print("OK: errored searches are retried on resume, not skipped")


if __name__ == "__main__":
    test_failures_count_toward_budget_and_abort()
    test_errored_searches_retried_on_resume()
    print("NEW TESTS PASSED")
