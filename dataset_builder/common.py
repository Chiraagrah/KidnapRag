"""Shared helpers for the MH-Web dataset builder (plan doc
claude/multihop-web-dataset-plan.md). Pure stdlib so it runs on Wulver's
bare Python >= 3.10 and on Windows alike."""
import hashlib
import json
import re
import unicodedata
import urllib.parse
from pathlib import Path


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Text normalisation and answer matching (port of
# ReliabilityRAG/data/check_relevance.contains_answer, made robust to
# accents, punctuation and partial-word matches)
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def contains_answer(text: str, answers) -> bool:
    """Verbatim, case/accent/punctuation-insensitive, whole-word match of any
    answer string. Answers that normalise to nothing are ignored."""
    hay = f" {normalize(text)} "
    for a in answers:
        na = normalize(a)
        if na and f" {na} " in hay:
            return True
    return False


# --------------------------------------------------------------------------
# Snippet cleaning (pilot issue 4: markdown residue in Keenable snippets)
# --------------------------------------------------------------------------
# A markdown link target, allowing one level of nested parentheses, as in
# (./Character_(arts)) on wikiwand pages.
_URL = r"\((?:[^()\s]|\([^()\s]*\))*(?:\s+\"[^\"]*\")?\)"
_RE_IMAGE = re.compile(r"!\[[^\]]*\]" + _URL)
_CITE_BODY = r"\s*(?:\d{1,3}|[a-z]|note\s*\d{1,3}|nb\s*\d{1,3}|citation needed)\s*"
_RE_CITE = re.compile(r"\[\[" + _CITE_BODY + r"\]\]" + _URL + r"|\[\[" + _CITE_BODY + r"\]\]"
                      r"|\[" + _CITE_BODY + r"\](?!\()", re.I)
_RE_LINK = re.compile(r"\[([^\[\]]*)\]" + _URL)
# Leftover "](target)" whose opening "[" was cut off at the snippet start.
_RE_ORPHAN_TARGET = re.compile(r"\]" + _URL)
_RE_BARE_URL_PAREN = re.compile(r"\(\s*(?:https?:)?//[^)\s]*\)|\(\./[^)\s]*\)")
_RE_EMPH = re.compile(r"(\*{1,3}|_{2,3})(\S(?:.*?\S)?)\1")
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_RE_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.M)
_RE_LIST_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.M)


def clean_snippet(text: str) -> str:
    """Strip markdown links/images/citation markers/emphasis from a Keenable
    snippet, keeping the visible text. Leaves plain prose untouched."""
    t = text or ""
    t = _RE_IMAGE.sub("", t)
    t = _RE_CITE.sub("", t)
    for _ in range(3):  # nested links like [[x](...)](...) need several passes
        t = _RE_LINK.sub(r"\1", t)
    t = _RE_ORPHAN_TARGET.sub("", t)
    t = _RE_BARE_URL_PAREN.sub("", t)
    t = t.replace("[", "").replace("]", "")
    t = _RE_EMPH.sub(r"\2", t)
    t = re.sub(r"\*{2,}|_{3,}", "", t)  # unpaired bold markers left after cutting
    t = _RE_HEADING.sub("", t)
    t = _RE_TABLE_RULE.sub("", t)
    t = _RE_LIST_BULLET.sub("", t)
    t = t.replace("|", " ").replace("\\", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return t.strip()


# --------------------------------------------------------------------------
# URL canonicalisation (pilot issue 1: mirror sites)
# --------------------------------------------------------------------------
_TRACKING_PARAMS = re.compile(r"^(utm_|ref$|ref_$|fbclid$|gclid$|mc_|igshid$|si$)")


def _wiki_title(t: str) -> str:
    t = urllib.parse.unquote(t).replace(" ", "_").strip("_")
    return t[:1].upper() + t[1:] if t else t


def canonical_url(url: str) -> str:
    """Map known mirrors onto one canonical address and drop noise (scheme,
    www./m. prefixes, fragments, tracking params, trailing slashes).

    Wikipedia mirrors handled: wikipedia.com, wikipedia.org (no language),
    <lang>.m.wikipedia.org, wikiwand.com/<lang>/[articles/]<Title>."""
    try:
        p = urllib.parse.urlparse(url.strip())
    except Exception:
        return url
    host = (p.netloc or "").lower().split("@")[-1].split(":")[0]
    for pre in ("www.", "m.", "mobile."):
        if host.startswith(pre):
            host = host[len(pre):]
    path = urllib.parse.unquote(p.path or "/")

    # Wikipedia language mobile hosts: en.m.wikipedia.org -> en.wikipedia.org
    host = re.sub(r"^([a-z\-]+)\.m\.wikipedia\.org$", r"\1.wikipedia.org", host)
    if host in ("wikipedia.com", "wikipedia.org"):
        host = "en.wikipedia.org"
    if host == "wikiwand.com":
        m = re.match(r"^/([a-z\-]{2,12})/(?:articles/)?(.+)$", path)
        if m:
            host, path = f"{m.group(1)}.wikipedia.org", "/wiki/" + m.group(2)
    if host.endswith(".wikipedia.org"):
        m = re.match(r"^/wiki/(.+)$", path)
        if m:
            path = "/wiki/" + _wiki_title(m.group(1))
        q = urllib.parse.parse_qs(p.query)
        if "title" in q:  # /w/index.php?title=X
            path = "/wiki/" + _wiki_title(q["title"][0])
        return f"{host}{path}"

    query = urllib.parse.urlencode(
        [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
         if not _TRACKING_PARAMS.match(k.lower())])
    path = path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{query}" if query else "")


def host_of(url: str) -> str:
    return canonical_url(url).split("/", 1)[0]


# --------------------------------------------------------------------------
# Near-duplicate detection (pilot issue 1: syndicated copies)
# --------------------------------------------------------------------------
def shingles(text: str, n: int = 5) -> set:
    toks = normalize(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {hash(" ".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


def overlap_coefficient(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------
# Question-text leakage (pilot issue 3): a result that quotes the question
# itself, e.g. a quiz or dataset-viewer page on a host not in exclude_domains.
# Uses contiguous word trigrams, not bag-of-words, so a normal topical page
# that merely shares the question's keywords is not flagged.
# --------------------------------------------------------------------------
def question_trigram_overlap(question: str, text: str) -> float:
    q = normalize(question).split()
    if len(q) < 3:
        return 0.0
    qg = {" ".join(q[i:i + 3]) for i in range(len(q) - 2)}
    t = normalize(text).split()
    tg = {" ".join(t[i:i + 3]) for i in range(len(t) - 2)}
    return len(qg & tg) / len(qg)
