"""
ReliabilityRAG MIS defense, adapted to run as a *read-only diagnostic* on each
ReAct loop of the (clean / no-attack) KidnapRAG pipeline.

Reference: Shen, Imana, Wu, Xiang, Mittal, Korolova, "ReliabilityRAG: Effective
and Provably Robust Defense for RAG-based Web-Search" (NeurIPS 2025),
Algorithm 1 (ordinal-reliability setting):

    1. Isolated answering : for each retrieved doc x_i, ask the LLM the query
       using ONLY x_i  ->  isolated answer y_i.
    2. Contradiction test : an NLI model scores every answer pair (y_i, y_j);
       an edge is drawn iff P(contradiction) >= beta  (beta = 0.5).
    3. Rank-aware MIS     : find the Maximum Independent Set on this
       contradiction graph; on ties prefer the set with the smallest
       lexicographic index tuple (i.e. higher-ranked / more-reliable docs).

In the KidnapRAG ReAct pipeline the retriever returns k=5 rank-ordered docs per
`Search[...]` sub-query, so each ReAct loop yields its own 5-node contradiction
graph. We do NOT change what the agent observes; we only measure how internally
(in)consistent each loop's retrieval is. The "weakness" of a loop is read off
its graph (small MIS / dense contradictions => weak, easy-to-hijack step).

The pure graph/MIS/metric helpers below have no heavy dependencies and are unit
-testable on CPU. torch/transformers are imported lazily inside the analyzer so
this module can be imported (and its math tested) without a GPU.
"""

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Pure graph / MIS / metric helpers (no torch; unit-testable)
# --------------------------------------------------------------------------- #
def _is_independent(members: Sequence[int], edges: set) -> bool:
    """True if no pair of `members` is connected by an edge."""
    for a, b in combinations(sorted(members), 2):
        if (a, b) in edges:
            return False
    return True


def find_maximum_independent_sets(n: int, edges: set) -> List[Tuple[int, ...]]:
    """Return every Maximum Independent Set of an n-node graph by brute force.

    Edges are stored as a set of (i, j) tuples with i < j. n is tiny (<= ~20 in
    RAG), so enumerating 2**n subsets is fine. Each returned set is a tuple of
    node indices sorted ascending.
    """
    if n <= 0:
        return [()]
    best_size = -1
    best: List[Tuple[int, ...]] = []
    for mask in range(1 << n):
        members = tuple(i for i in range(n) if (mask >> i) & 1)
        if not _is_independent(members, edges):
            continue
        size = len(members)
        if size > best_size:
            best_size = size
            best = [members]
        elif size == best_size:
            best.append(members)
    return best


def rank_aware_mis(n: int, edges: set) -> Tuple[int, ...]:
    """Pick the rank-aware MIS: maximum size, then smallest lexicographic index
    tuple (lower indices = higher-ranked = more reliable docs, per the paper)."""
    all_mis = find_maximum_independent_sets(n, edges)
    # every member tuple is already sorted ascending, so plain min() on the
    # tuples gives the smallest lexicographic order.
    return min(all_mis) if all_mis else ()


def mis_metrics(n: int, edges: set, selected: Sequence[int],
                poisoned_flags: Optional[Sequence[bool]] = None) -> Dict:
    """Composite per-loop weakness metrics derived from the contradiction graph.

    Logs several complementary signals (the user asked to keep them all):
      * mis_size / mis_ratio        : size of the consistent majority
      * excluded_fraction           : (n - |MIS|) / n   -- PRIMARY weakness
      * edge_count / edge_density   : total pairwise disagreement
      * num_max_mis                 : # of tied maximum sets (ambiguity)
      * weakness                    : 1 - mis_ratio (higher = weaker loop)
    """
    edge_count = len(edges)
    max_edges = n * (n - 1) // 2
    edge_density = (edge_count / max_edges) if max_edges > 0 else 0.0
    mis_size = len(selected)
    mis_ratio = (mis_size / n) if n > 0 else 0.0
    excluded_fraction = ((n - mis_size) / n) if n > 0 else 0.0
    num_max_mis = len(find_maximum_independent_sets(n, edges))

    out = {
        "n_docs": n,
        "mis_size": mis_size,
        "mis_ratio": round(mis_ratio, 4),
        "excluded_fraction": round(excluded_fraction, 4),
        "edge_count": edge_count,
        "max_edges": max_edges,
        "edge_density": round(edge_density, 4),
        "num_max_mis": num_max_mis,
        "weakness": round(1.0 - mis_ratio, 4),  # primary scalar: bigger = weaker
    }
    if poisoned_flags is not None:
        out["n_poisoned"] = int(sum(1 for f in poisoned_flags if f))
        out["poisoned_survived_mis"] = int(
            sum(1 for i in selected if i < len(poisoned_flags) and poisoned_flags[i])
        )
    return out


# --------------------------------------------------------------------------- #
# NLI-backed analyzer (lazy torch/transformers import)
# --------------------------------------------------------------------------- #
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

ISOLATED_ANSWER_TEMPLATE = (
    "You are given a single reference document. Answer the question using ONLY "
    "the information in this document, in a few words. If the document does not "
    "contain the answer, respond exactly with 'unknown'.\n\n"
    "Document:\n{doc}\n\n"
    "Question: {query}\nAnswer:"
)


class ReliabilityMIS:
    """Runs ReliabilityRAG Algorithm 1 over one set of retrieved docs."""

    def __init__(self, nli_model_name: str = DEFAULT_NLI_MODEL,
                 nli_device: str = "cuda:0", beta: float = 0.5,
                 doc_char_limit: int = 2000, answer_char_limit: int = 400,
                 unknown_contradicts: bool = True):
        import torch  # lazy
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        self._torch = torch
        self.beta = beta
        # When True, a doc whose isolated answer is "unknown" (it cannot answer
        # the sub-query, i.e. it is irrelevant) contradicts every other doc, so
        # irrelevant docs are excluded from the consistent set and raise the
        # loop's weakness. Set False to recover ReliabilityRAG's neutral
        # treatment of non-answers.
        self.unknown_contradicts = unknown_contradicts
        self.nli_device = nli_device
        self.doc_char_limit = doc_char_limit
        self.answer_char_limit = answer_char_limit

        print(f"[ReliabilityMIS] loading NLI model {nli_model_name} on {nli_device}")
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model.to(nli_device)
        self.model.eval()

        # locate the CONTRADICTION logit index robustly from the model config
        label2id = {str(k).lower(): v for k, v in self.model.config.label2id.items()}
        self.contra_idx = label2id.get("contradiction")
        if self.contra_idx is None:  # fall back to id2label scan
            for idx, name in self.model.config.id2label.items():
                if str(name).lower() == "contradiction":
                    self.contra_idx = int(idx)
                    break
        if self.contra_idx is None:
            raise ValueError(
                f"Could not find CONTRADICTION label in {self.model.config.id2label}"
            )

    # -- step 1: isolated answering ---------------------------------------- #
    def isolated_answer(self, generate_fn: Callable, query: str, doc_text: str) -> str:
        prompt = ISOLATED_ANSWER_TEMPLATE.format(
            doc=(doc_text or "")[: self.doc_char_limit], query=query
        )
        ans = generate_fn(prompt)
        return (ans or "").strip()

    # strong "no information" phrases. We match 'unknown' only when the answer
    # LEADS with it, so a real answer that merely mentions the word (e.g.
    # "Ortakoy ... unknown") is not misread as a non-answer; the other phrases
    # are matched anywhere, as they only appear in refusals.
    _REFUSAL_PATTERNS = (
        "i don't know", "i do not know", "not mentioned", "not specified",
        "no information", "not provided", "not stated", "cannot be determined",
        "does not contain", "does not mention", "no answer", "not available",
    )

    def _is_unknown(self, answer: str) -> bool:
        """True if the isolated answer is empty or a 'no information' response."""
        a = (answer or "").strip().lower().strip(" .\"'`:")
        if not a:
            return True
        if a.startswith("unknown"):
            return True
        return any(p in a for p in self._REFUSAL_PATTERNS)

    # -- step 2: pairwise contradiction ------------------------------------ #
    def _contra_prob(self, premise: str, hypothesis: str) -> float:
        torch = self._torch
        enc = self.tokenizer(premise, hypothesis, truncation=True, max_length=512,
                             return_tensors="pt").to(self.nli_device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[self.contra_idx].item())

    def pairwise_contradiction(self, answers: List[str]) -> List[List[float]]:
        """Symmetric contradiction-probability matrix (max over both directions)."""
        n = len(answers)
        mat = [[0.0] * n for _ in range(n)]
        for i, j in combinations(range(n), 2):
            p = max(self._contra_prob(answers[i], answers[j]),
                    self._contra_prob(answers[j], answers[i]))
            mat[i][j] = mat[j][i] = p
        return mat

    # -- full Algorithm 1 over one loop's retrieved docs ------------------- #
    def analyze(self, generate_fn: Callable, query: str, docs: List[Dict]) -> Dict:
        """docs: ranked list of {'title','contents','score','is_poisoned'} (rank 0 = top)."""
        n = len(docs)
        if n == 0:
            return {"query": query, "n_docs": 0, "skipped": "no_docs"}

        answers = [self.isolated_answer(generate_fn, query, d.get("contents", ""))
                   for d in docs]
        unknown_flags = [self._is_unknown(a) for a in answers]
        mat = self.pairwise_contradiction(answers)

        # A contradiction edge is drawn when the NLI model says the two answers
        # disagree (prob >= beta) OR -- when unknown_contradicts is on -- when
        # either endpoint is "unknown" (an irrelevant doc). Forcing unknown to
        # contradict everything (incl. other unknowns) keeps irrelevant docs out
        # of the consistent majority. The raw NLI prob is still stored in
        # contra_matrix; each edge is tagged reason="nli"/"unknown".
        edges = set()
        edge_list = []
        for i, j in combinations(range(n), 2):
            nli_edge = mat[i][j] >= self.beta
            unk_edge = self.unknown_contradicts and (unknown_flags[i] or unknown_flags[j])
            if nli_edge or unk_edge:
                edges.add((i, j))
                edge_list.append({
                    "i": i, "j": j,
                    "contra_prob": round(mat[i][j], 4),
                    "reason": "nli" if nli_edge else "unknown",
                })

        # explicit matrices so the graph is readable without any plotting:
        #   adjacency      -> binary contradiction edges (0 on diagonal)
        #   contra_matrix  -> full NLI contradiction probabilities (incl. sub-beta)
        adjacency = [[1 if (min(i, j), max(i, j)) in edges else 0
                      for j in range(n)] for i in range(n)]
        contra_matrix = [[round(mat[i][j], 4) if i != j else 0.0
                          for j in range(n)] for i in range(n)]

        selected = list(rank_aware_mis(n, edges))
        poisoned_flags = [bool(d.get("is_poisoned", False)) for d in docs]
        metrics = mis_metrics(n, edges, selected, poisoned_flags)

        nodes = []
        for rank, d in enumerate(docs):
            nodes.append({
                "rank": rank,
                "title": d.get("title", ""),
                "score": round(float(d.get("score", 0.0)), 4),
                "is_poisoned": bool(d.get("is_poisoned", False)),
                "isolated_answer": answers[rank][: self.answer_char_limit],
                "unknown": bool(unknown_flags[rank]),
                "in_mis": rank in selected,
            })

        return {
            "query": query,
            "nodes": nodes,
            "edges": edge_list,
            "adjacency": adjacency,
            "contra_matrix": contra_matrix,
            "n_unknown": int(sum(unknown_flags)),
            "unknown_contradicts": self.unknown_contradicts,
            "mis_selected": selected,
            "all_max_mis": [list(s) for s in find_maximum_independent_sets(n, edges)],
            "metrics": metrics,
        }
