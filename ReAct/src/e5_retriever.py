
import os
import glob
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from logzero import logger
try:
    from .models.encoder import SimpleEncoder
except (ImportError, ValueError):
    from models.encoder import SimpleEncoder

class E5_Retriever:
    """
    A standalone retriever class for E5/Contriever models using the custom SimpleEncoder.
    """
    def __init__(
        self,
        index_dir: str,
        corpus_path: str,
        poisoned_index_dir: Optional[str] = None,
        poisoned_corpus_path: Optional[str] = None,
        model_name: str = 'intfloat/e5-large-v2',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        self.device = device
        self.batch_size = batch_size
        
        print(f"Initializing E5_Retriever with model: {model_name}")
        self.encoder = SimpleEncoder(model_name, device, batch_size=batch_size)

        # 1. Load Data (Clean + Poisoned)
        clean_corpus = self._load_corpus(corpus_path)
        clean_embeddings = self._load_embeddings_from_dir(index_dir)
        
        # [CRITICAL] Truncate corpus if index is smaller (indexing might be incomplete)
        if clean_embeddings is not None and len(clean_corpus) > clean_embeddings.shape[0]:
            print(f"Warning: Clean corpus ({len(clean_corpus)}) is larger than clean index ({clean_embeddings.shape[0]}). Truncating corpus to match.")
            clean_corpus = clean_corpus[:clean_embeddings.shape[0]]
            
        self.num_clean_docs = len(clean_corpus)
        
        poison_corpus = self._load_corpus(poisoned_corpus_path) if poisoned_corpus_path else []
        poison_embeddings = self._load_embeddings_from_dir(poisoned_index_dir) if poisoned_index_dir else None

        # 2. Merge Corpora
        self.corpus = clean_corpus + poison_corpus
        print(f"Total Corpus: {len(self.corpus)} (Clean: {len(clean_corpus)}, Poisoned: {len(poison_corpus)})")

        # 3. Merge Embeddings
        all_embs = []
        if clean_embeddings is not None:
            all_embs.append(clean_embeddings)
        if poison_embeddings is not None:
            all_embs.append(poison_embeddings)
            
        if all_embs:
            self.all_embeddings = torch.cat(all_embs, dim=0).to(self.device)
            print(f"Total Index loaded. Shape: {self.all_embeddings.shape}")
            
            # Final sanity check
            if len(self.corpus) != self.all_embeddings.shape[0]:
                raise ValueError(f"CRITICAL: Corpus size ({len(self.corpus)}) does not match Index size ({self.all_embeddings.shape[0]})!")
        else:
            self.all_embeddings = None
            print("Warning: No embeddings loaded.")

    def _load_corpus(self, corpus_path: str) -> List[Dict]:
        corpus = []
        if not corpus_path: return corpus
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    corpus.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Corpus file not found at {corpus_path}.")
        return corpus

    def _load_embeddings_from_dir(self, index_dir: str) -> Optional[torch.Tensor]:
        print(f"Loading index shards from {index_dir}...")
        embeddings = []
        shard_files = sorted(glob.glob(os.path.join(index_dir, 'embedding-shard-*.pt')), 
                             key=lambda x: int(x.split('embedding-shard-')[-1].split('.pt')[0]) if '-' in x else 0)
        
        if not shard_files:
            return None
        
        for shard in shard_files:
            try:
                emb = torch.load(shard, map_location='cpu')
                embeddings.append(emb)
            except Exception as e:
                print(f"Error loading shard {shard}: {e}")

        if embeddings:
            return torch.cat(embeddings, dim=0) # Return on CPU for merging
        return None

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        # Delegate to the Clean SimpleEncoder
        return self.encoder.encode_queries(queries).to(self.device)

    # for oracle attack
    def add_corpus(self, corpus: List[Dict]):
        # embed new corpus
        if not corpus:
            return

        if self.all_embeddings is not None:
            new_embeddings = self.encoder.encode_corpus(corpus).to(self.all_embeddings.dtype).to(self.device)
            self.all_embeddings = torch.cat([self.all_embeddings, new_embeddings], dim=0)
        else:
            new_embeddings = self.encoder.encode_corpus(corpus).to(self.device)
            self.all_embeddings = new_embeddings
        
        self.corpus.extend(corpus)

        logger.info(f"Added {len(corpus)} new corpus to the retriever.")

    def search(self, query: str, k: int = 5, max_docs_per_docid: Optional[int] = 1,
               overfetch_factor: int = 20) -> List[Dict]:
        """
        Search for the query and return top-k documents.
        Returns a list of dicts: {'id': ..., 'title': ..., 'contents': ..., 'score': ...}

        max_docs_per_docid: cap on how many returned docs may share the same
        source-cluster `docid` (see dataset_builder/05_build_corpus.py). This
        stops mirror/duplicate copies of one source from filling the top-k and
        faking an "agreeing majority". Set to None to disable (legacy
        behavior: plain top-k, no dedup). Docs with no `docid` field (e.g. the
        original HotpotQA/MuSiQue/2WikiMultihopQA corpora) each count as their
        own unique cluster, so this is a no-op on those datasets.
        overfetch_factor: how many candidates beyond k to pull from the index
        before applying the per-docid cap, so capping doesn't starve the
        result count when top-ranked docs cluster heavily.
        """
        if self.all_embeddings is None:
            return []

        query_emb = self.encode_queries([query]) # [1, dim]

        # Ensure dtypes match (safety cast)
        if self.all_embeddings is not None:
            query_emb = query_emb.to(self.all_embeddings.dtype)

        # Dot product similarity
        scores = torch.matmul(query_emb, self.all_embeddings.T).squeeze(0) # [num_docs]

        # Pull extra candidates when dedup is on, so the per-docid cap has
        # room to skip duplicates without returning fewer than k docs.
        n_docs = scores.shape[0]
        fetch_k = k if max_docs_per_docid is None else min(n_docs, max(k * overfetch_factor, k))
        top_scores, top_indices = torch.topk(scores, k=fetch_k)

        results = []
        docid_counts: Dict[str, int] = {}
        for i in range(fetch_k):
            if len(results) >= k:
                break
            idx = top_indices[i].item()
            score = top_scores[i].item()

            doc = self.corpus[idx] if idx < len(self.corpus) else {}
            doc_id = doc.get('_id', doc.get('id', str(idx)))
            # Fall back to the doc's own id when it carries no explicit
            # `docid` (source-cluster id) field, so every doc is its own
            # cluster of one -- a no-op on corpora without that field.
            cluster_id = doc.get('docid', doc_id)

            if max_docs_per_docid is not None:
                seen = docid_counts.get(cluster_id, 0)
                if seen >= max_docs_per_docid:
                    continue
                docid_counts[cluster_id] = seen + 1

            results.append({
                'id': doc_id,
                'docid': cluster_id,
                'title': doc.get('title', ''),
                'contents': doc.get('text', doc.get('contents', '')),
                'score': score,
                'is_poisoned': idx >= self.num_clean_docs
            })

        return results

# Helper function for quick testing
if __name__ == "__main__":
    # Example usage
    retriever = E5_Retriever(
        index_dir="datasets/hotpotqa/test_index",
        corpus_path="datasets/hotpotqa/subset_corpus.jsonl"
    )
    results = retriever.search("What is the capital of France?", k=2)
    print(json.dumps(results, indent=2))
