"""
hybrid_retriever.py
===================
Two-stage retrieval:
  Stage 1: Fast BM25 keyword retrieval → top-K candidates
  Stage 2: FAISS dense semantic re-ranking on those K candidates

Falls back gracefully if sentence-transformers or faiss are unavailable
(uses TF-IDF cosine similarity instead).
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .candidate_profiler import CandidateProfile


class HybridRetriever:
    """
    BM25 + FAISS hybrid retriever.

    Usage:
        retriever = HybridRetriever()
        retriever.index(candidates)
        results = retriever.retrieve(jd_text, top_k=100)
        # returns list of (candidate_id, bm25_score, dense_score, hybrid_score)
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, good quality
    ALPHA = 0.5  # weight: alpha*BM25 + (1-alpha)*dense

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._candidates: list[CandidateProfile] = []
        self._corpus: list[list[str]] = []
        self._corpus_raw: list[str] = []
        self._bm25: Optional[object] = None
        self._dense_index = None
        self._encoder = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._indexed = False

    def index(self, candidates: list[CandidateProfile]) -> None:
        self._candidates = candidates
        docs = []
        for c in candidates:
            text = " ".join([
                c.current_title,
                " ".join(c.skills),
                " ".join(c.projects),
                " ".join(c.technical_domains),
                c.current_company,
                c.company_type,
            ]).lower()
            docs.append(text)
        self._corpus_raw = docs

        # BM25
        if _BM25_AVAILABLE:
            tokenized = [d.split() for d in docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            # fallback: TF-IDF
            self._tfidf = TfidfVectorizer(max_features=10000)
            self._tfidf_matrix = self._tfidf.fit_transform(docs)

        # Dense index
        if _FAISS_AVAILABLE:
            self._encoder = SentenceTransformer(self.EMBEDDING_MODEL)
            embeddings = self._encoder.encode(docs, show_progress_bar=False, batch_size=64)
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            self._dense_index = faiss.IndexFlatIP(dim)
            self._dense_index.add(embeddings)

        self._indexed = True

    def retrieve(self, jd_text: str, top_k: int = 200) -> list[dict]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve().")

        jd_lower = jd_text.lower()
        n = len(self._candidates)
        top_k = min(top_k, n)

        # ── BM25 scores ────────────────────────────────────────────────────
        if _BM25_AVAILABLE and self._bm25 is not None:
            bm25_raw = np.array(self._bm25.get_scores(jd_lower.split()), dtype=float)
        else:
            jd_vec = self._tfidf.transform([jd_lower])
            bm25_raw = cosine_similarity(jd_vec, self._tfidf_matrix).flatten()

        bm25_norm = self._minmax(bm25_raw)

        # ── Dense scores ───────────────────────────────────────────────────
        if _FAISS_AVAILABLE and self._dense_index is not None:
            jd_emb = self._encoder.encode([jd_lower], show_progress_bar=False).astype("float32")
            faiss.normalize_L2(jd_emb)
            # Search all
            scores, indices = self._dense_index.search(jd_emb, n)
            dense_raw = np.zeros(n, dtype=float)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                dense_raw[idx] = float(score)
        else:
            dense_raw = bm25_raw.copy()  # degrade gracefully

        dense_norm = self._minmax(dense_raw)

        # ── Hybrid combination ─────────────────────────────────────────────
        hybrid = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

        # Sort and return top-K
        top_indices = np.argsort(hybrid)[::-1][:top_k]
        results = []
        for idx in top_indices:
            c = self._candidates[idx]
            results.append({
                "candidate_id": c.candidate_id,
                "bm25_score": round(float(bm25_norm[idx]) * 100, 2),
                "dense_score": round(float(dense_norm[idx]) * 100, 2),
                "hybrid_score": round(float(hybrid[idx]) * 100, 2),
            })
        return results

    def _minmax(self, arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
