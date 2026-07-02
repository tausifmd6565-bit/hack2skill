"""
evidence_graph.py
=================
Builds a structured evidence graph for each candidate.

For every key JD requirement, it checks:
  - Skills list
  - Projects / work history text
  - Technical domains
  - GitHub URL presence (as proxy for real GitHub signals)
  - Portfolio URL presence

Produces a per-requirement confidence score:
  0 = no evidence
  1 = weak evidence (keyword in skills only)
  2 = partial evidence (skills + domain or partial project text)
  3 = strong evidence (multiple corroborating signals)
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .candidate_profiler import CandidateProfile


# ── Requirement definitions ────────────────────────────────────────────────

@dataclass
class EvidenceEntry:
    requirement: str
    evidence_text: str
    sources: list[str]
    confidence: int  # 0-3
    confidence_label: str  # None / Weak / Partial / Strong


@dataclass
class CandidateEvidenceGraph:
    candidate_id: str
    entries: list[EvidenceEntry] = field(default_factory=list)
    total_evidence_score: float = 0.0     # normalised 0-100
    has_production_evidence: bool = False
    has_vector_search_evidence: bool = False
    has_evaluation_evidence: bool = False
    has_ranking_evidence: bool = False
    has_github: bool = False
    has_portfolio: bool = False
    startup_fit: float = 0.0


CONFIDENCE_LABELS = {0: "None", 1: "Weak", 2: "Partial", 3: "Strong"}

# Map requirement name → (required keywords, bonus keywords for strong signal)
REQUIREMENTS: list[tuple[str, list[str], list[str]]] = [
    (
        "Production ML Deployment",
        ["production", "deployed", "fastapi", "flask", "docker", "kubernetes", "rest api", "serving", "ci/cd", "microservice"],
        ["kubernetes", "docker", "ci/cd", "mlflow", "serving"],
    ),
    (
        "Vector Search / Embeddings",
        ["faiss", "vector search", "embedding", "embeddings", "qdrant", "pinecone", "weaviate", "milvus", "chroma", "sentence-transformer", "semantic search", "dense retrieval"],
        ["faiss", "qdrant", "semantic search", "sentence-transformer"],
    ),
    (
        "Hybrid Retrieval",
        ["hybrid", "bm25", "sparse", "dense", "hybrid retrieval", "hybrid search", "elasticsearch"],
        ["hybrid retrieval", "bm25", "sparse"],
    ),
    (
        "Ranking Systems",
        ["ranking", "rank", "rerank", "re-rank", "search ranking", "learning to rank", "ltr", "recommendation"],
        ["learning to rank", "re-rank", "ranking system"],
    ),
    (
        "Evaluation Metrics (NDCG / MRR)",
        ["ndcg", "mrr", "map", "recall", "precision@", "evaluation", "offline evaluation", "a/b", "benchmark"],
        ["ndcg", "mrr", "a/b"],
    ),
    (
        "Python Engineering",
        ["python", "pytorch", "tensorflow", "sklearn", "scikit-learn", "numpy", "fastapi", "flask"],
        ["python", "fastapi", "pytorch"],
    ),
    (
        "Retrieval / Search Experience",
        ["retrieval", "search", "elasticsearch", "solr", "information retrieval", "query", "index"],
        ["information retrieval", "elasticsearch"],
    ),
    (
        "Product / Startup Experience",
        ["product company", "startup", "series", "b2c", "consumer", "saas", "founding"],
        ["founding", "startup", "series a"],
    ),
]


class EvidenceGraphBuilder:

    def build(self, candidate: CandidateProfile) -> CandidateEvidenceGraph:
        graph = CandidateEvidenceGraph(
            candidate_id=candidate.candidate_id,
            has_github=bool(candidate.github_url),
            has_portfolio=bool(candidate.portfolio_url),
        )

        # Combine all text signals for search
        skills_text = " ".join(candidate.skills).lower()
        projects_text = " ".join(candidate.projects).lower()
        domains_text = " ".join(candidate.technical_domains).lower()
        full_text = f"{skills_text} {projects_text} {domains_text}"

        for req_name, base_kws, strong_kws in REQUIREMENTS:
            entry = self._check_requirement(
                req_name, base_kws, strong_kws,
                skills_text, projects_text, domains_text, full_text,
                candidate,
            )
            graph.entries.append(entry)

        # Set convenience flags
        graph.has_production_evidence = self._entry_confidence(graph, "Production ML Deployment") >= 2
        graph.has_vector_search_evidence = self._entry_confidence(graph, "Vector Search / Embeddings") >= 1
        graph.has_evaluation_evidence = self._entry_confidence(graph, "Evaluation Metrics (NDCG / MRR)") >= 1
        graph.has_ranking_evidence = self._entry_confidence(graph, "Ranking Systems") >= 1

        # Startup fit: product/startup company + founding/ownership language
        startup_signals = sum([
            candidate.company_type in ("product", "startup"),
            "startup" in full_text or "founding" in full_text,
            candidate.experience_years < 10,
            "ownership" in full_text or "fast" in full_text,
        ])
        graph.startup_fit = min(1.0, startup_signals / 4)

        # Total evidence score: weighted mean confidence (0-3 → 0-100)
        if graph.entries:
            total = sum(e.confidence for e in graph.entries)
            max_possible = len(graph.entries) * 3
            graph.total_evidence_score = round(total / max_possible * 100, 1)

            # Boost for external evidence
            if graph.has_github:
                graph.total_evidence_score = min(100, graph.total_evidence_score + 5)
            if graph.has_portfolio:
                graph.total_evidence_score = min(100, graph.total_evidence_score + 3)

        return graph

    # ── helpers ────────────────────────────────────────────────────────────

    def _check_requirement(
        self,
        req_name: str,
        base_kws: list[str],
        strong_kws: list[str],
        skills_text: str,
        projects_text: str,
        domains_text: str,
        full_text: str,
        candidate: CandidateProfile,
    ) -> EvidenceEntry:
        sources: list[str] = []
        hits_skills = sum(1 for kw in base_kws if kw in skills_text)
        hits_projects = sum(1 for kw in base_kws if kw in projects_text)
        hits_domains = sum(1 for kw in base_kws if kw in domains_text)
        strong_hits = sum(1 for kw in strong_kws if kw in full_text)

        if hits_skills > 0:
            sources.append("Skills")
        if hits_projects > 0:
            sources.append("Projects")
        if hits_domains > 0:
            sources.append("Domains")
        if candidate.github_url:
            sources.append("GitHub")
        if candidate.portfolio_url:
            sources.append("Portfolio")

        # Confidence scoring
        if hits_skills == 0 and hits_projects == 0 and hits_domains == 0:
            confidence = 0
            evidence_text = "No evidence found in profile, projects, or domains."
        elif strong_hits >= 2 and (hits_projects > 0 or hits_domains > 0):
            confidence = 3
            evidence_text = self._best_project_snippet(base_kws, candidate.projects) or f"Strong evidence across skills and project history."
        elif (hits_skills >= 2 or hits_projects >= 1) and strong_hits >= 1:
            confidence = 2
            evidence_text = self._best_project_snippet(base_kws, candidate.projects) or f"Partial evidence in skills and projects."
        else:
            confidence = 1
            evidence_text = f"Keyword present in skills list but no project-level evidence."

        return EvidenceEntry(
            requirement=req_name,
            evidence_text=evidence_text,
            sources=sources,
            confidence=confidence,
            confidence_label=CONFIDENCE_LABELS[confidence],
        )

    def _best_project_snippet(self, keywords: list[str], projects: list[str]) -> str:
        for project in projects:
            lp = project.lower()
            if any(kw in lp for kw in keywords):
                return project
        return ""

    def _entry_confidence(self, graph: CandidateEvidenceGraph, req_name: str) -> int:
        for entry in graph.entries:
            if entry.requirement == req_name:
                return entry.confidence
        return 0
