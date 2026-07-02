"""
scoring_engine.py
=================
Computes the final weighted candidate score.

Formula (from SignalCortex spec):
  0.16 × Semantic JD Fit        (hybrid retrieval score)
  0.14 × Must-Have Skill Evidence
  0.14 × Production ML Evidence
  0.12 × Retrieval/Ranking Experience
  0.10 × Evaluation Experience
  0.10 × External Evidence Score
  0.08 × Product/Startup Fit
  0.07 × Availability Score
  0.05 × Seniority/Location Fit
  0.04 × External Validation (GitHub + portfolio)
  – Risk Penalties (from risk_engine)
"""

from __future__ import annotations
from dataclasses import dataclass

from .candidate_profiler import CandidateProfile
from .evidence_graph import CandidateEvidenceGraph
from .missionfit_engine import MissionFitResult
from .jd_parser import RoleDNA


@dataclass
class CandidateScore:
    candidate_id: str
    semantic_jd_fit: float = 0.0
    must_have_skill_evidence: float = 0.0
    production_ml_score: float = 0.0
    retrieval_ranking_score: float = 0.0
    evaluation_score: float = 0.0
    external_evidence_score: float = 0.0
    product_startup_fit: float = 0.0
    availability_score: float = 0.0
    seniority_location_fit: float = 0.0
    external_validation: float = 0.0
    risk_penalty: float = 0.0
    missionfit_score: float = 0.0
    final_score: float = 0.0
    fit_label: str = ""
    confidence_score: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


FIT_LABELS = [
    (90, "Excellent Match"),
    (80, "Strong Match"),
    (70, "Good Match"),
    (60, "Possible Match"),
    (0, "Weak Match"),
]


def _fit_label(score: float) -> str:
    for threshold, label in FIT_LABELS:
        if score >= threshold:
            return label
    return "Weak Match"


def _confidence(score: CandidateScore, evidence: CandidateEvidenceGraph) -> float:
    """Confidence = how many signals actually back the score (not just estimated)."""
    signals = 0
    total = 6
    if evidence.has_production_evidence:
        signals += 1
    if evidence.has_vector_search_evidence:
        signals += 1
    if evidence.has_ranking_evidence:
        signals += 1
    if evidence.has_evaluation_evidence:
        signals += 1
    if evidence.has_github:
        signals += 1
    if evidence.has_portfolio:
        signals += 1
    base_conf = (signals / total) * 100
    # Adjust: if final score is very different from evidence, reduce confidence
    evidence_expected = evidence.total_evidence_score
    score_gap = abs(score.final_score - evidence_expected)
    conf = base_conf - (score_gap * 0.2)
    return round(max(20.0, min(100.0, conf)), 1)


class ScoringEngine:

    WEIGHTS = {
        "semantic_jd_fit": 0.16,
        "must_have_skill_evidence": 0.14,
        "production_ml_score": 0.14,
        "retrieval_ranking_score": 0.12,
        "evaluation_score": 0.10,
        "external_evidence_score": 0.10,
        "product_startup_fit": 0.08,
        "availability_score": 0.07,
        "seniority_location_fit": 0.05,
        "external_validation": 0.04,
    }

    def score(
        self,
        candidate: CandidateProfile,
        evidence: CandidateEvidenceGraph,
        missionfit: MissionFitResult,
        retrieval_scores: dict,
        role_dna: RoleDNA,
        availability_score: float,
        risk_penalty: float,
    ) -> CandidateScore:
        cs = CandidateScore(candidate_id=candidate.candidate_id)

        # 1. Semantic JD Fit — from hybrid retriever
        cs.semantic_jd_fit = retrieval_scores.get("hybrid_score", 0.0)

        # 2. Must-have skill evidence — count how many must-haves are present
        must_haves = role_dna.must_have
        if must_haves:
            present = sum(1 for v in must_haves.values() if v)
            # Compare against candidate text
            skills_text = " ".join(candidate.skills + candidate.projects + candidate.technical_domains).lower()
            matched = sum(
                1 for cat, kws in self._must_have_kws().items()
                if any(kw in skills_text for kw in kws)
            )
            cs.must_have_skill_evidence = min(100.0, matched / max(len(must_haves), 1) * 100)
        else:
            cs.must_have_skill_evidence = 50.0

        # 3. Production ML evidence
        cs.production_ml_score = 100.0 if evidence.has_production_evidence else (
            50.0 if candidate.experience_years >= 5 else 20.0
        )

        # 4. Retrieval/Ranking experience
        cs.retrieval_ranking_score = self._retrieval_score(candidate, evidence)

        # 5. Evaluation experience
        cs.evaluation_score = 100.0 if evidence.has_evaluation_evidence else 20.0

        # 6. External evidence score
        cs.external_evidence_score = evidence.total_evidence_score

        # 7. Product/startup fit
        cs.product_startup_fit = evidence.startup_fit * 100

        # 8. Availability score (passed in)
        cs.availability_score = availability_score

        # 9. Seniority + location fit
        cs.seniority_location_fit = self._seniority_fit(candidate, role_dna)

        # 10. External validation (GitHub + portfolio)
        cs.external_validation = self._external_validation(candidate, evidence)

        # ── MissionFit score
        cs.missionfit_score = missionfit.overall_missionfit

        # ── Risk penalty (already computed, negative value expected)
        cs.risk_penalty = risk_penalty

        # ── Final score
        raw = sum(
            getattr(cs, k) * w
            for k, w in self.WEIGHTS.items()
        )
        cs.final_score = round(max(0.0, min(100.0, raw + risk_penalty)), 1)
        cs.fit_label = _fit_label(cs.final_score)
        cs.confidence_score = _confidence(cs, evidence)

        return cs

    # ── helpers ────────────────────────────────────────────────────────────

    def _must_have_kws(self) -> dict[str, list[str]]:
        return {
            "production_ml": ["production", "deployed", "fastapi", "docker", "serving"],
            "retrieval_ranking": ["retrieval", "ranking", "bm25", "search", "elasticsearch"],
            "embeddings_vectors": ["faiss", "embedding", "vector", "sentence-transformer", "semantic"],
            "evaluation_metrics": ["ndcg", "mrr", "map", "evaluation", "a/b"],
            "python_engineering": ["python", "pytorch", "fastapi", "scikit-learn"],
        }

    def _retrieval_score(self, candidate: CandidateProfile, evidence: CandidateEvidenceGraph) -> float:
        score = 0.0
        if evidence.has_vector_search_evidence:
            score += 40
        if evidence.has_ranking_evidence:
            score += 40
        if any("retrieval" in d.lower() or "search" in d.lower() or "ranking" in d.lower()
               for d in candidate.technical_domains):
            score += 20
        return min(100.0, score)

    def _seniority_fit(self, candidate: CandidateProfile, role_dna: RoleDNA) -> float:
        lo, hi = role_dna.seniority_years
        exp = candidate.experience_years
        if lo <= exp <= hi:
            return 90.0
        if exp < lo:
            return max(20.0, 90.0 - (lo - exp) * 15)
        # overqualified — small penalty
        return max(60.0, 90.0 - (exp - hi) * 5)

    def _external_validation(self, candidate: CandidateProfile, evidence: CandidateEvidenceGraph) -> float:
        score = 0.0
        if evidence.has_github:
            score += 50
        if evidence.has_portfolio:
            score += 50
        return score
