"""
explainer.py
============
Generates human-readable recruiter explanations for each candidate.

Produces:
  - short_reason (1-2 sentences for list view)
  - long_explanation (full paragraph for detail view)
  - evidence_bullets (list of evidence snippets)
  - risk_summary (short string)
  - interview_questions (3 targeted questions based on gaps)
  - recommended_action
  - hidden_gem_reason (if applicable)
  - keyword_trap_reason (if applicable)
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .candidate_profiler import CandidateProfile
from .evidence_graph import CandidateEvidenceGraph
from .missionfit_engine import MissionFitResult
from .risk_engine import RiskResult
from .scoring_engine import CandidateScore


@dataclass
class CandidateExplanation:
    candidate_id: str
    short_reason: str = ""
    long_explanation: str = ""
    evidence_bullets: list[str] = field(default_factory=list)
    risk_summary: str = ""
    interview_questions: list[str] = field(default_factory=list)
    recommended_action: str = ""
    is_hidden_gem: bool = False
    hidden_gem_reason: str = ""
    is_keyword_trap: bool = False
    keyword_trap_reason: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


# ── Interview question templates ───────────────────────────────────────────

QUESTION_TEMPLATES = {
    "no_evaluation_evidence": (
        "How would you evaluate whether a new candidate-ranking system is better than the old BM25 baseline? "
        "What metrics would you track offline vs. online?"
    ),
    "no_production_evidence": (
        "Describe a machine learning system you shipped to production. "
        "What was the serving architecture, and how did you handle monitoring and rollback?"
    ),
    "no_vector_search": (
        "Walk us through how you would implement a hybrid BM25 + dense embedding retrieval system from scratch. "
        "What tradeoffs would you make for a 50K candidate corpus?"
    ),
    "no_ranking_evidence": (
        "Tell us about a ranking or recommendation system you built. "
        "How did you iterate on relevance quality, and what signals proved most valuable?"
    ),
    "service_company_background": (
        "In a product startup you own the full ranking stack — no specs, no PMs to hand things off. "
        "Tell us about a time you shipped an ML feature end-to-end without a detailed specification."
    ),
    "high_notice_period": (
        "Your notice period is listed as {notice} days. Is there flexibility on start date, "
        "or have you already begun the process of resigning?"
    ),
    "keyword_only_skills": (
        "I see you have listed RAG, LangChain, and Pinecone on your profile. "
        "Can you describe a production system you shipped using these tools — "
        "specifically how you evaluated its quality and handled failure cases?"
    ),
    "generic_ranking": (
        "How would you improve a BM25 + rule-based ranking system in 4 weeks, "
        "given you have access to click-through and apply data?"
    ),
}


class Explainer:

    def explain(
        self,
        candidate: CandidateProfile,
        score: CandidateScore,
        evidence: CandidateEvidenceGraph,
        missionfit: MissionFitResult,
        risk: RiskResult,
        retrieval_scores: dict,
        keyword_score_without_evidence: float,
    ) -> CandidateExplanation:
        exp = CandidateExplanation(candidate_id=candidate.candidate_id)

        strengths = self._collect_strengths(candidate, evidence, missionfit)
        weaknesses = self._collect_weaknesses(candidate, evidence, risk)

        exp.short_reason = self._short_reason(strengths, weaknesses, score)
        exp.long_explanation = self._long_explanation(candidate, score, evidence, missionfit, risk, strengths, weaknesses)
        exp.evidence_bullets = self._evidence_bullets(evidence)
        exp.risk_summary = self._risk_summary(risk)
        exp.interview_questions = self._interview_questions(candidate, evidence, risk)
        exp.recommended_action = self._recommended_action(score, risk, candidate)

        # Hidden gem: strong evidence but low keyword overlap
        if retrieval_scores.get("bm25_score", 0) < 40 and score.final_score >= 70:
            exp.is_hidden_gem = True
            exp.hidden_gem_reason = (
                f"Low keyword overlap ({retrieval_scores.get('bm25_score', 0):.0f}/100) but strong real-world evidence. "
                f"SignalCortex upranked due to production retrieval/ranking experience."
            )

        # Keyword trap: high keyword match but low evidence-adjusted score
        if keyword_score_without_evidence > 65 and score.final_score < 50:
            exp.is_keyword_trap = True
            exp.keyword_trap_reason = (
                "High AI keyword density (RAG, LangChain, vector DB) but evidence verification found no production deployments, "
                "no evaluation metrics, and no product engineering history. Risk penalties applied."
            )

        return exp

    # ── private builders ───────────────────────────────────────────────────

    def _collect_strengths(
        self,
        candidate: CandidateProfile,
        evidence: CandidateEvidenceGraph,
        missionfit: MissionFitResult,
    ) -> list[str]:
        s = []
        if evidence.has_production_evidence:
            s.append("production ML deployment evidence")
        if evidence.has_vector_search_evidence:
            s.append("vector search / embedding expertise")
        if evidence.has_ranking_evidence:
            s.append("ranking / retrieval system experience")
        if evidence.has_evaluation_evidence:
            s.append("ranking evaluation metrics (NDCG/MRR)")
        if evidence.has_github:
            s.append("active GitHub presence")
        if evidence.has_portfolio:
            s.append("portfolio with case studies")
        if candidate.company_type in ("product", "startup"):
            s.append("product company background")
        if candidate.experience_years >= 5:
            s.append(f"{candidate.experience_years} years of relevant experience")
        strong_missions = [m for m in missionfit.mission_scores if m.label == "Strong"]
        if len(strong_missions) >= 3:
            s.append("strong across all 4 first-90-day missions")
        return s

    def _collect_weaknesses(
        self,
        candidate: CandidateProfile,
        evidence: CandidateEvidenceGraph,
        risk: RiskResult,
    ) -> list[str]:
        w = []
        if not evidence.has_production_evidence:
            w.append("no production ML deployment evidence")
        if not evidence.has_evaluation_evidence:
            w.append("no ranking evaluation experience")
        if not evidence.has_vector_search_evidence:
            w.append("no vector search evidence")
        if candidate.company_type == "service":
            w.append("consulting-only career history")
        if candidate.company_type == "research":
            w.append("research-only background")
        for flag in risk.flags:
            if flag.severity == "high" and flag.flag_type not in ("no_evaluation_evidence",):
                w.append(flag.reason.split(".")[0].lower())
        return w

    def _short_reason(self, strengths: list[str], weaknesses: list[str], score: CandidateScore) -> str:
        if strengths:
            top = strengths[:2]
            reason = "Strong candidate: " + " and ".join(top) + "."
        else:
            reason = "Candidate shows some relevant experience."
        if weaknesses:
            reason += f" Gap: {weaknesses[0]}."
        return reason

    def _long_explanation(
        self,
        candidate: CandidateProfile,
        score: CandidateScore,
        evidence: CandidateEvidenceGraph,
        missionfit: MissionFitResult,
        risk: RiskResult,
        strengths: list[str],
        weaknesses: list[str],
    ) -> str:
        lines = [
            f"{candidate.name} ({candidate.current_title}, {candidate.experience_years} yrs) "
            f"received a final SignalCortex score of {score.final_score}/100 ({score.fit_label}).",
        ]
        if strengths:
            lines.append("Key strengths: " + "; ".join(strengths) + ".")
        lines.append(
            f"MissionFit score: {missionfit.overall_missionfit}/100 (ramp-up risk: {missionfit.ramp_up_risk})."
        )
        if weaknesses:
            lines.append("Areas to verify: " + "; ".join(weaknesses[:3]) + ".")
        if risk.flags:
            penalty_desc = ", ".join(f.flag_type.replace("_", " ") for f in risk.flags[:2])
            lines.append(f"Risk penalty applied ({abs(risk.total_penalty)} pts): {penalty_desc}.")
        return " ".join(lines)

    def _evidence_bullets(self, evidence: CandidateEvidenceGraph) -> list[str]:
        bullets = []
        for entry in evidence.entries:
            if entry.confidence >= 2:
                src = ", ".join(entry.sources) if entry.sources else "Profile"
                bullets.append(f"[{entry.confidence_label}] {entry.requirement}: {entry.evidence_text} (Source: {src})")
        return bullets or ["No strong evidence found."]

    def _risk_summary(self, risk: RiskResult) -> str:
        if not risk.flags:
            return "No significant risk flags detected."
        high = [f.reason for f in risk.flags if f.severity == "high"]
        med = [f.reason for f in risk.flags if f.severity == "medium"]
        parts = []
        if high:
            parts.append("High: " + high[0])
        if med:
            parts.append("Medium: " + med[0])
        return " | ".join(parts)

    def _interview_questions(
        self,
        candidate: CandidateProfile,
        evidence: CandidateEvidenceGraph,
        risk: RiskResult,
    ) -> list[str]:
        questions = []
        flag_types = {f.flag_type for f in risk.flags}

        if not evidence.has_evaluation_evidence:
            questions.append(QUESTION_TEMPLATES["no_evaluation_evidence"])
        if not evidence.has_production_evidence:
            questions.append(QUESTION_TEMPLATES["no_production_evidence"])
        if not evidence.has_vector_search_evidence:
            questions.append(QUESTION_TEMPLATES["no_vector_search"])
        if not evidence.has_ranking_evidence:
            questions.append(QUESTION_TEMPLATES["no_ranking_evidence"])
        if "consulting_only" in flag_types or candidate.company_type == "service":
            questions.append(QUESTION_TEMPLATES["service_company_background"])
        if "keyword_stuffing" in flag_types:
            questions.append(QUESTION_TEMPLATES["keyword_only_skills"])
        if candidate.notice_period_days >= 45:
            questions.append(
                QUESTION_TEMPLATES["high_notice_period"].format(notice=candidate.notice_period_days)
            )

        # Always add a generic good question if we have fewer than 3
        if len(questions) < 3:
            questions.append(QUESTION_TEMPLATES["generic_ranking"])

        return questions[:4]  # max 4 questions

    def _recommended_action(
        self,
        score: CandidateScore,
        risk: RiskResult,
        candidate: CandidateProfile,
    ) -> str:
        if score.final_score >= 85 and risk.risk_level == "low":
            return "Contact immediately — top shortlist."
        if score.final_score >= 75:
            return "Shortlist — schedule screening call."
        if score.final_score >= 65:
            return "Consider — verify key gaps before outreach."
        if score.final_score >= 50:
            return "Hold — significant gaps; deprioritise for this role."
        return "Reject — does not meet core requirements."
