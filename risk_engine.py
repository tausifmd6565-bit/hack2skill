"""
risk_engine.py
==============
Computes risk penalty and risk flags for each candidate.

Penalty values (negative):
  Only keyword stuffing (many weak_skills, no production evidence): -15
  Only LangChain/demo projects detected: -12
  Pure research background (research company + no production):     -18
  Consulting-only background:                                       -10
  Inactive for 6+ months:                                           -12
  Low recruiter response rate (<0.3):                               -10
  No recent production coding evidence:                             -15
  Title hopping pattern (exp < 3 but many companies):               -8
  No evaluation evidence:                                           -7
  No product-company experience:                                     -6
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .candidate_profiler import CandidateProfile
from .evidence_graph import CandidateEvidenceGraph


WEAK_SKILL_KEYWORDS = [
    "langchain", "openai api", "gpt-4", "llama", "llamaindex", "langraph",
    "autogpt", "langsmith", "pinecone", "chromadb", "weaviate api",
]
DEMO_PROJECT_KEYWORDS = [
    "langchain", "rag chatbot", "chatbot", "openai", "gpt-4 powered",
    "pdf summarisation", "document qa", "llm agent",
]


@dataclass
class RiskFlag:
    flag_type: str
    reason: str
    penalty: float
    severity: str   # low / medium / high


@dataclass
class RiskResult:
    candidate_id: str
    flags: list[RiskFlag] = field(default_factory=list)
    total_penalty: float = 0.0
    risk_level: str = "low"   # low / medium / high

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "risk_level": self.risk_level,
            "total_penalty": self.total_penalty,
            "flags": [
                {
                    "flag_type": f.flag_type,
                    "reason": f.reason,
                    "penalty": f.penalty,
                    "severity": f.severity,
                }
                for f in self.flags
            ],
        }


def _risk_level(penalty: float) -> str:
    if abs(penalty) <= 10:
        return "low"
    if abs(penalty) <= 25:
        return "medium"
    return "high"


class RiskEngine:

    def assess(
        self,
        candidate: CandidateProfile,
        evidence: CandidateEvidenceGraph,
    ) -> RiskResult:
        flags: list[RiskFlag] = []
        skills_text = " ".join(candidate.skills).lower()
        projects_text = " ".join(candidate.projects).lower()

        # ── Keyword stuffing ───────────────────────────────────────────────
        weak_count = sum(1 for kw in WEAK_SKILL_KEYWORDS if kw in skills_text)
        if weak_count >= 4 and not evidence.has_production_evidence:
            flags.append(RiskFlag(
                flag_type="keyword_stuffing",
                reason=f"Profile lists {weak_count} AI buzzwords (LangChain, GPT-4, Pinecone) without production deployment evidence.",
                penalty=-15,
                severity="high",
            ))

        # ── Only demo/tutorial projects ────────────────────────────────────
        demo_count = sum(1 for kw in DEMO_PROJECT_KEYWORDS if kw in projects_text)
        strong_project_signals = ["ndcg", "production", "deployed", "api", "faiss", "hybrid", "bm25", "evaluation"]
        strong_count = sum(1 for s in strong_project_signals if s in projects_text)
        if demo_count >= 2 and strong_count == 0:
            flags.append(RiskFlag(
                flag_type="demo_projects_only",
                reason="Projects appear to be tutorial or demo implementations (LangChain, RAG chatbots) with no production system evidence.",
                penalty=-12,
                severity="high",
            ))

        # ── Pure research background ───────────────────────────────────────
        if candidate.company_type == "research" and not evidence.has_production_evidence:
            flags.append(RiskFlag(
                flag_type="pure_research",
                reason="Career history is primarily academic/research with no evidence of deploying ML systems to production.",
                penalty=-18,
                severity="high",
            ))

        # ── Consulting-only ────────────────────────────────────────────────
        if candidate.company_type == "service":
            flags.append(RiskFlag(
                flag_type="consulting_only",
                reason="Candidate's career appears to be entirely in service/consulting companies. Limited product ownership evidence.",
                penalty=-10,
                severity="medium",
            ))

        # ── Inactive profile ───────────────────────────────────────────────
        if candidate.last_login_days_ago > 180:
            flags.append(RiskFlag(
                flag_type="inactive_profile",
                reason=f"Profile last active {candidate.last_login_days_ago} days ago (>180 days). Candidate may not be actively looking.",
                penalty=-12,
                severity="high",
            ))
        elif candidate.last_login_days_ago > 90:
            flags.append(RiskFlag(
                flag_type="low_activity",
                reason=f"Profile last active {candidate.last_login_days_ago} days ago. Moderate inactivity.",
                penalty=-5,
                severity="medium",
            ))

        # ── Low recruiter response rate ────────────────────────────────────
        if candidate.recruiter_response_rate < 0.3:
            flags.append(RiskFlag(
                flag_type="low_response_rate",
                reason=f"Recruiter response rate is {candidate.recruiter_response_rate:.0%}. Candidate is unlikely to respond to outreach.",
                penalty=-10,
                severity="medium",
            ))

        # ── No evaluation evidence ─────────────────────────────────────────
        if not evidence.has_evaluation_evidence:
            flags.append(RiskFlag(
                flag_type="no_evaluation_evidence",
                reason="No evidence of using ranking evaluation metrics (NDCG, MRR, A/B testing) in any project or role.",
                penalty=-7,
                severity="medium",
            ))

        # ── No product company experience ──────────────────────────────────
        if candidate.company_type not in ("product", "startup"):
            flags.append(RiskFlag(
                flag_type="no_product_experience",
                reason="No product or startup company experience detected. May lack product engineering mindset.",
                penalty=-6,
                severity="low",
            ))

        # ── High notice period ─────────────────────────────────────────────
        if candidate.notice_period_days >= 60:
            flags.append(RiskFlag(
                flag_type="high_notice_period",
                reason=f"Notice period is {candidate.notice_period_days} days. May delay onboarding for urgent founding team role.",
                penalty=-3,
                severity="low",
            ))

        total = sum(f.penalty for f in flags)
        return RiskResult(
            candidate_id=candidate.candidate_id,
            flags=flags,
            total_penalty=round(total, 1),
            risk_level=_risk_level(total),
        )
