"""
missionfit_engine.py
====================
Scores each candidate against the 4 JD-extracted 90-day missions.

For each mission, it checks the candidate's full text profile
against the mission's required_signals list and computes a 0-100 score.

Also produces:
  - overall MissionFit score (weighted average)
  - ramp_up_risk: low / medium / high
  - mission_labels: strong / medium / weak per mission
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .candidate_profiler import CandidateProfile
from .jd_parser import Mission, RoleDNA


@dataclass
class MissionScore:
    mission_number: int
    mission_title: str
    score: float          # 0-100
    label: str            # Strong / Medium / Weak
    matched_signals: list[str]
    missing_signals: list[str]


@dataclass
class MissionFitResult:
    candidate_id: str
    mission_scores: list[MissionScore] = field(default_factory=list)
    overall_missionfit: float = 0.0
    ramp_up_risk: str = "medium"   # low / medium / high


def _label(score: float) -> str:
    if score >= 75:
        return "Strong"
    if score >= 50:
        return "Medium"
    return "Weak"


def _ramp_risk(score: float) -> str:
    if score >= 75:
        return "low"
    if score >= 55:
        return "medium"
    return "high"


class MissionFitEngine:
    """
    Scores a candidate against each 90-day mission extracted from the JD.
    """

    def score(
        self,
        candidate: CandidateProfile,
        role_dna: RoleDNA,
    ) -> MissionFitResult:
        # Build candidate text corpus for signal matching
        full_text = " ".join([
            candidate.current_title,
            " ".join(candidate.skills),
            " ".join(candidate.projects),
            " ".join(candidate.technical_domains),
            candidate.current_company,
            candidate.company_type,
        ]).lower()

        mission_scores: list[MissionScore] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for mission in role_dna.missions:
            ms = self._score_mission(mission, full_text, candidate)
            mission_scores.append(ms)
            weighted_sum += ms.score * mission.weight
            total_weight += mission.weight

        overall = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

        return MissionFitResult(
            candidate_id=candidate.candidate_id,
            mission_scores=mission_scores,
            overall_missionfit=overall,
            ramp_up_risk=_ramp_risk(overall),
        )

    def _score_mission(
        self,
        mission: Mission,
        full_text: str,
        candidate: CandidateProfile,
    ) -> MissionScore:
        signals = mission.required_signals
        matched = [s for s in signals if s in full_text]
        missing = [s for s in signals if s not in full_text]

        # Base score from signal coverage
        coverage = len(matched) / len(signals) if signals else 0.0
        score = coverage * 80  # max 80 from signal matching

        # Bonus points for experience and company type
        if candidate.experience_years >= 5:
            score += 8
        if candidate.company_type in ("product", "startup") and mission.number == 4:
            score += 10
        if candidate.github_url and mission.number == 2:
            score += 5
        if candidate.portfolio_url:
            score += 3

        score = min(100.0, round(score, 1))

        return MissionScore(
            mission_number=mission.number,
            mission_title=mission.title,
            score=score,
            label=_label(score),
            matched_signals=matched,
            missing_signals=missing,
        )
