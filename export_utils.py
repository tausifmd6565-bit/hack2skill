"""
export_utils.py
===============
Exports ranked results to CSV and JSON.
"""

from __future__ import annotations
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def export_ranked_csv(rows: list[dict], filename: str = "ranked_candidates.csv") -> str:
    """Write full ranked output to CSV. Returns file path."""
    out_path = str(OUTPUT_DIR / filename)

    if not rows:
        return out_path

    fieldnames = [
        "rank", "candidate_id", "name", "current_title", "experience_years",
        "location", "company_type", "final_score", "fit_label",
        "missionfit_score", "production_ml_score", "retrieval_ranking_score",
        "evaluation_score", "external_evidence_score", "availability_score",
        "risk_penalty", "confidence_score", "risk_level", "notice_period_days",
        "has_github", "has_portfolio", "is_hidden_gem", "is_keyword_trap",
        "short_reason", "risk_summary", "recommended_action",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return out_path


def export_evidence_json(evidence_list: list[dict], filename: str = "evidence_report.json") -> str:
    """Write evidence graphs to JSON."""
    out_path = str(OUTPUT_DIR / filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence_list, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def export_explained_json(explanations: list[dict], filename: str = "explained_shortlist.json") -> str:
    """Write explanation objects to JSON."""
    out_path = str(OUTPUT_DIR / filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def build_ranked_row(
    rank: int,
    candidate,
    score,
    evidence,
    missionfit,
    risk,
    availability: float,
    explanation,
) -> dict:
    """Merge all engine outputs into a single flat dict for CSV/API response."""
    return {
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "name": candidate.name,
        "current_title": candidate.current_title,
        "experience_years": candidate.experience_years,
        "location": candidate.location,
        "company_type": candidate.company_type,
        "final_score": score.final_score,
        "fit_label": score.fit_label,
        "missionfit_score": missionfit.overall_missionfit,
        "semantic_jd_fit": score.semantic_jd_fit,
        "must_have_skill_evidence": score.must_have_skill_evidence,
        "production_ml_score": score.production_ml_score,
        "retrieval_ranking_score": score.retrieval_ranking_score,
        "evaluation_score": score.evaluation_score,
        "external_evidence_score": score.external_evidence_score,
        "product_startup_fit": score.product_startup_fit,
        "availability_score": availability,
        "seniority_location_fit": score.seniority_location_fit,
        "external_validation": score.external_validation,
        "risk_penalty": score.risk_penalty,
        "confidence_score": score.confidence_score,
        "risk_level": risk.risk_level,
        "notice_period_days": candidate.notice_period_days,
        "has_github": evidence.has_github,
        "has_portfolio": evidence.has_portfolio,
        "is_hidden_gem": explanation.is_hidden_gem,
        "is_keyword_trap": explanation.is_keyword_trap,
        "short_reason": explanation.short_reason,
        "hidden_gem_reason": explanation.hidden_gem_reason,
        "keyword_trap_reason": explanation.keyword_trap_reason,
        "risk_summary": explanation.risk_summary,
        "recommended_action": explanation.recommended_action,
        "evidence_bullets": explanation.evidence_bullets,
        "interview_questions": explanation.interview_questions,
        "mission_scores": [
            {
                "number": m.mission_number,
                "title": m.mission_title,
                "score": m.score,
                "label": m.label,
            }
            for m in missionfit.mission_scores
        ],
        "risk_flags": [
            {
                "type": f.flag_type,
                "reason": f.reason,
                "penalty": f.penalty,
                "severity": f.severity,
            }
            for f in risk.flags
        ],
    }
