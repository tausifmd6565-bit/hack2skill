"""
candidate_profiler.py
=====================
Normalises raw candidate rows from CSV into structured CandidateProfile objects.
Handles missing fields gracefully — system degrades but never crashes.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, date


# ── Data class ─────────────────────────────────────────────────────────────

@dataclass
class CandidateProfile:
    candidate_id: str
    name: str
    current_title: str
    experience_years: int
    skills: list[str]
    projects: list[str]
    technical_domains: list[str]
    current_company: str
    company_type: str          # product | service | research | startup
    location: str
    github_url: str
    portfolio_url: str
    # availability signals
    last_login_days_ago: int   # days since last login (-1 if unknown)
    profile_updated_days_ago: int
    recruiter_response_rate: float   # 0-1
    notice_period_days: int
    relocation_preference: str       # yes | no | maybe
    open_to_work: bool
    # ground truth (used for evaluation only, not scoring)
    _profile_type: str = "unknown"

    def to_dict(self) -> dict:
        return self.__dict__


# ── Profiler ───────────────────────────────────────────────────────────────

class CandidateProfiler:
    """Converts a raw dict (CSV row) into a CandidateProfile."""

    def profile(self, row: dict) -> CandidateProfile:
        return CandidateProfile(
            candidate_id=str(row.get("candidate_id", "UNKNOWN")),
            name=str(row.get("name", row.get("candidate_id", "Unknown"))),
            current_title=str(row.get("current_title", "")),
            experience_years=self._parse_int(row.get("experience_years"), default=0),
            skills=self._parse_list(row.get("skills", "")),
            projects=self._parse_list(row.get("projects", "")),
            technical_domains=self._parse_list(row.get("technical_domains", "")),
            current_company=str(row.get("current_company", "")),
            company_type=self._normalize_company_type(str(row.get("company_type", ""))),
            location=str(row.get("location", "")),
            github_url=str(row.get("github_url", "")),
            portfolio_url=str(row.get("portfolio_url", "")),
            last_login_days_ago=self._days_since(row.get("last_login")),
            profile_updated_days_ago=self._days_since(row.get("profile_updated_at")),
            recruiter_response_rate=self._parse_float(row.get("recruiter_response_rate"), default=0.5),
            notice_period_days=self._parse_int(row.get("notice_period"), default=30),
            relocation_preference=str(row.get("relocation_preference", "maybe")).lower(),
            open_to_work=self._parse_bool(row.get("open_to_work")),
            _profile_type=str(row.get("_profile_type", "unknown")),
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _parse_list(self, value) -> list[str]:
        if not value or str(value).strip() in ("", "nan", "None"):
            return []
        raw = str(value)
        # support pipe, comma, or semicolon separated
        items = re.split(r"[|,;]", raw)
        return [i.strip() for i in items if i.strip()]

    def _parse_int(self, value, default: int = 0) -> int:
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return default

    def _parse_float(self, value, default: float = 0.0) -> float:
        try:
            v = float(str(value))
            return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            return default

    def _parse_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        s = str(value).lower().strip()
        return s in ("true", "1", "yes", "y")

    def _days_since(self, date_str) -> int:
        """Returns days since date_str, or -1 if unparseable."""
        if not date_str or str(date_str).strip() in ("", "nan", "None"):
            return -1
        try:
            d = datetime.strptime(str(date_str).strip(), "%Y-%m-%d").date()
            return (date.today() - d).days
        except ValueError:
            return -1

    def _normalize_company_type(self, raw: str) -> str:
        raw = raw.lower().strip()
        if raw in ("product", "product company", "product-based"):
            return "product"
        if raw in ("startup",):
            return "startup"
        if raw in ("service", "service company", "services", "consulting"):
            return "service"
        if raw in ("research", "research lab", "academia"):
            return "research"
        return "unknown"
