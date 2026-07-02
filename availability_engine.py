"""
availability_engine.py
======================
Computes a 0-100 availability & intent score for each candidate.

Signals used:
  last_login_days_ago      — recency of platform activity
  profile_updated_days_ago — freshness of profile data
  recruiter_response_rate  — historical responsiveness
  notice_period_days       — how quickly they can start
  relocation_preference    — location flexibility
  open_to_work             — explicit signal
"""

from __future__ import annotations
from .candidate_profiler import CandidateProfile


class AvailabilityEngine:

    def score(self, candidate: CandidateProfile) -> float:
        total = 0.0

        # 1. Last login recency (max 30 pts)
        days = candidate.last_login_days_ago
        if days == -1:
            total += 10  # unknown — neutral
        elif days <= 7:
            total += 30
        elif days <= 30:
            total += 25
        elif days <= 60:
            total += 15
        elif days <= 90:
            total += 8
        elif days <= 180:
            total += 3
        else:
            total += 0

        # 2. Profile update recency (max 15 pts)
        updated = candidate.profile_updated_days_ago
        if updated == -1:
            total += 7
        elif updated <= 14:
            total += 15
        elif updated <= 30:
            total += 12
        elif updated <= 60:
            total += 7
        elif updated <= 90:
            total += 3
        else:
            total += 0

        # 3. Recruiter response rate (max 25 pts)
        total += candidate.recruiter_response_rate * 25

        # 4. Notice period (max 20 pts)
        np = candidate.notice_period_days
        if np == 0:
            total += 20
        elif np <= 15:
            total += 18
        elif np <= 30:
            total += 14
        elif np <= 45:
            total += 9
        elif np <= 60:
            total += 5
        else:
            total += 2

        # 5. Open-to-work signal (max 5 pts)
        if candidate.open_to_work:
            total += 5

        # 6. Relocation (max 5 pts)
        rel = candidate.relocation_preference
        if rel == "yes":
            total += 5
        elif rel == "maybe":
            total += 3
        else:
            total += 0

        return round(min(100.0, total), 1)
