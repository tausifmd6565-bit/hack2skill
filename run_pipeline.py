"""
run_pipeline.py
===============
CLI runner — reads JD from data/job_description.txt and candidates from data/candidates.csv,
runs the full SignalCortex pipeline, and prints a summary + top-10 shortlist.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

# Add project root to path so `src` package resolves
sys.path.insert(0, str(Path(__file__).parent))

from main import run_pipeline


JD_PATH = Path("data/job_description.txt")
CANDIDATES_PATH = Path("data/candidates.csv")


def main() -> None:
    if not JD_PATH.exists():
        print(f"ERROR: {JD_PATH} not found. Create it or run from signalcortex/")
        sys.exit(1)
    if not CANDIDATES_PATH.exists():
        print(f"ERROR: {CANDIDATES_PATH} not found. Run: python data/generate_candidates.py")
        sys.exit(1)

    jd_text = JD_PATH.read_text(encoding="utf-8")
    df = pd.read_csv(CANDIDATES_PATH)

    print(f"\n{'='*60}")
    print("  SignalCortex — Intelligent Candidate Discovery")
    print(f"{'='*60}")
    print(f"  JD: {JD_PATH}")
    print(f"  Candidates: {len(df)} profiles")
    print(f"{'='*60}\n")

    ranked = run_pipeline(jd_text, df)

    # ── Print summary ──────────────────────────────────────────────────────
    strong = sum(1 for r in ranked if r["final_score"] >= 80)
    gems = sum(1 for r in ranked if r.get("is_hidden_gem"))
    traps = sum(1 for r in ranked if r.get("is_keyword_trap"))
    avg_conf = sum(r.get("confidence_score", 0) for r in ranked) / len(ranked) if ranked else 0

    print("-"*60)
    print("  PIPELINE SUMMARY")
    print("-"*60)
    print(f"  Total ranked:       {len(ranked)}")
    print(f"  Strong matches:     {strong}")
    print(f"  Hidden gems found:  {gems}")
    print(f"  Keyword traps:      {traps}")
    print(f"  Avg confidence:     {avg_conf:.1f}%")
    print("-"*60)
    print()

    # ── Print top 10 ───────────────────────────────────────────────────────
    print("  TOP 10 SHORTLIST")
    print("-"*60)
    header = f"{'Rank':<5} {'ID':<8} {'Score':<7} {'MF':<6} {'Risk':<8} {'Fit Label':<18} {'Action'}"
    print(f"  {header}")
    print("  " + "-"*70)
    for row in ranked[:10]:
        gem_tag = " [GEM]" if row.get("is_hidden_gem") else ""
        trap_tag = " [TRAP]" if row.get("is_keyword_trap") else ""
        line = (
            f"  {row['rank']:<5} {row['candidate_id']:<8} "
            f"{row['final_score']:<7.1f} {row['missionfit_score']:<6.1f} "
            f"{row['risk_level']:<8} {row['fit_label']:<18} "
            f"{row['recommended_action']}{gem_tag}{trap_tag}"
        )
        print(line)

    print(f"\n  Outputs written to:")
    print(f"    outputs/ranked_candidates.csv")
    print(f"    outputs/explained_shortlist.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
