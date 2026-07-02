"""
main.py — SignalCortex FastAPI Application
==========================================

Endpoints:
  POST /api/analyze          — Full pipeline: JD + candidates CSV → ranked results
  POST /api/parse-jd         — Parse JD text only → Role DNA
  GET  /api/results          — Return last pipeline results (cached in memory)
  GET  /api/candidate/{id}   — Detailed view for one candidate
  GET  /api/hidden-gems      — Candidates flagged as hidden gems
  GET  /api/keyword-traps    — Candidates flagged as keyword traps
  GET  /api/evaluation       — Evaluation report vs. baselines
  GET  /api/export/csv       — Download ranked_candidates.csv
  GET  /api/export/json      — Download explained_shortlist.json
  GET  /health               — Health check
"""

from __future__ import annotations
import io
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.jd_parser import JDParser
from src.candidate_profiler import CandidateProfiler
from src.evidence_graph import EvidenceGraphBuilder
from src.hybrid_retriever import HybridRetriever
from src.missionfit_engine import MissionFitEngine
from src.scoring_engine import ScoringEngine
from src.risk_engine import RiskEngine
from src.availability_engine import AvailabilityEngine
from src.explainer import Explainer
from src.evaluator import Evaluator
from src.export_utils import (
    export_ranked_csv,
    export_evidence_json,
    export_explained_json,
    build_ranked_row,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("signalcortex")

# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="SignalCortex API",
    description="Intelligent Candidate Discovery & Ranking — rank by evidence, not keywords.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state (single session, MVP) ──────────────────────────────────

_state: dict = {
    "role_dna": None,
    "ranked_rows": [],
    "eval_reports": [],
    "jd_text": "",
    "total_candidates": 0,
    "processing_time_s": 0.0,
}

# ── Pydantic models ────────────────────────────────────────────────────────

class JDTextRequest(BaseModel):
    jd_text: str


class AnalysisStats(BaseModel):
    total_candidates: int
    strong_matches: int
    hidden_gems: int
    keyword_traps: int
    average_confidence: float
    processing_time_s: float


# ── Pipeline ───────────────────────────────────────────────────────────────

def run_pipeline(jd_text: str, df: pd.DataFrame) -> list[dict]:
    import time
    t0 = time.time()

    log.info("Starting SignalCortex pipeline...")

    # Instantiate engines
    jd_parser = JDParser()
    profiler = CandidateProfiler()
    evidence_builder = EvidenceGraphBuilder()
    retriever = HybridRetriever(alpha=0.5)
    mission_engine = MissionFitEngine()
    scorer = ScoringEngine()
    risk_engine = RiskEngine()
    availability_engine = AvailabilityEngine()
    explainer = Explainer()

    # 1. Parse JD
    log.info("Parsing JD...")
    role_dna = jd_parser.parse(jd_text)
    _state["role_dna"] = role_dna.to_dict()
    _state["jd_text"] = jd_text

    # 2. Profile candidates
    log.info(f"Profiling {len(df)} candidates...")
    candidates = [profiler.profile(row) for row in df.to_dict("records")]
    _state["total_candidates"] = len(candidates)

    # 3. Build evidence graphs (before retrieval so we can use in scoring)
    log.info("Building evidence graphs...")
    evidence_map = {c.candidate_id: evidence_builder.build(c) for c in candidates}

    # 4. Hybrid retrieval (index all, retrieve top-K)
    log.info("Running hybrid retrieval (BM25 + FAISS)...")
    retriever.index(candidates)
    top_k = min(len(candidates), 200)
    retrieval_results = retriever.retrieve(jd_text, top_k=top_k)
    retrieval_map = {r["candidate_id"]: r for r in retrieval_results}

    # Build candidate lookup
    cand_map = {c.candidate_id: c for c in candidates}

    # 5. Score, assess risk, compute availability, explain for each retrieved candidate
    log.info("Scoring and ranking candidates...")
    ranked_rows: list[dict] = []

    # Use top-K from retrieval + ensure all candidates get at least a basic row
    retrieved_ids = {r["candidate_id"] for r in retrieval_results}

    for r in retrieval_results:
        cid = r["candidate_id"]
        candidate = cand_map[cid]
        evidence = evidence_map[cid]

        # MissionFit
        missionfit = mission_engine.score(candidate, role_dna)

        # Risk
        risk = risk_engine.assess(candidate, evidence)

        # Availability
        avail = availability_engine.score(candidate)

        # Score
        score = scorer.score(
            candidate=candidate,
            evidence=evidence,
            missionfit=missionfit,
            retrieval_scores=r,
            role_dna=role_dna,
            availability_score=avail,
            risk_penalty=risk.total_penalty,
        )

        # Explain
        explanation = explainer.explain(
            candidate=candidate,
            score=score,
            evidence=evidence,
            missionfit=missionfit,
            risk=risk,
            retrieval_scores=r,
            keyword_score_without_evidence=r.get("bm25_score", 0),
        )

        row = build_ranked_row(
            rank=0,  # assigned after sorting
            candidate=candidate,
            score=score,
            evidence=evidence,
            missionfit=missionfit,
            risk=risk,
            availability=avail,
            explanation=explanation,
        )
        # Carry ground-truth for evaluation
        row["_profile_type"] = candidate._profile_type
        row["bm25_score"] = r.get("bm25_score", 0)
        row["dense_score"] = r.get("dense_score", 0)
        ranked_rows.append(row)

    # 6. Sort by final_score descending
    ranked_rows.sort(key=lambda x: x["final_score"], reverse=True)
    for i, row in enumerate(ranked_rows):
        row["rank"] = i + 1

    # 7. Export files
    log.info("Exporting outputs...")
    export_ranked_csv(ranked_rows)
    export_explained_json([
        {k: v for k, v in row.items() if k not in ("_profile_type",)}
        for row in ranked_rows
    ])
    export_evidence_json([
        evidence_map[r["candidate_id"]].entries.__repr__()
        for r in ranked_rows[:50]  # top 50 evidence graphs
    ])

    # 8. Evaluation
    log.info("Running evaluation vs. baselines...")
    evaluator = Evaluator()
    eval_reports = evaluator.run(ranked_rows)
    _state["eval_reports"] = [e.to_dict() for e in eval_reports]

    t1 = time.time()
    _state["processing_time_s"] = round(t1 - t0, 2)
    _state["ranked_rows"] = ranked_rows
    log.info(f"Pipeline complete in {_state['processing_time_s']}s. {len(ranked_rows)} candidates ranked.")
    return ranked_rows


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "SignalCortex", "version": "1.0.0"}


@app.post("/api/parse-jd")
def parse_jd(req: JDTextRequest):
    """Parse a JD and return Role DNA — no candidate data needed."""
    parser = JDParser()
    dna = parser.parse(req.jd_text)
    return dna.to_dict()


@app.post("/api/analyze")
async def analyze(
    jd_text: str = Form(...),
    candidates_file: UploadFile = File(...),
):
    """
    Main analysis endpoint.
    Accepts JD as form text and candidates as a CSV file upload.
    Returns ranked candidate list.
    """
    # Read uploaded CSV
    content = await candidates_file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Candidate CSV is empty.")

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="JD text is required.")

    ranked_rows = run_pipeline(jd_text.strip(), df)

    # Return summary + top 50 for API response (full CSV available via /export)
    summary = _build_summary()
    top50 = [_serialize_row(r) for r in ranked_rows[:50]]
    return {
        "summary": summary,
        "role_dna": _state["role_dna"],
        "results": top50,
        "evaluation": _state["eval_reports"],
    }


@app.get("/api/results")
def get_results(
    page: int = 1,
    page_size: int = 20,
    fit_label: Optional[str] = None,
    min_score: Optional[float] = None,
    risk_level: Optional[str] = None,
    hidden_gems_only: bool = False,
    keyword_traps_only: bool = False,
):
    """Paginated, filterable ranked results."""
    rows = _state["ranked_rows"]
    if not rows:
        return {"results": [], "total": 0, "page": page, "summary": None}

    # Apply filters
    if fit_label:
        rows = [r for r in rows if r.get("fit_label") == fit_label]
    if min_score is not None:
        rows = [r for r in rows if r.get("final_score", 0) >= min_score]
    if risk_level:
        rows = [r for r in rows if r.get("risk_level") == risk_level]
    if hidden_gems_only:
        rows = [r for r in rows if r.get("is_hidden_gem")]
    if keyword_traps_only:
        rows = [r for r in rows if r.get("is_keyword_trap")]

    total = len(rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = rows[start:end]

    return {
        "results": [_serialize_row(r) for r in page_rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "summary": _build_summary(),
    }


@app.get("/api/candidate/{candidate_id}")
def get_candidate(candidate_id: str):
    """Full detail for one candidate."""
    for row in _state["ranked_rows"]:
        if row["candidate_id"] == candidate_id:
            return _serialize_row(row, full=True)
    raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found.")


@app.get("/api/hidden-gems")
def get_hidden_gems():
    """All hidden gem candidates."""
    gems = [r for r in _state["ranked_rows"] if r.get("is_hidden_gem")]
    return {"count": len(gems), "hidden_gems": [_serialize_row(r, full=True) for r in gems]}


@app.get("/api/keyword-traps")
def get_keyword_traps():
    """All keyword trap candidates."""
    traps = [r for r in _state["ranked_rows"] if r.get("is_keyword_trap")]
    return {"count": len(traps), "keyword_traps": [_serialize_row(r, full=True) for r in traps]}


@app.get("/api/evaluation")
def get_evaluation():
    """Evaluation report: SignalCortex vs. keyword and embedding baselines."""
    if not _state["eval_reports"]:
        raise HTTPException(status_code=404, detail="No evaluation data. Run /api/analyze first.")
    return {
        "reports": _state["eval_reports"],
        "jd_text": _state["jd_text"][:300],
    }


@app.get("/api/role-dna")
def get_role_dna():
    """Return the parsed Role DNA for the current JD."""
    if not _state["role_dna"]:
        raise HTTPException(status_code=404, detail="No JD analyzed yet. Run /api/analyze first.")
    return _state["role_dna"]


@app.get("/api/export/csv")
def export_csv():
    """Download ranked_candidates.csv."""
    path = Path("outputs/ranked_candidates.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No CSV generated yet. Run /api/analyze first.")
    return FileResponse(
        str(path),
        media_type="text/csv",
        filename="ranked_candidates.csv",
    )


@app.get("/api/export/json")
def export_json():
    """Download explained_shortlist.json."""
    path = Path("outputs/explained_shortlist.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No JSON generated yet. Run /api/analyze first.")
    return FileResponse(
        str(path),
        media_type="application/json",
        filename="explained_shortlist.json",
    )


@app.get("/api/stats")
def get_stats():
    """Summary statistics for the current analysis."""
    return _build_summary()


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_summary() -> dict:
    rows = _state["ranked_rows"]
    if not rows:
        return {}
    strong = sum(1 for r in rows if r.get("final_score", 0) >= 80)
    gems = sum(1 for r in rows if r.get("is_hidden_gem"))
    traps = sum(1 for r in rows if r.get("is_keyword_trap"))
    avg_conf = round(sum(r.get("confidence_score", 0) for r in rows) / len(rows), 1) if rows else 0
    return {
        "total_candidates": _state["total_candidates"],
        "ranked_candidates": len(rows),
        "strong_matches": strong,
        "hidden_gems": gems,
        "keyword_traps": traps,
        "average_confidence": avg_conf,
        "processing_time_s": _state["processing_time_s"],
    }


def _serialize_row(row: dict, full: bool = False) -> dict:
    """Clean up non-serializable fields from a row dict."""
    exclude = {"_profile_type"} if not full else set()
    result = {}
    for k, v in row.items():
        if k in exclude:
            continue
        # Ensure JSON serializable
        if isinstance(v, float):
            result[k] = round(v, 2)
        elif isinstance(v, (list, dict, str, int, bool, type(None))):
            result[k] = v
        else:
            result[k] = str(v)
    return result


# ── Serve Next.js static export ────────────────────────────────────────────
# Single port: FastAPI serves both /api/* and the full React UI.
# Build first: cd frontend && npm run build  (outputs to frontend/out/)
# Run:         uvicorn main:app --port 8000

_STATIC_DIR = Path(__file__).parent / "frontend" / "out"

# HTML pages served explicitly (must come BEFORE the static mount)
_UI_PAGES = [
    ("",                "index.html"),
    ("upload",          "upload/index.html"),
    ("jd-intelligence", "jd-intelligence/index.html"),
    ("ranking",         "ranking/index.html"),
    ("candidate",       "candidate/index.html"),
    ("hidden-gems",     "hidden-gems/index.html"),
    ("keyword-traps",   "keyword-traps/index.html"),
    ("evaluation",      "evaluation/index.html"),
    ("exports",         "exports/index.html"),
    ("settings",        "settings/index.html"),
]

if _STATIC_DIR.exists():
    # Register each page route explicitly so API routes stay clean
    for _route, _html in _UI_PAGES:
        _html_path = str(_STATIC_DIR / _html)
        # use a closure to capture the path correctly
        def _make_handler(p: str):
            def _handler():
                return FileResponse(p)
            return _handler

        app.add_api_route(
            f"/{_route}" if _route else "/",
            _make_handler(_html_path),
            methods=["GET"],
            include_in_schema=False,
        )

    # Mount all static assets (_next/static, images, icons, etc.)
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="frontend")

else:
    @app.get("/", include_in_schema=False)
    def root():
        return {"message": "SignalCortex API. Run: cd frontend && npm run build"}


# ── Dev runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
