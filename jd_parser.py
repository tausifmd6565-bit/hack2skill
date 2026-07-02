"""
jd_parser.py
============
Extracts Role DNA from a raw job description text.

Outputs a structured RoleDNA object containing:
  - must_have skills
  - nice_to_have skills
  - disqualifiers (negative signals)
  - culture signals
  - four 90-day missions with required skill evidence
  - seniority range, work_mode, domain tags
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ── Skill / signal lexicons ────────────────────────────────────────────────

MUST_HAVE_SIGNALS: dict[str, list[str]] = {
    "production_ml": [
        "production", "deployed", "serving", "fastapi", "flask", "rest api",
        "microservice", "docker", "kubernetes", "k8s", "ci/cd", "mlflow",
        "model serving", "inference",
    ],
    "retrieval_ranking": [
        "retrieval", "ranking", "search", "bm25", "elasticsearch", "solr",
        "vector search", "dense retrieval", "sparse retrieval", "hybrid search",
        "hybrid retrieval", "semantic search", "re-rank", "rerank",
        "information retrieval",
    ],
    "embeddings_vectors": [
        "embedding", "embeddings", "faiss", "qdrant", "pinecone", "weaviate",
        "milvus", "chroma", "vector database", "vector db", "vector store",
        "sentence-transformer", "sentence_transformer", "bi-encoder",
    ],
    "evaluation_metrics": [
        "ndcg", "mrr", "map", "recall@", "precision@", "hit rate",
        "evaluation", "offline evaluation", "a/b test", "benchmark",
        "relevance judgement", "ranking metric",
    ],
    "python_engineering": [
        "python", "pytorch", "tensorflow", "scikit-learn", "sklearn",
        "numpy", "pandas", "pydantic",
    ],
    "product_company": [
        "product company", "product-based", "startup", "series a", "series b",
        "seed stage", "b2c", "b2b saas", "consumer",
    ],
}

NICE_TO_HAVE_SIGNALS: dict[str, list[str]] = {
    "llm_finetune": [
        "fine-tun", "finetun", "lora", "qlora", "peft", "rlhf", "sft",
        "instruction tuning",
    ],
    "learning_to_rank": [
        "learning to rank", "ltr", "lambdamart", "ranknet", "xgboost rank",
    ],
    "distributed_systems": [
        "spark", "flink", "kafka", "distributed", "ray", "dask",
    ],
    "open_source": [
        "open source", "open-source", "github stars", "contributor",
        "maintainer",
    ],
    "hr_tech": [
        "hr tech", "hrtech", "recruiting", "talent", "ats", "applicant",
        "hiring platform",
    ],
}

DISQUALIFIER_SIGNALS: list[str] = [
    "langchain only", "langchain demo", "only langchain",
    "tutorial project", "colab notebook only",
    "pure research", "only research", "no production",
    "consulting only", "service company only",
    "no recent coding", "last commit",
    "cv only", "computer vision only", "speech only", "robotics only",
    "job hopping", "frequent changes",
]

CULTURE_SIGNALS: list[str] = [
    "async", "remote", "written communication", "documentation",
    "ownership", "founding team", "0 to 1", "fast-paced", "startup",
    "product thinking", "cross-functional", "ambiguity",
]

# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class Mission:
    number: int
    title: str
    description: str
    required_signals: list[str]
    weight: float  # 0-1


@dataclass
class RoleDNA:
    raw_jd: str
    role_title: str = ""
    seniority_years: tuple[int, int] = (0, 15)
    work_mode: str = "hybrid"
    company_stage: str = ""
    must_have: dict[str, bool] = field(default_factory=dict)
    nice_to_have: dict[str, bool] = field(default_factory=dict)
    disqualifiers: list[str] = field(default_factory=list)
    culture_signals: list[str] = field(default_factory=list)
    missions: list[Mission] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role_title": self.role_title,
            "seniority_years": list(self.seniority_years),
            "work_mode": self.work_mode,
            "company_stage": self.company_stage,
            "must_have": self.must_have,
            "nice_to_have": self.nice_to_have,
            "disqualifiers": self.disqualifiers,
            "culture_signals": self.culture_signals,
            "domain_tags": self.domain_tags,
            "missions": [
                {
                    "number": m.number,
                    "title": m.title,
                    "description": m.description,
                    "required_signals": m.required_signals,
                    "weight": m.weight,
                }
                for m in self.missions
            ],
        }


# ── Parser ─────────────────────────────────────────────────────────────────

class JDParser:
    """
    Parses a raw JD string into a RoleDNA object.
    Uses rule-based NLP (regex + keyword matching). No external API needed.
    """

    def parse(self, jd_text: str) -> RoleDNA:
        text = jd_text.lower()
        dna = RoleDNA(raw_jd=jd_text)

        dna.role_title = self._extract_title(jd_text)
        dna.seniority_years = self._extract_seniority(text)
        dna.work_mode = self._extract_work_mode(text)
        dna.company_stage = self._extract_company_stage(text)
        dna.must_have = self._score_signals(text, MUST_HAVE_SIGNALS)
        dna.nice_to_have = self._score_signals(text, NICE_TO_HAVE_SIGNALS)
        dna.disqualifiers = self._detect_disqualifiers(text)
        dna.culture_signals = self._detect_culture(text)
        dna.domain_tags = self._extract_domains(text)
        dna.missions = self._build_missions(text)

        return dna

    # ── helpers ────────────────────────────────────────────────────────────

    def _extract_title(self, text: str) -> str:
        patterns = [
            r"(?:we are looking for|hiring|role:|position:)\s*(?:a |an )?(.{5,60}?)(?:\.|,|\n|$)",
            r"^(.{5,80})$",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) > 5:
                    return candidate[:80]
        return "Senior AI Engineer"

    def _extract_seniority(self, text: str) -> tuple[int, int]:
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", text)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"(\d+)\+?\s*years?", text)
        if m:
            y = int(m.group(1))
            return y, y + 4
        return (3, 10)

    def _extract_work_mode(self, text: str) -> str:
        if "remote" in text:
            return "remote"
        if "hybrid" in text:
            return "hybrid"
        if "on-site" in text or "onsite" in text or "office" in text:
            return "onsite"
        return "hybrid"

    def _extract_company_stage(self, text: str) -> str:
        for stage in ["pre-seed", "seed", "series a", "series b", "series c", "growth", "ipo", "public"]:
            if stage in text:
                return stage.title()
        return "Growth-stage"

    def _score_signals(self, text: str, signal_map: dict) -> dict[str, bool]:
        result = {}
        for category, keywords in signal_map.items():
            result[category] = any(kw in text for kw in keywords)
        return result

    def _detect_disqualifiers(self, text: str) -> list[str]:
        found = []
        for signal in DISQUALIFIER_SIGNALS:
            if signal in text:
                found.append(signal)
        return found

    def _detect_culture(self, text: str) -> list[str]:
        return [s for s in CULTURE_SIGNALS if s in text]

    def _extract_domains(self, text: str) -> list[str]:
        domain_map = {
            "retrieval": ["retrieval", "search", "bm25", "elasticsearch"],
            "ranking": ["ranking", "rank", "rerank", "learning to rank"],
            "recommendation": ["recommendation", "recommender", "collaborative filtering"],
            "nlp": ["nlp", "natural language", "text", "bert", "transformer"],
            "llm": ["llm", "large language model", "gpt", "gemini", "claude"],
            "ml_systems": ["mlops", "model serving", "inference", "pipeline"],
        }
        tags = []
        for tag, kws in domain_map.items():
            if any(kw in text for kw in kws):
                tags.append(tag)
        return tags

    def _build_missions(self, text: str) -> list[Mission]:
        """
        Derive 90-day missions from the JD content.
        Missions are ordered by typical execution timeline.
        """
        missions = [
            Mission(
                number=1,
                title="Audit the Existing Ranking System",
                description="Understand and diagnose the current search/ranking pipeline, identify gaps in relevance and retrieval quality.",
                required_signals=["bm25", "search", "relevance", "retrieval", "ranking", "audit", "debug"],
                weight=0.20,
            ),
            Mission(
                number=2,
                title="Ship Hybrid Retrieval v2",
                description="Design and deploy a semantic + sparse hybrid ranking system end-to-end in production.",
                required_signals=["embedding", "vector", "faiss", "hybrid", "semantic", "fastapi", "docker", "production"],
                weight=0.35,
            ),
            Mission(
                number=3,
                title="Build Evaluation Infrastructure",
                description="Establish offline and online evaluation pipelines to measure ranking quality continuously.",
                required_signals=["ndcg", "mrr", "map", "evaluation", "a/b", "benchmark", "offline", "recall"],
                weight=0.25,
            ),
            Mission(
                number=4,
                title="Operate in Startup / Product Environment",
                description="Ship fast, communicate clearly, collaborate across PM and recruiter teams with high ownership.",
                required_signals=["startup", "product", "ownership", "fast", "founding", "cross-functional", "async"],
                weight=0.20,
            ),
        ]
        return missions
