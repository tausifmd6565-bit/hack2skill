"""
Generate a realistic synthetic candidate dataset for SignalCortex.
Produces data/candidates.csv with 500 rows.
"""
import random
import csv
import json
from datetime import datetime, timedelta

random.seed(42)

# ── vocabulary pools ────────────────────────────────────────────────────────
STRONG_SKILLS = [
    "Python", "FAISS", "sentence-transformers", "hybrid retrieval",
    "vector search", "BM25", "Elasticsearch", "ranking evaluation",
    "NDCG", "MRR", "MAP", "semantic search", "dense retrieval",
    "sparse retrieval", "learning-to-rank", "FastAPI", "Docker",
    "Kubernetes", "PostgreSQL", "PyTorch", "scikit-learn", "embeddings",
    "production ML", "A/B testing", "search relevance", "re-ranking",
]
WEAK_SKILLS = [
    "LangChain", "Pinecone", "OpenAI API", "GPT-4", "RAG",
    "ChromaDB", "Weaviate", "ChatGPT", "prompt engineering",
    "LLM chains", "LangGraph", "LlamaIndex", "AutoGPT",
]
NEUTRAL_SKILLS = [
    "SQL", "REST APIs", "Git", "Linux", "AWS", "GCP", "Azure",
    "Spark", "Kafka", "Redis", "MongoDB", "TensorFlow", "Keras",
    "pandas", "NumPy", "matplotlib", "Jupyter",
]
COMPANIES_PRODUCT = [
    "Flipkart", "Meesho", "Swiggy", "Zomato", "Razorpay",
    "CRED", "Groww", "Zepto", "Urban Company", "Nykaa",
    "Dunzo", "ShareChat", "Unacademy", "Byju's", "PhonePe",
]
COMPANIES_STARTUP = [
    "Redrob AI", "Turing", "Moengage", "CleverTap", "Slintel",
    "Haptik", "Observe.AI", "Sprinklr", "Yellow.ai", "Exotel",
]
COMPANIES_SERVICE = [
    "Infosys", "TCS", "Wipro", "HCL", "Cognizant",
    "Capgemini", "Accenture", "Mphasis", "LTIMindtree",
]
COMPANIES_RESEARCH = [
    "IIT Research Lab", "IISc", "Microsoft Research India",
    "IBM Research", "Adobe Research",
]
TITLES = [
    "Senior ML Engineer", "ML Engineer", "Applied AI Engineer",
    "Senior Data Scientist", "Data Scientist", "AI Engineer",
    "Search Engineer", "NLP Engineer", "Backend + ML Engineer",
    "Principal Engineer", "Staff ML Engineer", "Research Scientist",
    "ML Researcher", "AI Developer", "Generative AI Specialist",
]
LOCATIONS = ["Bangalore", "Pune", "Mumbai", "Hyderabad", "Delhi", "Chennai",
             "Remote", "Gurgaon", "Noida", "Kolkata"]
DOMAINS = [
    ["retrieval", "ranking", "recommendation", "NLP"],
    ["computer vision", "NLP"],
    ["NLP", "text classification"],
    ["search", "information retrieval"],
    ["recommendation systems"],
    ["LLM applications", "RAG"],
    ["generative AI", "LLMs"],
    ["ranking", "search", "NLP"],
    ["speech", "audio"],
    ["CV", "robotics"],
]
PROJECTS_STRONG = [
    "Built semantic search system for product discovery using FAISS and sentence-transformers",
    "Improved search relevance using hybrid BM25 + dense retrieval; evaluated with NDCG@10",
    "Deployed learning-to-rank model for marketplace search; 18% lift in CTR",
    "Built and productionised candidate ranking API using FastAPI and vector search",
    "Implemented offline evaluation benchmark for ranking: NDCG, MRR, MAP",
    "Designed hybrid retrieval pipeline combining sparse and dense embeddings",
    "Owned A/B testing infrastructure for ranking experiments at scale",
    "Re-ranked search results using cross-encoder model; improved MRR by 22%",
    "Built recommendation engine for e-commerce using collaborative + content-based filtering",
    "Migrated BM25 search to dense retrieval; 35% improvement in recall@10",
]
PROJECTS_WEAK = [
    "Built RAG chatbot using LangChain and OpenAI",
    "Created GPT-4 powered Q&A system using LlamaIndex",
    "Fine-tuned LLM using LoRA for customer support chatbot",
    "Built document QA system with Pinecone vector store",
    "Deployed LangChain pipeline for PDF summarisation",
    "Created AI assistant using OpenAI function calling",
    "Built LangGraph agent for automated research",
]
PROJECTS_NEUTRAL = [
    "Sentiment analysis classifier for product reviews",
    "Named entity recognition model using spaCy",
    "Time series forecasting with LSTM",
    "Image classification with ResNet",
    "Text summarisation with BART",
]


def rand_date_str(days_back_min: int, days_back_max: int) -> str:
    d = datetime.now() - timedelta(days=random.randint(days_back_min, days_back_max))
    return d.strftime("%Y-%m-%d")


def build_candidate(idx: int) -> dict:
    profile_type = random.choices(
        ["strong", "weak", "medium", "trap", "research"],
        weights=[15, 20, 30, 20, 15],
    )[0]

    cid = f"C{1000 + idx}"

    # experience
    if profile_type == "strong":
        exp = random.randint(5, 10)
    elif profile_type == "research":
        exp = random.randint(4, 9)
    elif profile_type == "trap":
        exp = random.randint(2, 5)
    else:
        exp = random.randint(3, 9)

    title = random.choice(TITLES)
    if profile_type == "trap":
        title = random.choice(["Generative AI Specialist", "AI Developer", "LLM Engineer"])
    if profile_type == "research":
        title = random.choice(["Research Scientist", "ML Researcher"])

    # company
    if profile_type == "strong":
        company = random.choice(COMPANIES_PRODUCT + COMPANIES_STARTUP)
        company_type = "product"
    elif profile_type == "trap":
        company = random.choice(COMPANIES_SERVICE)
        company_type = "service"
    elif profile_type == "research":
        company = random.choice(COMPANIES_RESEARCH)
        company_type = "research"
    elif profile_type == "medium":
        company = random.choice(COMPANIES_PRODUCT + COMPANIES_SERVICE + COMPANIES_STARTUP)
        company_type = "product" if company in COMPANIES_PRODUCT + COMPANIES_STARTUP else "service"
    else:
        company = random.choice(COMPANIES_SERVICE)
        company_type = "service"

    location = random.choice(LOCATIONS)

    # skills
    if profile_type == "strong":
        skills = random.sample(STRONG_SKILLS, random.randint(8, 14))
        skills += random.sample(NEUTRAL_SKILLS, random.randint(3, 6))
    elif profile_type == "trap":
        skills = random.sample(WEAK_SKILLS, random.randint(6, 12))
        skills += random.sample(NEUTRAL_SKILLS, random.randint(2, 4))
    elif profile_type == "research":
        skills = random.sample(STRONG_SKILLS[:6], random.randint(2, 4))
        skills += random.sample(NEUTRAL_SKILLS, random.randint(4, 7))
    else:
        skills = random.sample(STRONG_SKILLS, random.randint(2, 6))
        skills += random.sample(WEAK_SKILLS, random.randint(1, 4))
        skills += random.sample(NEUTRAL_SKILLS, random.randint(3, 6))

    skills = list(dict.fromkeys(skills))  # dedup

    # projects
    if profile_type == "strong":
        projects = random.sample(PROJECTS_STRONG, random.randint(2, 4))
    elif profile_type == "trap":
        projects = random.sample(PROJECTS_WEAK, random.randint(2, 4))
    elif profile_type == "research":
        projects = random.sample(PROJECTS_NEUTRAL, random.randint(2, 3))
    else:
        projects = random.sample(PROJECTS_STRONG, 1) + random.sample(PROJECTS_WEAK + PROJECTS_NEUTRAL, 2)

    domains = random.choice(DOMAINS)

    # availability signals
    last_login = rand_date_str(1, 30) if profile_type in ("strong", "trap") else rand_date_str(10, 180)
    profile_updated = rand_date_str(1, 60)
    response_rate = round(random.uniform(0.3, 0.95), 2) if profile_type != "weak" else round(random.uniform(0.1, 0.4), 2)
    notice_period = random.choice([0, 15, 30, 45, 60, 90])
    relocation = random.choice(["yes", "no", "maybe"])
    open_to_work = random.choice([True, True, True, False])

    # external evidence
    has_github = random.random() < (0.7 if profile_type == "strong" else 0.25)
    has_portfolio = random.random() < (0.5 if profile_type == "strong" else 0.15)
    github_url = f"https://github.com/candidate_{cid.lower()}" if has_github else ""
    portfolio_url = f"https://portfolio.dev/{cid.lower()}" if has_portfolio else ""

    return {
        "candidate_id": cid,
        "name": f"Candidate {cid}",
        "current_title": title,
        "experience_years": exp,
        "skills": "|".join(skills),
        "projects": "|".join(projects),
        "technical_domains": "|".join(domains),
        "current_company": company,
        "company_type": company_type,
        "location": location,
        "github_url": github_url,
        "portfolio_url": portfolio_url,
        "last_login": last_login,
        "profile_updated_at": profile_updated,
        "recruiter_response_rate": response_rate,
        "notice_period": notice_period,
        "relocation_preference": relocation,
        "open_to_work": open_to_work,
        "_profile_type": profile_type,  # ground truth for evaluation
    }


def main():
    candidates = [build_candidate(i) for i in range(500)]
    fieldnames = list(candidates[0].keys())
    import pathlib
    out = str(pathlib.Path(__file__).parent / "candidates.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Generated {len(candidates)} candidates -> {out}")


if __name__ == "__main__":
    main()
