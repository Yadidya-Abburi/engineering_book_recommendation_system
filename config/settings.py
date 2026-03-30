"""
config/settings.py
──────────────────
Single source of truth for every tunable constant in the project.
Import directly: from config.settings import PATHS, MODEL, SCORE
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PATHS:
    ROOT       = _ROOT
    DATA       = os.path.join(_ROOT, "data")
    OUTPUTS    = os.path.join(_ROOT, "outputs")
    MODELS     = os.path.join(_ROOT, "models")
    APP        = os.path.join(_ROOT, "app")
    PLOTS      = os.path.join(_ROOT, "outputs", "plots")

    GOODREADS_CSV = os.path.join(DATA,    "goodreads_engineering_books.csv")
    GOOGLE_CSV    = os.path.join(DATA,    "google_books_technical.csv")
    FINAL_CSV     = os.path.join(DATA,    "final_books.csv")
    CLEAN_CSV     = os.path.join(OUTPUTS, "books_clean.csv")
    TFIDF_PKL     = os.path.join(MODELS,  "tfidf_vectorizer.pkl")
    COSINE_NPY    = os.path.join(MODELS,  "cosine_sim_matrix.npy")
    TOP_BOOKS_CSV = os.path.join(OUTPUTS, "top_books.csv")
    RECS_JSON     = os.path.join(OUTPUTS, "recommendations.json")


class MODEL:
    TFIDF_MAX_FEATURES = 8_000
    TFIDF_NGRAM_RANGE  = (1, 2)
    TFIDF_SUBLINEAR_TF = True
    TFIDF_MIN_DF       = 2

    # FIX #1 — was 150, raised to 500 so that similar-book links can be
    # opened in the app. Previously 85% of "Similar Books" clicks silently
    # failed because the recommended book was outside the top-150 BOOKS array.
    TOP_N = 10000

    N_RECS              = 5
    DESC_SHORT_LEN      = 300

    # Similarity threshold above which two normalised titles are treated as
    # near-duplicates and deduplicated (keep higher-scoring copy).
    NEAR_DUP_THRESHOLD  = 0.98

    # Bayesian average minimum-vote threshold (IMDb-style).
    # Books with fewer ratings than this are pulled toward the global mean,
    # preventing a 3-review 5-star book from outranking an established classic.
    # Lower  → raw ratings carry more weight (favours niche books).
    # Higher → small-sample books regress more toward the mean.
    # For a ~3,000-book corpus, 500 is a reasonable default (IMDb uses 25,000).
    BAYES_MIN_VOTES = 500


class SCORE:
    WEIGHT_RATING     = 0.60
    WEIGHT_POPULARITY = 0.40


# ── Category keyword rules ──────────────────────────────────────────────────
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("AI/ML",          ["machine learning", "deep learning", "neural network",
                        " ai ", "artificial intelligence", "reinforcement",
                        "tensorflow", "pytorch", "llm", "natural language"]),
    ("Security",       ["cybersecurity", "security", "hacking", "cryptograph",
                        "encryption", "penetration", "cissp", "malware", "vulnerability"]),
    ("Cloud/DevOps",   ["cloud", "devops", "kubernetes", "docker", "aws",
                        "microservice", "terraform", "ci/cd", "azure", "gcp"]),
    ("Data Science",   ["data science", "data analysis", "statistics",
                        "analytics", "visualization", "pandas", "tableau"]),
    ("Databases",      ["database", " sql", "nosql", "mongodb",
                        "postgresql", "oracle", "mysql", "redis"]),
    ("Software Eng",   ["software engineering", "agile", "scrum",
                        "design pattern", "software architecture",
                        "clean code", "refactoring"]),
    ("Game/Graphics",  ["game engine", "3d rendering", "opengl", "directx",
                        "unity", "unreal", "vulkan", "shader", "game development",
                        "real-time rendering", "computer graphics"]),
    ("Electrical",     ["electrical", "circuit", "signal processing",
                        "embedded", "microcontroller", "vhdl", "fpga", "power system"]),
    ("Mechanical",     ["mechanical", "thermodynamics", "fluid", "structural",
                        "manufacturing", "cad", "finite element"]),
    ("Civil",          ["boiler", "dam engineering", "bridge", "concrete",
                        "reinforced", "geotechnical", "hydraulics", "surveying"]),
    ("Networking",     ["network", "tcp/ip", "protocol", "routing",
                        "wireless", "optical", "communication system"]),
    ("Programming",    ["python", "java ", "javascript", "c++", "c#",
                        "programming", "algorithm", "data structure",
                        "compiler", "operating system"]),
]
DEFAULT_CATEGORY = "General Engineering"


# ── Domain exclusion list ────────────────────────────────────────────────────
# Books matching ANY of these phrases are removed from the corpus entirely.
# FIX #5 — added trading/finance terms; previously "Professional Stock Trading:
# System Design and Automation" passed the filter and appeared in rank 11.
NON_ENGINEERING_KEYWORDS: list[str] = [
    # Finance / trading (FIX #5)
    "stock trading", "trading system", "algorithmic trading", "forex trading",
    "options trading", "futures trading", "cryptocurrency trading",
    "technical analysis trading", "day trading", "swing trading",
    "investing for beginners", "stock market basics", "personal finance",

    # Marketing / social media
    "principles of modern marketing", "instagram marketing",
    "social media marketing", "email marketing", "content marketing",

    # Medical / clinical
    "clinical atlas", "anatomy atlas", "medical diagnosis",
    "diagnostics vaccine", "covid vaccine", "rapid diagnostics",

    # Food / lifestyle
    "cooking recipes", "baking recipes", "recipe book",
    "yoga", "meditation", "mindfulness", "wellness",
    "diet plan", "weight loss", "fitness nutrition",

    # Fiction / non-technical
    "romance novel", "mystery thriller", "science fiction novel",

    # General business / soft-skills
    "leadership secrets", "management secrets",

    # Arts
    "music theory for beginners", "learn guitar", "piano lessons",

    # Travel
    "travel guide", "phrasebook",
]
