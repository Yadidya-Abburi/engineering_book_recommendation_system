"""
train_model.py
──────────────
Trains the content-based recommendation model.

Fixes applied:
  #1  TOP_N raised to 500 (config change) so similar-book links work in UI
  #3  Score re-scaling within top-N — spreads retained books across full 0-1
      range so "Sort by Best Match" is actually meaningful
  #8  Bayesian average scoring (unchanged, already correct)
  #11 Imports directly from config.settings

Run:
    python scripts/train_model.py
Exit code 0 = success, 1 = failure.
"""

import json
import logging
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS, MODEL, SCORE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    if not os.path.exists(PATHS.CLEAN_CSV):
        log.info("Clean CSV not found — running preprocess.py first…")
        result = subprocess.run(
            [sys.executable, os.path.join(PATHS.ROOT, "scripts", "preprocess.py")],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("preprocess.py failed — check logs above.")

    df = pd.read_csv(PATHS.CLEAN_CSV)
    df["description"]   = df["description"].fillna("")
    df["author"]        = df["author"].fillna("Unknown")
    df["rating"]        = pd.to_numeric(df["rating"],        errors="coerce").fillna(0)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)
    df["download_link"] = df["download_link"].fillna("")
    df["file_info"]     = df["file_info"].fillna("")
    df["price"]         = pd.to_numeric(df["price"],         errors="coerce").fillna(0.0)
    df["currency"]      = df["currency"].fillna("USD")
    for col in ["preview_link", "info_link", "image"]:
        df[col] = df[col].fillna("").replace("nan", "")
    return df


def build_tfidf(df: pd.DataFrame):
    """Fit TF-IDF on combined title + author + description."""
    log.info("Building TF-IDF matrix (max_features=%d, ngrams=%s)…",
             MODEL.TFIDF_MAX_FEATURES, MODEL.TFIDF_NGRAM_RANGE)
    df["content"] = df["title"] + " " + df["author"] + " " + df["description"]
    vectorizer = TfidfVectorizer(
        max_features = MODEL.TFIDF_MAX_FEATURES,
        stop_words   = "english",
        ngram_range  = MODEL.TFIDF_NGRAM_RANGE,
        sublinear_tf = MODEL.TFIDF_SUBLINEAR_TF,
        min_df       = MODEL.TFIDF_MIN_DF,
    )
    matrix = vectorizer.fit_transform(df["content"])
    log.info("  Matrix: %d books × %d features", *matrix.shape)
    return vectorizer, matrix


def build_similarity(matrix) -> np.ndarray:
    """Compute full pairwise cosine similarity matrix."""
    log.info("Computing cosine similarity matrix…")
    t0 = time.time()
    sim = cosine_similarity(matrix, matrix)
    log.info("  Done in %.1fs — shape: %s", time.time() - t0, sim.shape)
    return sim


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bayesian-average hybrid score.

    WR = (v / (v+m)) × R  +  (m / (v+m)) × C
      v = book's ratings_count
      m = BAYES_MIN_VOTES (see config/settings.py for explanation)
      R = book's average rating
      C = global mean rating

    Then normalised: score = 0.60×bayes_norm + 0.40×log(count)_norm
    """
    df = df.copy()
    rated = df[df["rating"] > 0]
    C = rated["rating"].mean() if not rated.empty else 4.0
    m = MODEL.BAYES_MIN_VOTES

    v = df["ratings_count"]
    df["bayes_rating"] = (v / (v + m)) * df["rating"] + (m / (v + m)) * C
    df.loc[df["rating"] == 0, "bayes_rating"] = 0

    df["log_count"] = np.log1p(df["ratings_count"])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["bayes_rating", "log_count"]])
    df["score"] = SCORE.WEIGHT_RATING * scaled[:, 0] + SCORE.WEIGHT_POPULARITY * scaled[:, 1]
    df.loc[df["rating"] == 0, "score"] = 0.0
    return df


def extract_top_and_recs(df: pd.DataFrame, sim: np.ndarray):
    """
    Select top-N books and generate N_RECS recommendations each.

    FIX #3: Re-scale scores within the top-N after selection.
    All scores in the full corpus (3,000+ books) get compressed into a
    narrow band because only the best books are retained. Re-scaling
    maps the retained range to [0, 1] so "Sort by Best Match" in the UI
    gives meaningful ranking (rank-1 scores 1.0, rank-N scores 0.0).

    FIX #1 (via config): TOP_N is now 500 so similar-book links can open
    in the app — previously 85% silently failed.
    """
    top = df.nlargest(MODEL.TOP_N, "score").copy().reset_index()
    top["desc_short"] = top["description"].str.strip().str[:MODEL.DESC_SHORT_LEN]
    top["year"]       = top["year"].astype(int)
    top["pages"]      = top["pages"].astype(int)
    top["title"]      = top["title"].str.strip()

    # ── FIX #3: re-scale score within top-N ───────────────────────────────
    s_min, s_max = top["score"].min(), top["score"].max()
    if s_max > s_min:
        top["score"] = ((top["score"] - s_min) / (s_max - s_min)).round(4)
    else:
        top["score"] = top["score"].round(4)

    # ── Near-duplicate deduplication within top-N ────────────────────────────
    # TF-IDF + cosine can't distinguish "8051 Microcontroller" from "8051 MICROCONTROLLER"
    # Remove the lower-scoring copy of any pair with normalised-title similarity > MODEL.NEAR_DUP_THRESHOLD
    from difflib import SequenceMatcher as _SM
    import re as _re
    norms = top["title"].str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.strip()
    drop_idx: set[int] = set()
    for _i in range(len(norms)):
        for _j in range(_i + 1, len(norms)):
            if _j in drop_idx:
                continue
            if _SM(None, norms.iloc[_i], norms.iloc[_j]).ratio() > MODEL.NEAR_DUP_THRESHOLD:
                keep = _i if top.iloc[_i]["score"] >= top.iloc[_j]["score"] else _j
                drop = _j if keep == _i else _i
                drop_idx.add(drop)
                log.info("  Near-dup removed: '%s'", top.iloc[drop]["title"][:65])
    if drop_idx:
        top = top.drop(index=list(drop_idx)).reset_index(drop=True)
        log.info("  Near-dup dedup: %d removed → %d books", len(drop_idx), len(top))

    rated = top[top["rating"] > 0]
    log.info("  Top %d books — avg rating: %.2f  score range: %.3f–%.3f  std: %.4f",
             len(top),
             rated["rating"].mean() if not rated.empty else 0,
             top["score"].min(), top["score"].max(), top["score"].std())

    # Build recs — restricted to top-N pool so every link is openable in the UI
    top_index_set = set(top["index"].astype(int).tolist())
    # Build recs — exclude self (j==i) and exact duplicates (sim==1.0)
    recs: dict[str, list[dict]] = {}
    for _, row in top.iterrows():
        i     = int(row["index"])
        title = row["title"]
        # FIX #1 (complete): restrict candidates to top-N indices only.
        # Previously recs were drawn from the full 3,233-book corpus so
        # clicking a similar book in the UI called openDP() on a title that
        # wasn't in the BOOKS array -> silent blank panel.
        scores = [
            (j, float(s))
            for j, s in enumerate(sim[i])
            if j != i and float(s) < 1.0 and j in top_index_set
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[: MODEL.N_RECS]
        recs[title] = [
            {
                "title"      : df.iloc[j]["title"].strip(),
                "author"     : df.iloc[j]["author"],
                "rating"     : round(float(df.iloc[j]["rating"]), 1),
                "similarity" : round(s, 3),
            }
            for j, s in scores
        ]

    remaining_self = sum(1 for t, rl in recs.items() for r in rl if r["title"] == t)
    log.info("  Self-recommendations remaining: %d (must be 0)", remaining_self)

    out_cols = [
        "title", "author", "publisher", "year", "pages",
        "rating", "ratings_count", "score", "desc_short",
        "image", "category", "source", "platform",
        "download_link", "file_info", "price", "currency",
        "preview_link", "info_link",
    ]
    out_cols = [c for c in out_cols if c in top.columns]
    return top[out_cols], recs


def save_artefacts(vectorizer, sim_matrix, top_books, recs):
    os.makedirs(PATHS.MODELS,  exist_ok=True)
    os.makedirs(PATHS.OUTPUTS, exist_ok=True)

    joblib.dump(vectorizer, PATHS.TFIDF_PKL)
    log.info("Saved TF-IDF vectoriser → %s", PATHS.TFIDF_PKL)

    np.save(PATHS.COSINE_NPY, sim_matrix)
    log.info("Saved cosine sim matrix  → %s", PATHS.COSINE_NPY)

    top_books.to_csv(PATHS.TOP_BOOKS_CSV, index=False)
    log.info("Saved top books CSV      → %s", PATHS.TOP_BOOKS_CSV)

    with open(PATHS.RECS_JSON, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    log.info("Saved recommendations    → %s", PATHS.RECS_JSON)


def main() -> int:
    log.info("=" * 58)
    log.info("  Bookify – Training the Recommendation Model")
    log.info("=" * 58)

    try:
        log.info("Step 1/5  Loading data…")
        df = load_data()
        log.info("  %d books loaded", len(df))

        log.info("Step 2/5  TF-IDF vectorisation…")
        vectorizer, tfidf_matrix = build_tfidf(df)

        log.info("Step 3/5  Cosine similarity…")
        sim_matrix = build_similarity(tfidf_matrix)

        log.info("Step 4/5  Bayesian scoring…")
        df = compute_scores(df)

        log.info("Step 5/5  Extracting top %d books + recommendations…", MODEL.TOP_N)
        top_books, recs = extract_top_and_recs(df, sim_matrix)

        save_artefacts(vectorizer, sim_matrix, top_books, recs)

    except Exception as exc:
        log.exception("Training failed: %s", exc)
        return 1

    log.info("-" * 58)
    log.info("  Corpus:       %d books", len(df))
    log.info("  Top retained: %d", len(top_books))
    for cat, count in top_books["category"].value_counts().items():
        log.info("    %-26s %d", cat, count)
    log.info("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
