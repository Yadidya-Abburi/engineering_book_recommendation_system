# 📚 Bookify – Engineering Book Recommendation System

A content-based ML recommendation system for ~3,200 engineering and technical books,
with a Spotify-style web interface featuring VS Code-style book detail pages,
live cover images, download links, platform badges, and ML-powered similarity recs.

---

## 🗂️ Project Structure

```
bookify/
├── config/
│   ├── __init__.py
│   └── settings.py          ← Single source of truth: PATHS, MODEL, SCORE, CATEGORY_RULES
│
├── data/
│   ├── goodreads_engineering_books.csv
│   └── google_books_technical.csv
│
├── scripts/
│   ├── preprocess.py        ← Clean, merge, validate → outputs/books_clean.csv
│   ├── train_model.py       ← TF-IDF + cosine similarity + Bayesian hybrid scoring
│   ├── evaluate.py          ← 5 metrics + 6 visualisation plots
│   └── build_app.py         ← Pre-fetch covers + inject model data → app/index.html
│
├── tests/
│   ├── test_preprocess.py   ← 26 unit tests
│   ├── test_train_model.py  ← 22 unit tests
│   ├── test_evaluate.py     ←  6 unit tests
│   └── test_build_app.py    ←  9 unit tests  (82 total)
│
├── models/
│   ├── tfidf_vectorizer.pkl     ← Fitted TF-IDF vectoriser (gitignored, regenerable)
│   └── cosine_sim_matrix.npy   ← Pre-computed N×N similarity matrix (gitignored)
│
├── outputs/
│   ├── books_clean.csv          ← Merged, deduplicated, categorised dataset (~3,233 rows)
│   ├── top_books.csv            ← Top 498 ranked books with all 19 fields
│   ├── recommendations.json     ← Top-5 recs per book (pool-restricted: all links openable)
│   └── plots/                   ← 6 PNG charts from evaluate.py
│
├── app/
│   └── index.html           ← Self-contained Spotify-style web app (749 KB)
│
├── notebooks/
│   └── BookRecommender_EDA_and_Model.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Step 1 – Clean & merge datasets (skips 13-min API enrichment for quick runs)
python scripts/preprocess.py --skip-enrichment

# Full run with Google Books description enrichment (~13 min for 791 missing descriptions)
python scripts/preprocess.py

# Step 2 – Train the ML model (~5 seconds)
python scripts/train_model.py

# Step 3 – Evaluate & generate plots
python scripts/evaluate.py

# Step 4 – Rebuild the web app
python scripts/build_app.py --skip-covers   # fast: skip Open Library cover pre-fetch
python scripts/build_app.py                 # full: pre-fetch all cover images
```

Each script exits **0** on success, **1** on failure — safe to chain:
```bash
python scripts/preprocess.py --skip-enrichment && \
python scripts/train_model.py && \
python scripts/evaluate.py && \
python scripts/build_app.py --skip-covers
```

### 3. Run the tests
```bash
pytest tests/ -v
# With coverage report:
pytest tests/ -v --cov=scripts --cov-report=term-missing
```
All **82 tests** should pass.

### 4. Open the app
Open `app/index.html` in any modern browser. No server needed.

### 5. Explore the notebook
```bash
jupyter notebook notebooks/BookRecommender_EDA_and_Model.ipynb
```
Run cells top-to-bottom after completing step 1–2 above
(`outputs/books_clean.csv` must exist).

---

## ⚙️ Configuration

**All constants live in `config/settings.py`. Never hard-code paths or hyperparameters in scripts.**

```python
from config.settings import PATHS, MODEL, SCORE

# --- Key tunable constants ---

MODEL.TOP_N = 500          # Books retained for the app (raised from 150 to fix broken rec links)
MODEL.BAYES_MIN_VOTES = 500  # Bayesian rating threshold — see comment in settings.py

SCORE.WEIGHT_RATING     = 0.60   # Fraction of score from Bayesian average rating
SCORE.WEIGHT_POPULARITY = 0.40   # Fraction of score from log(ratings count)

# --- Auto-resolved paths ---
PATHS.CLEAN_CSV       # outputs/books_clean.csv
PATHS.TOP_BOOKS_CSV   # outputs/top_books.csv
PATHS.RECS_JSON       # outputs/recommendations.json
PATHS.TFIDF_PKL       # models/tfidf_vectorizer.pkl
PATHS.COSINE_NPY      # models/cosine_sim_matrix.npy
PATHS.APP_HTML        # app/index.html
```

To add new non-engineering keywords (domain filter), edit `NON_ENGINEERING_KEYWORDS` in
`config/settings.py` and re-run `preprocess.py` + `train_model.py`.

---

## 🤖 How the ML Works

### Pipeline overview

```
Raw CSVs  (Goodreads 2,375 + Google Books 1,488)
   ↓
Preprocessing
  · Clean fields · Fill nulls · Strip blocked CDN image URLs
  · Exact dedup + fuzzy dedup (normalised title key)
  · Domain filter (NON_ENGINEERING_KEYWORDS)
  · Optional: enrich missing descriptions via Google Books API
  · Assign one of 13 categories per book
   ↓
Feature Engineering
  title + author + description  →  TF-IDF  →  8,000-feature sparse vector
  (max_features=8000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
   ↓
Cosine Similarity Matrix
  pairwise cosine for all ~3,233 books  (N×N float64, ~80 MB)
   ↓
Bayesian Hybrid Scoring
  WR = (v/(v+m))×R + (m/(v+m))×C        (m=500 min votes, C=global mean)
  score = 0.60×bayes_norm + 0.40×log(count)_norm
  Re-scaled within top-N so rank-1 = 1.0, rank-N ≈ 0.0
   ↓
Top 500 books + top-5 similar books per title
  (recs restricted to the top-500 pool so every "Similar Books" link opens)
```

### TF-IDF in plain English
Words that appear often in **one** book but rarely across **all** books get high
scores — they identify that book's unique content. "Microcontroller" appearing 20
times in an embedded-systems book is more distinctive than "the".

### Cosine Similarity in plain English
Each book is a vector in 8,000-dimensional space. Cosine similarity measures the
angle between two vectors — **1.0** = same direction (near-identical content),
**0.0** = perpendicular (completely different).

### Why Bayesian scoring?
A book with 3 ratings of 5.0 should not outrank a book with 5,000 ratings at 4.2.
The Bayesian formula pulls low-vote books toward the global mean (the same approach
IMDb uses for its Top 250). See the comment in `config/settings.py` for tuning guidance.

---

## 📊 Datasets

| | Goodreads | Google Books | Combined |
|---|---|---|---|
| Raw rows | 2,375 | 1,488 | 3,863 |
| After exact dedup | — | — | 3,349 |
| After fuzzy dedup | — | — | 3,249 |
| After domain filter | — | — | ~3,233 |
| Has download link | 2,296 / 2,375 | 0 | 2,296 |
| Has rating | 2,375 / 2,375 | 102 / 1,488 | — |
| Has price | 0 | 520 / 1,488 | 520 |

---

## 🧪 Test Coverage

| Module | Tests | Classes | What's covered |
|---|---|---|---|
| `preprocess.py` | 69 | 12 | `parse_year`, `clean_image_url`, `is_engineering_book`, `assign_category`, `normalise_title`, `validate` (all warning paths), both CSV loaders, description enrichment, `main()` |
| `train_model.py` | 33 | 8 | TF-IDF, cosine, Bayesian scoring, recs structure, near-dup dedup, flat-score edge case, `save_artefacts`, `load_data`, `main()` |
| `evaluate.py`    | 19 | 5 | All 5 metrics, empty inputs, `load()` FileNotFoundError, all 6 plot functions, self-rec warning in `main()` |
| `build_app.py`   | 30 | 9 | Load/clean outputs, validate HTML, `read_template`, `inject_data` (guard + injection), `fetch_cover_url` (5 paths), `prefetch_covers`, `main()` success + failure |
| **Total** | **149** | **34** | 92% line coverage across all 4 modules |

Run: `pytest tests/ -v` — all 149 pass.  
Run with coverage: `pytest tests/ --cov=scripts --cov-report=term-missing`

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.2 | Data loading, cleaning, merging |
| `numpy` | ≥ 1.26 | Matrix ops, cosine similarity storage |
| `scikit-learn` | ≥ 1.5 | TfidfVectorizer, cosine_similarity, MinMaxScaler |
| `joblib` | ≥ 1.4 | Serialise / load fitted TF-IDF vectoriser |
| `requests` | ≥ 2.31 | Description enrichment + cover pre-fetch (lazy import) |
| `matplotlib` / `seaborn` | ≥ 3.9 / 0.13 | Evaluation plots |
| `pytest` / `pytest-cov` | ≥ 8.3 / 5.0 | 82-test suite + coverage |
| `jupyter` + `ipykernel` | pinned | EDA + training walkthrough notebook |
| HTML / CSS / JS | — | 749 KB self-contained Spotify-style app |
| Open Library API | — | Book cover images (build-time pre-fetch + runtime fallback) |

> **Python version**: tested on **3.10 – 3.12**.  
> **pandas 3.x and numpy 2.x** are fully supported — no deprecation warnings.

---

## 🗺️ Known Limitations

| Issue | Impact | Workaround |
|---|---|---|
| 24.5% of books have no description | Lower TF-IDF quality for those books | Run `preprocess.py` without `--skip-enrichment` |
| Open Library covers not always available | ~30–40% of books show placeholder emoji | Pre-fetch at build time with `build_app.py` (no `--skip-covers`) |
| `year=0` for ~4 classic/undated books | Shows "Classic" badge instead of year | ISBN lookup via Google Books API (future work) |
| Cosine similarity matrix is ~80 MB | Loaded fully into RAM at train time | Sparse approximation or ANN index (future work) |

---

## 🔗 Repository

```
github.com/yourname/bookify
```
