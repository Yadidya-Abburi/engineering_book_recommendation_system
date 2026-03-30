# 📚 Bookify – Engineering Book Recommendation System

A content-based ML recommendation system for ~3,200 engineering and technical books,
with a premium dark-themed web interface featuring glassmorphism UI, fuzzy search,
live cover images, download links, AI-powered similarity recommendations,
favorites (localStorage), and full mobile responsiveness.

---

## 🗂️ Project Structure

```
bookify/
├── config/
│   ├── __init__.py
│   └── settings.py          ← Single source of truth: PATHS, MODEL, SCORE, CATEGORY_RULES
│
├── data/
│   └── final_books.csv      ← Merged Goodreads + Google Books dataset (3,293 rows)
│
├── scripts/
│   ├── preprocess.py        ← Clean, merge, validate → outputs/books_clean.csv
│   ├── train_model.py       ← TF-IDF + cosine similarity + Bayesian hybrid scoring
│   └── evaluate.py          ← 5 metrics + 6 visualisation plots
│
├── tests/
│   ├── test_preprocess.py   ← 69 unit tests
│   ├── test_train_model.py  ← 33 unit tests
│   ├── test_evaluate.py     ← 19 unit tests
│   └── test_build_app.py    ← 30 unit tests  (149 total)
│
├── models/
│   ├── tfidf_vectorizer.pkl     ← Fitted TF-IDF vectoriser (gitignored, regenerable)
│   └── cosine_sim_matrix.npy    ← Pre-computed N×N similarity matrix (gitignored)
│
├── outputs/
│   ├── books_clean.csv          ← Merged, deduplicated, categorised dataset (~3,233 rows)
│   ├── top_books.csv            ← Full processed corpus ready for the app (~3,233 rows)
│   ├── recommendations.json     ← Top-5 recs per book (category-boosted, pool-restricted)
│   └── plots/                   ← 6 PNG charts from evaluate.py
│
├── app/
│   ├── app.py               ← Flask backend with caching, search API, health check
│   └── templates/
│       └── index.html       ← Premium dark-themed web app (glassmorphism, responsive)
│
├── notebooks/
│   └── BookRecommender_EDA_and_Model.ipynb
│
├── launch.py                ← Convenience launcher
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Step-by-Step Execution Guide (From Pull to Run)

Follow these steps to get the full 3,200+ book recommendation system running locally:

### 1. Pull the repository and set up environment
First, pull the latest code and create a clean Python virtual environment.
```bash
git pull origin main
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Data Pipeline & ML Training
Process the raw data, generate the TF-IDF vectors, compute cosine similarities, and extract the JSON recommendations. This processes the entire 3,200+ book corpus.
```bash
# Step 1: Clean & merge raw CSV datasets 
# (omitting --skip-enrichment will take ~13 mins to fetch descriptions from Google Books API)
python scripts/preprocess.py --skip-enrichment

# Step 2: Train the ML model and generate recommendations
python scripts/train_model.py

# Step 3: (Optional) Evaluate model metrics and generate charts
python scripts/evaluate.py
```

### 4. Launch the Web App
Start the Flask backend server that serves the beautiful UI and in-memory search APIs:
```bash
python app/app.py
```
Open **http://127.0.0.1:8765** in any browser to explore the massive AI-curated engineering library!

### 5. Run the tests (Optional)
```bash
pytest tests/ -v
# With coverage report:
pytest tests/ -v --cov=scripts --cov-report=term-missing
```
All **119 tests** should pass (~92% line coverage).

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

MODEL.TOP_N = 10000        # Ensure all processed books are retained for the app
MODEL.BAYES_MIN_VOTES = 500  # Bayesian rating threshold

SCORE.WEIGHT_RATING     = 0.60   # Fraction of score from Bayesian average rating
SCORE.WEIGHT_POPULARITY = 0.40   # Fraction of score from log(ratings count)
```

---

## 🤖 How the ML Works

### Pipeline overview

```
Raw CSV  (merged Goodreads 2,375 + Google Books filtered 918 = 3,293 rows)
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
Top ~3233 books + top-5 similar books per title
  (category-boosted: same-category books get +0.15 similarity boost)
  (recs restricted to top-500 pool so every "Similar Books" link works)
```

### Category-Boosted Recommendations

When selecting similar books, same-category books receive a +0.15 boost to their
cosine similarity score. This ensures recommendations are topically relevant while
still allowing genuinely similar cross-category matches to surface.

**Result:** Category precision improved from 51.6% → 92.0%.

---

## 🌐 API Reference

The Flask backend serves the following endpoints on `http://127.0.0.1:8765`:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web app |
| `/api/books` | GET | All ~3,233 processed books (JSON array) |
| `/api/books?domain=AI` | GET | Books filtered by category |
| `/api/recs` | GET | All recommendations (JSON object: title → [recs]) |
| `/api/search?q=python` | GET | Server-side fuzzy search |
| `/api/health` | GET | System health: book count, categories, cache status |

All data is cached in memory at startup for fast responses.

---

## 📊 Datasets

| | Goodreads | Google Books | Combined |
|---|---|---|---|
| Input rows | 2,375 | 918 (filtered) | 3,293 |
| After exact dedup | — | — | 3,249 |
| After fuzzy dedup | — | — | ~3,233 |
| After domain filter | — | — | ~3,233 |
| Has download link | 2,296 / 2,375 | 0 | 2,296 |
| Has rating | 2,375 / 2,375 | 102 / 918 | — |

---

## 📈 Model Evaluation Metrics

| Metric | Value |
|---|---|
| Coverage | 100% (all books have recommendations) |
| Category Precision | 92.0% |
| Avg Similarity Score | 0.186 |
| Intra-list Diversity | 0.916 |
| Self-recommendations | 0 |

---

## 🧪 Test Coverage

| Module | Tests | What's covered |
|---|---|---|
| `preprocess.py` | 69 | `parse_year`, `clean_image_url`, `is_engineering_book`, `assign_category`, `normalise_title`, `validate`, both CSV loaders, description enrichment, `main()` |
| `train_model.py` | 33 | TF-IDF, cosine, Bayesian scoring, recs structure, near-dup dedup, `save_artefacts`, `load_data`, `main()` |
| `evaluate.py` | 19 | All 5 metrics, empty inputs, all 6 plot functions, `main()` |
| **Total** | **119** | ~92% line coverage |

Run: `pytest tests/ -v` — all 119 pass.

---

## 🎨 Web App Features

- **Premium dark theme** with glassmorphism sidebar and gradient accents
- **13 engineering domains** with book counts (Electrical, AI/ML, Security, etc.)
- **Fuzzy search** with word-prefix matching
- **3 sort modes**: Best Match (AI score), Rating, Popularity
- **Book detail panel** with cover art, ratings, download links, and similar books
- **Favorites** (localStorage) — heart any book to save it
- **Keyboard shortcuts** — `/` to search, `Esc` to close
- **Skeleton loading** with shimmer animation during API fetch
- **Mobile responsive** — hamburger menu, full-screen detail panel on phones
- **Cover images** from Open Library API with 5-second fallback to emoji

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.2 | Data loading, cleaning, merging |
| `numpy` | ≥ 1.26 | Matrix ops, cosine similarity storage |
| `scikit-learn` | ≥ 1.5 | TfidfVectorizer, cosine_similarity, MinMaxScaler |
| `joblib` | ≥ 1.4 | Serialise / load fitted TF-IDF vectoriser |
| `flask` | ≥ 3.0 | Web backend with caching and search API |
| `requests` | ≥ 2.31 | Description enrichment + cover pre-fetch |
| `matplotlib` / `seaborn` | ≥ 3.9 / 0.13 | Evaluation plots |
| `pytest` / `pytest-cov` | ≥ 8.3 / 5.0 | 149-test suite + coverage |
| `thefuzz` | ≥ 0.22 | Fuzzy string matching |
| HTML / CSS / JS | — | Premium responsive web app |
| Open Library API | — | Book cover images |

> **Python version**: tested on **3.10 – 3.14**.

---

## 🗺️ Known Limitations

| Issue | Impact | Workaround |
|---|---|---|
| 24.5% of books have no description | Lower TF-IDF quality | Run `preprocess.py` without `--skip-enrichment` |
| Open Library covers not always available | ~30–40% show emoji placeholder | Covers fetched dynamically on client-side |
| Cosine similarity matrix is ~80 MB | Loaded fully into RAM | Sparse approximation or ANN index (future) |
| `year=0` for a few undated books | Shows "Classic" badge | ISBN lookup via Google Books API (future) |
