"""
preprocess.py
─────────────
Loads both raw CSVs, cleans & normalises fields, merges, deduplicates,
filters non-engineering books, enriches missing descriptions, assigns
categories, validates output, and writes outputs/books_clean.csv.

Fixes applied:
  #2  Fuzzy deduplication — normalise title before dedup (catches case/
      punctuation variants like "and" vs "And", ":" vs " ")
  #4  Wire up fetch_google_description — called for all rows with empty
      description after the merge step
  #5  Stronger domain filter — trading/finance keywords added
  #6  Robust year parser (unchanged from previous version)
  #11 Imports directly from config.settings

Run:
    python scripts/preprocess.py
Exit code 0 = success, 1 = failure.
"""

import argparse
import logging
import os
import re
import sys
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PATHS, CATEGORY_RULES, DEFAULT_CATEGORY, NON_ENGINEERING_KEYWORDS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REQUIRED_COLS = [
    "title", "author", "publisher", "year", "pages",
    "rating", "ratings_count", "description", "image",
    "download_link", "file_info", "price", "currency",
    "preview_link", "info_link", "source", "platform", "category",
]


# ── Year parser ────────────────────────────────────────────────────────────
def parse_year(val) -> int:
    """Extract a 4-digit year from any date string. Returns 0 on failure."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if re.match(r"^\d{4}$", s):
        y = int(s)
        return y if 1800 <= y <= 2030 else 0
    m = re.match(r"^(\d{4})[-/]", s)
    if m:
        y = int(m.group(1))
        return y if 1800 <= y <= 2030 else 0
    m = re.search(r"\b(1[89]\d{2}|20[012]\d)\b", s)
    if m:
        return int(m.group(1))
    return 0


# ── Image URL cleaner ──────────────────────────────────────────────────────
def clean_image_url(url: str) -> str:
    """Return empty string for known-blocked CDN URLs."""
    if not url or pd.isna(url):
        return ""
    url = str(url).strip()
    if "zlibcdn" in url or "1lib.in" in url:
        return ""
    return url.replace("http://", "https://")


# ── FIX #2: Normalised title key for fuzzy dedup ──────────────────────────
def normalise_title(title: str) -> str:
    """
    Return a lowercase, punctuation-stripped version of the title used
    exclusively for deduplication — the original title is preserved.

    This catches pairs like:
      "Programming And Applications" vs "Programming and Applications"
      "Design, Implementation, & Management" vs "Design, Implementation, & Management"
      "Microcontroller: Architecture" vs "Microcontroller Architecture"
    """
    s = str(title).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # strip all punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Loaders ────────────────────────────────────────────────────────────────
def load_goodreads(path: str) -> pd.DataFrame:
    log.info("Loading Goodreads CSV: %s", path)
    df = pd.read_csv(path)
    df = df.dropna(subset=["title"]).copy()
    df["title"]         = df["title"].str.strip()
    df["description"]   = df["desc"].fillna("")
    df["author"]        = df["author"].fillna("Unknown")
    df["publisher"]     = df["publisher"].fillna("Unknown")
    df["year"]          = df["year"].apply(parse_year)
    df["pages"]         = pd.to_numeric(df["pages"], errors="coerce").fillna(0).astype(int)
    df["rating"]        = pd.to_numeric(df["goodreads_rating"], errors="coerce").fillna(0)
    df["ratings_count"] = pd.to_numeric(df["goodreads_ratings_count"], errors="coerce").fillna(0).astype(int)
    df["image"]         = df["image"].apply(clean_image_url)
    df["download_link"] = df["download_link"].fillna("")
    df["file_info"]     = df["file"].fillna("")
    df["price"]         = 0.0
    df["currency"]      = "USD"
    df["preview_link"]  = ""
    df["info_link"]     = ""
    df["source"]        = "Goodreads / Z-Library"
    df["platform"]      = df["file_info"].apply(
        lambda x: x.split(",")[0].strip() if x else "PDF"
    )
    log.info("  Goodreads: %d books loaded", len(df))
    return df[REQUIRED_COLS[:-1]]


def load_google(path: str) -> pd.DataFrame:
    log.info("Loading Google Books CSV: %s", path)
    df = pd.read_csv(path)
    df = df[df["language"] == "en"].dropna(subset=["title"]).copy()
    df["title"]         = df["title"].str.strip()
    df["description"]   = df["description"].fillna("")
    df["author"]        = df["authors"].fillna("Unknown")
    df["publisher"]     = df["publisher"].fillna("Unknown")
    df["year"]          = df["published_date"].apply(parse_year)
    df["pages"]         = pd.to_numeric(df["page_count"],     errors="coerce").fillna(0).astype(int)
    df["rating"]        = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"],  errors="coerce").fillna(0).astype(int)
    df["image"]         = df["thumbnail"].apply(clean_image_url)
    df["download_link"] = ""
    df["file_info"]     = ""
    df["price"]         = pd.to_numeric(df["list_price"], errors="coerce").fillna(0.0)
    df["currency"]      = df["currency"].fillna("USD")
    df["preview_link"]  = df["preview_link"].fillna("").str.replace("http://", "https://", regex=False)
    df["info_link"]     = df["info_link"].fillna("").str.replace("http://", "https://", regex=False)
    df["source"]        = "Google Books"
    df["platform"]      = "Google Books"
    log.info("  Google Books: %d books loaded", len(df))
    return df[REQUIRED_COLS[:-1]]


# ── FIX #5: Domain filter ──────────────────────────────────────────────────
def is_engineering_book(title: str, description: str) -> bool:
    """Return False for books that clearly don't belong in an engineering corpus."""
    text = (str(title) + " " + str(description)).lower()
    return not any(kw in text for kw in NON_ENGINEERING_KEYWORDS)


# ── Category assignment ────────────────────────────────────────────────────
def assign_category(title: str, description: str) -> str:
    text = (str(title) + " " + str(description)).lower()
    for category, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return DEFAULT_CATEGORY


# ── FIX #4: Description enrichment ────────────────────────────────────────
def enrich_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For books whose description is empty, attempt to fetch it from the
    Google Books API (no key required, free endpoint).

    Rate-limited to 1 request/second to stay within API limits.
    Only runs if there are books needing enrichment — silent no-op otherwise.
    """
    import requests  # lazy import — only needed when descriptions are missing
    import urllib.parse

    empty_mask = df["description"].str.strip() == ""
    empty_count = empty_mask.sum()
    if empty_count == 0:
        log.info("  No missing descriptions — skipping enrichment.")
        return df

    log.info("  Enriching %d books with empty descriptions via Google Books API…", empty_count)
    df = df.copy()
    enriched = 0
    failed   = 0

    for idx in df[empty_mask].index:
        title  = df.at[idx, "title"]
        author = df.at[idx, "author"]
        try:
            q   = urllib.parse.quote(f"intitle:{title[:40]}")
            url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=1"
            r   = requests.get(url, timeout=6)
            items = r.json().get("items", [])
            if items:
                desc = items[0].get("volumeInfo", {}).get("description", "")
                if desc:
                    df.at[idx, "description"] = desc
                    enriched += 1
        except Exception:
            failed += 1
        time.sleep(1.0)   # respect API rate limit

    log.info(
        "  Enrichment: %d recovered, %d failed, %d still empty",
        enriched, failed, empty_count - enriched - failed,
    )
    return df


# ── Validation ────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> bool:
    ok = True
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        log.error("Missing columns: %s", missing_cols)
        return False

    dupes = df["title"].duplicated().sum()
    if dupes:
        log.warning("%d duplicate titles remain", dupes)

    # FIX #2: also check normalised-key duplicates
    norm_dupes = df["title"].apply(normalise_title).duplicated().sum()
    if norm_dupes:
        log.warning("%d near-duplicate titles remain after normalised dedup", norm_dupes)

    bad_ratings = df[(df["rating"] < 0) | (df["rating"] > 5)]
    if not bad_ratings.empty:
        log.warning("%d books have ratings outside [0, 5]", len(bad_ratings))

    bad_years = df[(df["year"] > 0) & ((df["year"] < 1800) | (df["year"] > 2030))]
    if not bad_years.empty:
        log.warning("%d books have suspicious years", len(bad_years))

    blocked = df["image"].str.contains("zlibcdn|1lib.in", na=False).sum()
    if blocked:
        log.warning("%d blocked image URLs still present", blocked)

    empty_descs = (df["description"].str.strip() == "").sum()
    pct = empty_descs / len(df) * 100
    if pct > 30:
        log.warning("%.1f%% of books have empty descriptions", pct)
    else:
        log.info("  Description coverage: %.1f%% have content", 100 - pct)

    if len(df) < 500:
        log.error("Only %d rows — expected 2,500+", len(df))
        ok = False

    return ok


# ── Main pipeline ──────────────────────────────────────────────────────────
def main() -> int:
    if os.path.exists(PATHS.CLEAN_CSV):
        log.info("✅ Clean data already exists. Skipping enrichment to save time.")
    return 0
    parser = argparse.ArgumentParser(description="Bookify preprocessing pipeline")
    parser.add_argument("--skip-enrichment", action="store_true",
                        help="Skip Google Books API description enrichment (~13 min)")
    args = parser.parse_args()

    log.info("=" * 58)
    log.info("  Bookify – Preprocessing Pipeline")
    log.info("=" * 58)

    os.makedirs(PATHS.OUTPUTS, exist_ok=True)
    for path in (PATHS.GOODREADS_CSV, PATHS.GOOGLE_CSV):
        if not os.path.exists(path):
            log.error("Data file not found: %s", path)
            return 1

    # ── 1. Load ────────────────────────────────────────────────────────────
    df1 = load_goodreads(PATHS.GOODREADS_CSV)
    df2 = load_google(PATHS.GOOGLE_CSV)

    # ── 2. Merge & exact dedup ─────────────────────────────────────────────
    df = pd.concat([df1, df2], ignore_index=True)
    before_exact = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    log.info("Exact dedup: %d → %d (%d removed)",
             before_exact, len(df), before_exact - len(df))

    # ── FIX #2: Normalised fuzzy dedup ────────────────────────────────────
    df["_title_key"] = df["title"].apply(normalise_title)
    before_fuzzy = len(df)
    df = df.drop_duplicates(subset=["_title_key"], keep="first").reset_index(drop=True)
    df = df.drop(columns=["_title_key"])
    removed_fuzzy = before_fuzzy - len(df)
    if removed_fuzzy:
        log.info("Fuzzy dedup:  %d → %d (%d near-duplicates removed)",
                 before_fuzzy, len(df), removed_fuzzy)
    else:
        log.info("Fuzzy dedup:  no additional duplicates found")

    # ── FIX #5: Domain filter ──────────────────────────────────────────────
    before_filter = len(df)
    df = df[df.apply(
        lambda r: is_engineering_book(r["title"], r["description"]), axis=1
    )].reset_index(drop=True)
    log.info("Domain filter: %d → %d (%d non-engineering removed)",
             before_filter, len(df), before_filter - len(df))

    # ── FIX #4: Enrich missing descriptions ───────────────────────────────
    if not args.skip_enrichment:
        df = enrich_missing_descriptions(df)
    else:
        log.info("Skipping description enrichment (--skip-enrichment)")

    # ── Category assignment ────────────────────────────────────────────────
    log.info("Assigning categories…")
    df["category"] = df.apply(
        lambda r: assign_category(r["title"], r["description"]), axis=1
    )
    for cat, count in df["category"].value_counts().items():
        log.info("  %-26s %d", cat, count)

    # ── Validate ───────────────────────────────────────────────────────────
    log.info("Validating…")
    if not validate(df):
        log.error("Validation failed.")
        return 1

    df.to_csv(PATHS.CLEAN_CSV, index=False)
    log.info("Saved → %s  (%d rows)", PATHS.CLEAN_CSV, len(df))
    log.info("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
