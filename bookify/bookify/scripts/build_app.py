"""
build_app.py
────────────
Reads trained model outputs, pre-fetches book cover images (FIX #6),
and bakes everything into a self-contained app/index.html.

FIX #6: Cover images are fetched at build time from Open Library API and
embedded as b.image in the BOOKS array. The app shows covers immediately
with no runtime API dependency. If Open Library is unavailable during build,
the field is left empty and the app falls back to live fetch at runtime.

Run:
    python scripts/build_app.py
    python scripts/build_app.py --skip-covers   # fast rebuild without re-fetching

Exit code 0 = success, 1 = failure.
"""

import argparse
import json
import logging
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS, MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

APP_COLS = [
    "title", "author", "publisher", "year", "pages",
    "rating", "ratings_count", "score", "desc_short",
    "image", "category", "source", "platform",
    "download_link", "file_info", "price", "currency",
    "preview_link", "info_link",
]

# ── FIX #6: build-time cover pre-fetcher ──────────────────────────────────

def fetch_cover_url(title: str, author: str) -> str:
    """
    Query Open Library search API and return the best cover URL.
    Returns empty string on failure — app will fall back to runtime fetch.
    Rate-limited to avoid hitting the 100 req/min Open Library limit.
    """
    import requests  # lazy import — only needed during cover pre-fetch
    try:
        q   = f"{title[:50]} {author[:30]}".strip()
        url = (
            "https://openlibrary.org/search.json"
            f"?q={requests.utils.quote(q)}&limit=1&fields=cover_i,isbn"
        )
        r = requests.get(url, timeout=6)
        if not r.ok:
            return ""
        doc = (r.json().get("docs") or [None])[0]
        if not doc:
            return ""
        if doc.get("cover_i"):
            return f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg"
        isbns = doc.get("isbn") or []
        if isbns:
            return f"https://covers.openlibrary.org/b/isbn/{isbns[0]}-M.jpg"
    except Exception:
        pass
    return ""


def prefetch_covers(books: list[dict], delay: float = 0.7) -> list[dict]:
    """
    For every book with an empty image field, fetch the cover URL from
    Open Library and embed it. Books that already have a valid cover URL
    are skipped.

    delay: seconds between requests (0.7 keeps us well under 100 req/min).
    """
    need_fetch = [b for b in books if not b.get("image")]
    already    = len(books) - len(need_fetch)
    log.info("Cover pre-fetch: %d already have URLs, fetching %d…", already, len(need_fetch))

    fetched = 0
    failed  = 0
    for i, b in enumerate(need_fetch):
        url = fetch_cover_url(b["title"], b.get("author", ""))
        if url:
            b["image"] = url
            fetched += 1
        else:
            failed += 1
        if (i + 1) % 25 == 0:
            log.info("  … %d / %d  (found: %d, not found: %d)",
                     i + 1, len(need_fetch), fetched, failed)
        time.sleep(delay)

    log.info("Cover pre-fetch complete: %d found, %d not found", fetched, failed)
    return books


# ── Load & clean model outputs ─────────────────────────────────────────────

def load_model_outputs() -> tuple[list[dict], dict]:
    missing = [p for p in (PATHS.TOP_BOOKS_CSV, PATHS.RECS_JSON) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Model outputs missing: {missing}\n"
            "Run  python scripts/train_model.py  first."
        )

    top = pd.read_csv(PATHS.TOP_BOOKS_CSV)
    top["rating"]        = pd.to_numeric(top["rating"],        errors="coerce").fillna(0)
    top["ratings_count"] = pd.to_numeric(top["ratings_count"], errors="coerce").fillna(0).astype(int)
    top["year"]          = pd.to_numeric(top["year"],          errors="coerce").fillna(0).astype(int)
    top["pages"]         = pd.to_numeric(top["pages"],         errors="coerce").fillna(0).astype(int)
    top["price"]         = pd.to_numeric(
        top["price"] if "price" in top.columns else pd.Series([0.0] * len(top)),
        errors="coerce"
    ).fillna(0.0)
    top["score"]         = pd.to_numeric(top["score"],         errors="coerce").fillna(0).round(4)
    top["desc_short"]    = top["desc_short"].fillna("").str.strip()

    # Ensure all optional columns exist and are clean
    for col in ["image", "download_link", "file_info", "preview_link", "info_link", "platform"]:
        if col not in top.columns:
            top[col] = ""
        top[col] = top[col].fillna("").astype(str).replace("nan", "")
    top["currency"] = top.get("currency", pd.Series(["USD"] * len(top))).fillna("USD")

    existing = [c for c in APP_COLS if c in top.columns]
    books = top[existing].to_dict(orient="records")

    with open(PATHS.RECS_JSON, encoding="utf-8") as f:
        recs = json.load(f)

    return books, recs


# ── HTML injection ─────────────────────────────────────────────────────────

def read_template() -> str:
    tpl = os.path.join(PATHS.APP, "template.html")
    idx = PATHS.APP_HTML
    if os.path.exists(tpl):
        with open(tpl, encoding="utf-8") as f:
            return f.read()
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(
        "No app/template.html or app/index.html found."
    )


def inject_data(template: str, books: list[dict], recs: dict) -> str:
    books_json = json.dumps(books, separators=(",", ":"), ensure_ascii=False)
    recs_json  = json.dumps(recs,  separators=(",", ":"), ensure_ascii=False)
    html = template.replace("__BOOKS_JSON__", books_json).replace("__RECS_JSON__", recs_json)
    # Guard: warn if pattern appears more than once (greedy DOTALL could match wrong range)
    for pat, label in [(r"const BOOKS\s*=\s*\[", "BOOKS"), (r"const RECS\s*=\s*\{", "RECS")]:
        if len(re.findall(pat, html)) > 1:
            log.warning("Pattern '%s' found multiple times — injection may be incorrect", label)
    html = re.sub(r"const BOOKS\s*=\s*\[.*?\];",
                  f"const BOOKS={books_json};", html, flags=re.DOTALL)
    html = re.sub(r"const RECS\s*=\s*\{.*?\};",
                  f"const RECS={recs_json};",  html, flags=re.DOTALL)
    return html


def validate_output(html: str) -> bool:
    checks = {
        "BOOKS const present"   : "const BOOKS=" in html,
        "RECS const present"    : "const RECS="  in html,
        "getPlatforms present"  : "getPlatforms" in html,
        "download_link present" : "download_link" in html,
        "platform field present": "platform" in html,
    }
    ok = True
    for name, result in checks.items():
        if not result:
            log.error("Validation FAIL: %s", name)
            ok = False
        else:
            log.info("  ✓ %s", name)
    return ok


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-covers", action="store_true",
                        help="Skip cover pre-fetching (fast rebuild)")
    args = parser.parse_args()

    log.info("=" * 58)
    log.info("  Bookify – Building Web App")
    log.info("=" * 58)

    try:
        os.makedirs(PATHS.APP, exist_ok=True)

        log.info("Loading model outputs…")
        books, recs = load_model_outputs()
        log.info("  %d books, %d rec entries", len(books), len(recs))

        # ── FIX #6: pre-fetch covers at build time ─────────────────────────
        if not args.skip_covers:
            books = prefetch_covers(books)
            cover_count = sum(1 for b in books if b.get("image"))
            log.info("Cover coverage: %d / %d  (%.0f%%)",
                     cover_count, len(books), cover_count / len(books) * 100)
        else:
            log.info("Skipping cover pre-fetch (--skip-covers)")

        log.info("Reading HTML template…")
        template = read_template()

        log.info("Injecting data…")
        html = inject_data(template, books, recs)

        log.info("Validating output…")
        if not validate_output(html):
            return 1

        with open(PATHS.APP_HTML, "w", encoding="utf-8") as f:
            f.write(html)

        size_kb = os.path.getsize(PATHS.APP_HTML) // 1024
        log.info("Built → %s  (%d KB)", PATHS.APP_HTML, size_kb)
        log.info("Open app/index.html in any browser to launch Bookify.")

    except Exception as exc:
        log.exception("Build failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
