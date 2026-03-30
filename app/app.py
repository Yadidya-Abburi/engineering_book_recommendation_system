from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)

# --- SMART PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'outputs'))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))

# --- IN-MEMORY CACHE ---
_cache = {"books": None, "recs": None, "loaded_at": None}


def _load_cache():
    """Load data into memory once at startup, not per-request."""
    if _cache["books"] is not None:
        return

    # Load top_books.csv
    csv_path = os.path.join(OUTPUTS_DIR, 'top_books.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).fillna('')
        _cache["books"] = df.to_dict(orient='records')
    else:
        _cache["books"] = []
        print(f"WARNING: {csv_path} not found")

    # Load recommendations.json
    json_path = os.path.join(OUTPUTS_DIR, 'recommendations.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            _cache["recs"] = json.load(f)
    else:
        _cache["recs"] = {}
        print(f"WARNING: {json_path} not found")

    _cache["loaded_at"] = datetime.now().isoformat()
    print(f"  Cached {len(_cache['books'])} books, {len(_cache['recs'])} rec entries")


@app.before_request
def ensure_cache():
    _load_cache()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/books')
def get_books():
    """Get all books, optionally filtered by domain."""
    domain = request.args.get('domain')
    books = _cache["books"]

    if not books:
        return jsonify({"error": "No book data loaded", "hint": "Run: python scripts/train_model.py"}), 404

    if domain:
        books = [b for b in books if domain.lower() in str(b.get('category', '')).lower()]

    return jsonify(books)


@app.route('/api/recs')
def get_recs():
    """Get ML-powered recommendations for all books."""
    recs = _cache["recs"]
    if not recs:
        return jsonify({"error": "No recommendations loaded", "hint": "Run: python scripts/train_model.py"}), 404
    return jsonify(recs)


@app.route('/api/search')
def search_books():
    """Server-side search with fuzzy matching."""
    q = request.args.get('q', '').strip().lower()
    if not q:
        return jsonify([])

    results = []
    for b in _cache["books"]:
        title = str(b.get('title', '')).lower()
        author = str(b.get('author', '')).lower()
        category = str(b.get('category', '')).lower()
        text = f"{title} {author} {category}"

        if q in text:
            score = 1.0
        else:
            # Word-prefix match
            words = text.split()
            score = 0.8 if any(w.startswith(q) or q.startswith(w) for w in words) else 0
        if score > 0:
            results.append({**b, "_score": score})

    results.sort(key=lambda x: x["_score"], reverse=True)
    # Remove internal score before returning
    for r in results:
        r.pop("_score", None)
    return jsonify(results)


@app.route('/api/health')
def health():
    """Health check with model stats."""
    books = _cache["books"]
    recs = _cache["recs"]
    categories = {}
    for b in books:
        cat = b.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1

    return jsonify({
        "status": "ok",
        "books_loaded": len(books),
        "recommendations_loaded": len(recs),
        "categories": categories,
        "cached_at": _cache["loaded_at"],
        "data_source": OUTPUTS_DIR,
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("------------------------------------------")
    print("  Bookify Professional UI Engine")
    print(f"  Data Source: {OUTPUTS_DIR}")
    print("------------------------------------------")
    _load_cache()
    app.run(host='0.0.0.0', port=8765, debug=True)