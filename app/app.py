from flask import Flask, render_template, jsonify, request
import json
import os
import requests
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- SMART PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'outputs'))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))

# --- IN-MEMORY CACHE ---
_cache = {
    "books": None, 
    "recs": None, 
    "vectorizer": None,
    "matrix": None,
    "bm25_scorer": None, # [NEW] Added BM25 Scorer
    "loaded_at": None
}


def _file_mtime(path):
    """Return a file's mtime or None when it does not exist."""
    return os.path.getmtime(path) if os.path.exists(path) else None

# --- BM25 SCORER IMPLEMENTATION ---
class BM25Scorer:
    """Lightweight, optimized BM25 algorithm for search ranking."""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.avgdl = sum(len(d.split()) for d in corpus) / self.n if self.n > 0 else 0
        self.corpus = [d.lower().split() for d in corpus]
        self.df = self._compute_df()
        self.idf = self._compute_idf()

    def _compute_df(self):
        df = {}
        for doc in self.corpus:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        return df

    def _compute_idf(self):
        return {word: np.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0) for word, freq in self.df.items()}

    def get_scores(self, query):
        q_words = query.lower().split()
        scores = np.zeros(self.n)
        for doc_idx, doc in enumerate(self.corpus):
            doc_len = len(doc)
            for word in q_words:
                if word in self.idf:
                    freq = doc.count(word)
                    num = self.idf[word] * freq * (self.k1 + 1)
                    den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += num / den
        return scores


def _load_cache():
    """Load data and ML models into memory once at startup."""
    _cache["bm25_scorer"] = None
    csv_candidates = [
        os.path.join(OUTPUTS_DIR, 'books_enriched.csv'),
        os.path.join(OUTPUTS_DIR, 'books_clean.csv'),
        os.path.join(OUTPUTS_DIR, 'books_ultimate.csv'),
        os.path.join(OUTPUTS_DIR, 'top_books.csv'),
        os.path.join(DATA_DIR, 'final_books.csv'),
    ]
    csv_path = next((path for path in csv_candidates if os.path.exists(path)), csv_candidates[-1])
    json_path = os.path.join(OUTPUTS_DIR, 'recommendations.json')

    current_books_mtime = _file_mtime(csv_path)
    current_recs_mtime = _file_mtime(json_path)

    if (
        _cache["books"] is not None
        and _cache.get("books_source_path") == csv_path
        and _cache.get("books_source_mtime") == current_books_mtime
        and _cache.get("recs_source_mtime") == current_recs_mtime
    ):
        return

    # 1. Load Books (Priority: enriched outputs > committed outputs > raw merged data)
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path).fillna('')
        _cache["books"] = df.to_dict(orient='records')
        
        # Pre-compute short descriptions and fix column mapping
        for b in _cache["books"]:
            # 1. Description mapping (desc -> description)
            d = str(b.get('description', '')) or str(b.get('desc', ''))
            b['description'] = d
            b['desc_short'] = d[:300] if len(d) > 300 else d
            
            # 2. Cover mapping (thumbnail -> cover_url)
            if not b.get('cover_url') and b.get('thumbnail'):
                b['cover_url'] = b['thumbnail']
            
            # 3. Category mapping (categories -> category)
            raw_cat = str(b.get('categories', '') or b.get('category', 'Unknown'))
            if raw_cat.startswith('['):
                try: 
                    import json as j
                    raw_cat = j.loads(raw_cat.replace("'",'"'))[0]
                except: 
                    raw_cat = raw_cat.strip("[]'\"")
            b['category'] = raw_cat or "General Engineering"
            
        print(f"  Loaded {len(_cache['books'])} books from {os.path.basename(csv_path)}")
    else:
        _cache["books"] = []
        print(f"WARNING: No data found at {csv_path}")

    # 2. Load recommendations.json
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            _cache["recs"] = json.load(f)
    else:
        _cache["recs"] = {}
        print(f"WARNING: {json_path} not found")

    # 3. Build BM25 / Semantic Index
    contents = [
        f"{b.get('title','')} {b.get('author','')} {b.get('desc_short','')}" 
        for b in _cache["books"]
    ]
    
    if len(contents) > 0:
        _cache["bm25_scorer"] = BM25Scorer(contents)
        print(f"  BM25 Search Index built for {len(contents)} books")
    
    _cache["loaded_at"] = datetime.now().isoformat()
    _cache["books_source_path"] = csv_path
    _cache["books_source_mtime"] = current_books_mtime
    _cache["recs_source_mtime"] = current_recs_mtime
    print(f"  Cached {len(_cache['books'])} books")


@app.before_request
def ensure_cache():
    _load_cache()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/books')
def get_books():
    """Get books with pagination and domain filtering."""
    domain = request.args.get('domain')
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 200))
    
    books = _cache["books"]
    if not books:
        return jsonify({"error": "No data", "hint": "Run pipeline"}), 404

    if domain:
        books = [b for b in books if domain.lower() in str(b.get('category', '')).lower()]

    # Pagination
    total = len(books)
    start = (page - 1) * limit
    end = start + limit
    
    return jsonify({
        "books": books[start:end],
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit
    })


@app.route('/api/recs')
def get_recs():
    """Get ML-powered recommendations for all books."""
    recs = _cache["recs"]
    if not recs:
        return jsonify({"error": "No recommendations loaded", "hint": "Run: python scripts/train_model.py"}), 404
    return jsonify(recs)


@app.route('/api/search')
def search_books():
    """Optimized Search using BM25 Algorithm."""
    q = request.args.get('q', '').strip().lower()
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    
    if not q:
        return jsonify({"books": [], "total": 0})

    books = _cache["books"]
    scorer = _cache["bm25_scorer"]

    if not scorer:
        # Fallback to simple matching if scorer missing
        results = [b for b in books if q in b.get('title','').lower() or q in b.get('author','').lower()]
        return jsonify({"books": results[:limit], "total": len(results), "engine": "fallback"})

    # 1. Compute BM25 scores
    bm25_scores = scorer.get_scores(q)
    
    # 2. Add Title Boost & Combine
    results = []
    for i, base_score in enumerate(bm25_scores):
        if base_score > 0:
            b = books[i].copy()
            title = str(b.get('title','')).lower()
            
            final_score = base_score
            if q in title: final_score *= 1.5 # Boost exact title match
            elif any(w in title for w in q.split() if len(w) > 3): final_score *= 1.2
            
            b["_score"] = float(final_score)
            results.append(b)

    # 3. Sort and Paginate
    results.sort(key=lambda x: x["_score"], reverse=True)
    
    total = len(results)
    start = (page - 1) * limit
    end = start + limit
    
    final_books = results[start:end]
    for b in final_books:
        b["match_pct"] = min(99, int((b["_score"] / 10) * 100)) if b["_score"] < 10 else 99
        b.pop("_score", None)

    return jsonify({
        "books": final_books,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
        "engine": "bm25"
    })


@app.route('/api/toc')
def get_toc():
    """Fetch TOC dynamically from Google Books API if missing."""
    title = request.args.get('title', '')
    author = request.args.get('author', '')
    if not title:
        return jsonify({"items": []})
        
    try:
        q = f'intitle:"{title}"'
        if author:
            q += f'+inauthor:"{author}"'
            
        url = f'https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=1'
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            if items:
                vinfo = items[0].get('volumeInfo', {})
                toc = vinfo.get('tableOfContents', [])
                if toc:
                    return jsonify({"items": toc})
                
                desc = vinfo.get('description', '')
                if desc:
                    import re
                    clean_desc = re.sub('<[^<]+>', '', desc)
                    # Try to find actual bullet points if present
                    if '•' in clean_desc:
                        parts = [p.strip() for p in clean_desc.split('•') if len(p.strip()) > 10]
                        return jsonify({"items": parts})
                        
                    # Split into sentences
                    parts = [s.strip() for s in clean_desc.split('. ') if len(s.strip()) > 15]
                    if len(parts) > 0:
                        parts = [p + ('.' if not p.endswith('.') else '') for p in parts]
                        return jsonify({"items": parts[:6]})
                        
                    if len(clean_desc) > 10:
                        return jsonify({"items": [clean_desc]})
                        
    except Exception as e:
        print(f"TOC Fetch Error: {e}")
        pass
        
    return jsonify({"items": []})


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