from flask import Flask, render_template, jsonify, request
import json
import os
import requests
import joblib
import numpy as np
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# --- GLOBAL NOISE FILTER ---
STOP_MSGS = ['this book', 'the author', 'learn how', 'includes', 'overview', 'preface', 'introduction', 'conclusion', 'summary']
def is_valid_topic(topic):
    if not topic or len(topic) < 4: return False
    low = topic.lower().strip()
    return not any(msg in low for msg in STOP_MSGS)

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
        q_words = query.lower().replace('-', ' ').replace('/', ' ').split()
        scores = np.zeros(self.n)
        for doc_idx, doc in enumerate(self.corpus):
            doc_len = len(doc)
            doc_str = ' '.join(doc)
            for word in q_words:
                # 1. Exact Word Match (Classic BM25)
                if word in self.idf:
                    freq = doc.count(word)
                    num = self.idf[word] * freq * (self.k1 + 1)
                    den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += num / den
                
                # 2. Substring Match Boost (for partials like 'crypto')
                if len(word) > 3 and word in doc_str:
                    scores[doc_idx] += 0.5 # Subtle boost for partial matches
        return scores


def _load_cache():
    """Load data and ML models into memory once at startup."""
    if _cache["books"] is not None:
        return

    # 1. Load Skeleton (The Full 3,293-book collection)
    f_p = os.path.join(DATA_DIR, 'final_books.csv')
    if os.path.exists(f_p):
        import pandas as pd
        df = pd.read_csv(f_p).fillna('')
        
        # Mapping column names if needed (e.g., thumbnail -> cover_url)
        if 'cover_url' not in df.columns and 'thumbnail' in df.columns:
            df['cover_url'] = df['thumbnail']
            
        # 2. Overlay Enrichment (The High-Accuracy Syllabuses)
        u_p = os.path.join(OUTPUTS_DIR, 'books_ultimate.csv')
        if os.path.exists(u_p):
            try:
                du = pd.read_csv(u_p).fillna('')
                # Merge: Take 'contents' and 'toc_source' from the Ultimate file
                # Use 'title' as the unique key for matching
                df = df.set_index('title')
                du = du.set_index('title')
                df.update(du[['contents', 'toc_source']])
                df = df.reset_index()
                print(f"  Merged {len(du)} high-accuracy syllabuses into the full corpus")
            except Exception as e:
                print(f"  Enrichment merge skipped: {e}")
            
        _cache["books"] = df.to_dict(orient='records')
        
        # Pre-compute short descriptions for index
        for b in _cache["books"]:
            desc = str(b.get('description', ''))
            b['desc_short'] = desc[:300] if len(desc) > 300 else desc
            
        print(f"  Successfully loaded the full 3,293-book library")
    else:
        _cache["books"] = []
        print(f"CRITICAL ERROR: {f_p} not found!")

    # 2. Load recommendations.json
    json_path = os.path.join(OUTPUTS_DIR, 'recommendations.json')
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
    if request.args.get('verified') == 'true':
        books = [b for b in books if b.get('toc_source') == 'Official TOC']
        total = len(books)

    start = (page - 1) * limit
    end = start + limit
    
    return jsonify({
        "books": books[start:end],
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit
    })


@app.route('/api/discover')
def get_discover():
    """Dynamic discovery feed including history and trending."""
    viewed = request.args.getlist('viewed[]')
    books = _cache["books"] or []
    
    # 1. History (if provided)
    history = [b for b in books if b['title'] in viewed]
    history = history[::-1][:10] # Reverse to get latest
    
    # 3. Domain Shelves with counts
    all_categories = {}
    for b in books:
        c = b.get('category', 'Unknown')
        all_categories[c] = all_categories.get(c, 0) + 1

    domains = ["AI/ML", "Programming", "Security", "Networking"]
    shelves = []
    import random
    for d in domains:
        d_books = [b for b in books if d.lower() in str(b.get('category','')).lower()]
        if d_books:
            shelves.append({
                "title": f"Explore {d}",
                "books": random.sample(d_books, min(len(d_books), 10))
            })
            
    return jsonify({
        "history": history,
        "shelves": shelves,
        "categories": all_categories
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
                
                    # Priority 1: Try Open Library (Official)
                    try:
                        q_ol = f"title={requests.utils.quote(vinfo['title'])}&limit=1"
                        ol_res = requests.get(f"https://openlibrary.org/search.json?{q_ol}", timeout=3).json()
                        if ol_res.get('docs'):
                            work_id = ol_res['docs'][0].get('key')
                            work = requests.get(f"https://openlibrary.org{work_id}.json", timeout=3).json()
                            toc = work.get('table_of_contents', [])
                            if toc:
                                items = [t.get('title') if isinstance(t, dict) else str(t) for t in toc]
                                items = [i for i in items if is_valid_topic(i)]
                                if items: return jsonify({"items": items[:12], "source": "Official TOC (OL)"})
                    except: pass

                    # Priority 2: Google Description Bullets
                    desc = vinfo.get('description', '')
                    if desc:
                        clean_desc = re.sub('<[^<]+>', '', desc)
                        parts = [s.strip() for s in clean_desc.split('. ') if len(s.strip()) > 15]
                        parts = [p for p in parts if is_valid_topic(p)]
                        if len(parts) > 0:
                            return jsonify({"items": parts[:8], "source": "Official TOC (G)"})
                        
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