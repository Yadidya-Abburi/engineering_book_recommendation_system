import os
import sys
import json
import time
import requests
import pandas as pd
import re
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, 'data', 'final_books.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'outputs', 'books_ultimate.csv')

# --- STOP LISTS FOR HIGH PRECISION ---
STOP_MSGS = [
    'this book', 'the author', 'learn how', 'includes', 'overview', 'preface',
    'introduction', 'conclusion', 'summary', 'about this', 'copyright', 'isbn',
    'publisher', 'apress', 'springer', 'wiley', 'national workshop', 'proceedings',
    'conference', 'symposium', 'edition', 'seminar', 'meeting', 'vol.', 'issue'
]

TECH_ANCHORS = [
    'circuit', 'network', 'algorithm', 'intelligence', 'neural', 'design', 'architecture',
    'model', 'system', 'data', 'protocol', 'layer', 'synthesis', 'calculus', 'physics',
    'logic', 'program', 'security', 'crypto', 'signal', 'machine', 'learning', 'deep',
    'cloud', 'database', 'structure', 'engineering', 'mechanical', 'solid', 'fluid'
]

def is_valid_topic(topic, author):
    """Deep verification of topic quality."""
    if not topic or len(topic) < 4: return False
    low = topic.lower().strip()
    
    # 1. Author Name Check (Entity Guard)
    if author and author.lower() in low: return False
    
    # 2. Noise & Event Filter
    for msg in STOP_MSGS:
        if msg in low: return False
        
    # 3. Scientific Anchor Check (Required if NLP curated)
    has_anchor = any(a in low for a in TECH_ANCHORS)
    if not has_anchor and len(topic.split()) < 3: # Allow long specific sentences but block short noise
        return False
        
    return True

# --- API FETCHERS ---
def fetch_open_library_toc(title, author):
    """Fetch official TOC from Open Library Work ID."""
    try:
        q = f"title={requests.utils.quote(title)}&author={requests.utils.quote(author)}"
        search = requests.get(f"https://openlibrary.org/search.json?{q}&limit=1", timeout=5).json()
        if not search.get('docs'): return None
        
        work_id = search['docs'][0].get('key') # e.g., /works/OL123W
        if not work_id: return None
        
        work = requests.get(f"https://openlibrary.org{work_id}.json", timeout=5).json()
        toc = work.get('table_of_contents', [])
        
        # Open Library TOC can be complex objects
        items = []
        for t in toc:
            label = t.get('title') if isinstance(t, dict) else str(t)
            if is_valid_topic(label, author):
                items.append(label)
        
        return items[:12] if items else None
    except: return None

def fetch_wiki_curated(query, author):
    """High-accuracy Wikipedia subject mapping."""
    try:
        # Strip title to core concept
        core = re.sub(r'^(Intro|Principles|Fundamentals|Handbook|Advanced)\s+', '', query, flags=re.I)
        core = core.split(':')[0].strip()
        
        s_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={requests.utils.quote(core)}&limit=1&format=json"
        s_res = requests.get(s_url, timeout=5).json()
        if not s_res[1]: return None
        
        page = s_res[1][0]
        sec_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={requests.utils.quote(page)}&prop=sections&format=json"
        sec_res = requests.get(sec_url, timeout=5).json()
        
        sections = sec_res.get('parse', {}).get('sections', [])
        BAD = ['see also', 'references', 'external links', 'further reading', 'notes', 'history']
        
        clean = [s['line'] for s in sections if s['level'] == '2' and s['line'].lower() not in BAD]
        valid = [s for s in clean if is_valid_topic(s, author)]
        
        return valid[:10] if valid else None
    except: return None

def ultimate_syllabus_engine(title, author, description):
    """The Gold Standard Decision Path."""
    # Priority 1: Open Library Official TOC
    print(f"  - Trying Open Library TOC...")
    ol_toc = fetch_open_library_toc(title, author)
    if ol_toc: return ol_toc, "Official TOC (OL)"
    
    # Priority 2: Wikipedia Subject Curated
    print(f"  - Trying Wikipedia Synthesis...")
    wiki_toc = fetch_wiki_curated(title, author)
    if wiki_toc: return wiki_toc, "Curated (Wiki)"
    
    # Priority 3: Fallback - Static subjects based on domain
    return ["Introduction", "Core Theory", "System Design", "Methods & Models", "Future Trends"], "Curated (Static)"

def main():
    print(f"🏆 Starting THE ULTIMATE SYLLABUS GENERATOR")
    df = pd.read_csv(INPUT_CSV).fillna('')
    ultimate_data = []

    # Filter: Only re-enrich if current data is 'NLP Curated' or 'Static'
    # Actually, user wants a "total file" reset for quality.
    for idx, row in df.iterrows():
        title = row['title']
        author = row.get('author', 'Unknown')
        desc = row.get('description', '')
        
        print(f"[{idx+1}/{len(df)}] 🛠️ Building: {title[:50]}...")
        
        new_contents, new_source = ultimate_syllabus_engine(title, author, desc)
        
        row['contents'] = json.dumps(new_contents)
        row['toc_source'] = new_source
        ultimate_data.append(row)
        
        # Save every 50 to avoid loss
        if (idx + 1) % 50 == 0:
            pd.DataFrame(ultimate_data).to_csv(OUTPUT_CSV, index=False)
            time.sleep(0.5)

    pd.DataFrame(ultimate_data).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ COMPLETED! Gold Standard Database saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
