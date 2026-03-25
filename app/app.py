from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os

app = Flask(__name__)

# --- SMART PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check both folders just in case
OUTPUTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'outputs'))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))

def find_file(filename):
    """Checks outputs folder first, then data folder."""
    out_path = os.path.join(OUTPUTS_DIR, filename)
    data_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(out_path):
        return out_path
    if os.path.exists(data_path):
        return data_path
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/books')
def get_books():
    """Unified route to get all books OR filter by domain."""
    domain = request.args.get('domain') # Get ?domain=Software from URL if present
    target = find_file('top_books.csv')
    
    if target:
        df = pd.read_csv(target)
        # Fix NaNs to prevent JSON errors in the browser
        df = df.fillna('') 
        
        # Convert to list of dictionaries
        data = df.to_dict(orient='records')
        
        # If a domain was requested in the URL, filter the list
        if domain:
            # Note: matches 'category' column to match your index.html logic
            filtered_data = [b for b in data if domain.lower() in str(b.get('category', '')).lower()]
            return jsonify(filtered_data)
        
        return jsonify(data)
    
    print("❌ API Error: top_books.csv not found")
    return jsonify([])

@app.route('/api/recs')
def get_recs():
    target = find_file('recs.json')
    if target:
        with open(target, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({})

if __name__ == '__main__':
    print("------------------------------------------")
    print("🚀 Bookify Professional UI Engine")
    print(f"📂 Primary Data Source: {OUTPUTS_DIR}")
    print("------------------------------------------")
    # Using port 8765 as per your previous setup
    app.run(host='0.0.0.0', port=8765, debug=True)