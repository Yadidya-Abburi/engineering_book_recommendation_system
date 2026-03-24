from flask import Flask, render_template, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load your REAL ML data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
books = pickle.load(open(os.path.join(data_path, 'books.pkl'), 'rb'))
popular_df = pickle.load(open(os.path.join(data_path, 'popular.pkl'), 'rb'))

@app.route('/')
def index():
    # This renders your HTML file
    return render_template('index.html')

@app.route('/api/books')
def get_books():
    # This sends your ACTUAL processed books to the frontend
    return jsonify(popular_df.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True, port=8765)
