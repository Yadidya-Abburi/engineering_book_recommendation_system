import sys
import os
import json
import numpy as np

# Mocking the BM25Scorer because we want to test its logic
class BM25Scorer:
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
        # We use a standard BM25 IDF: log((N - DF + 0.5) / (DF + 0.5) + 1)
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

def test_relevance():
    corpus = [
        "A textbook on algorithms and data structures for beginners.",
        "Advanced deep learning and neural networks for computer vision.",
        "Network Security: A comprehensive guide to cryptography.",
        "Computer Networking: A Top-Down Approach Featuring the Internet.",
        "The art of computer programming by Donald Knuth."
    ]
    
    scorer = BM25Scorer(corpus)
    
    # Test 1: Generic term "textbook"
    print("\n--- Search: 'textbook' ---")
    scores = scorer.get_scores("textbook")
    for i, s in enumerate(scores):
        if s > 0: print(f"  Score: {s:.4f} | Doc: {corpus[i]}")

    # Test 2: Specific term "cryptography" (which maps to security)
    print("\n--- Search: 'cryptography' ---")
    scores = scorer.get_scores("cryptography")
    for i, s in enumerate(scores):
        if s > 0: print(f"  Score: {s:.4f} | Doc: {corpus[i]}")

    # Test 3: Multiple terms "computer programming"
    print("\n--- Search: 'computer programming' ---")
    scores = scorer.get_scores("computer programming")
    # Sort docs by score
    indexed_scores = [(s, corpus[i]) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[0], reverse=True)
    for s, doc in indexed_scores:
        if s > 0: print(f"  Score: {s:.4f} | Doc: {doc}")

if __name__ == "__main__":
    test_relevance()
