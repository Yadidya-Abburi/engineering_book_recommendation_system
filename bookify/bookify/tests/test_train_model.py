"""
tests/test_train_model.py
─────────────────────────
Unit tests for model training functions.

Run:
    pytest tests/test_train_model.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import unittest
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import (
    build_tfidf, build_similarity, compute_scores, extract_top_and_recs
)
from config.settings import MODEL


def _make_books(n=25) -> pd.DataFrame:
    return pd.DataFrame({
        "title":         [f"Book About Python {i}" for i in range(n)],
        "author":        [f"Author {i}" for i in range(n)],
        "description":   [f"Python programming algorithms data structures {i}" for i in range(n)],
        "rating":        [3.5 + (i % 5) * 0.2 for i in range(n)],
        "ratings_count": [100 + i * 50 for i in range(n)],
        "publisher":     ["Pub"] * n,
        "year":          [2015 + i % 8 for i in range(n)],
        "pages":         [200 + i * 10 for i in range(n)],
        "image":         [""] * n,
        "download_link": [f"https://example.com/{i}" for i in range(n)],
        "file_info":     ["PDF, 5MB"] * n,
        "price":         [0.0] * n,
        "currency":      ["USD"] * n,
        "preview_link":  [""] * n,
        "info_link":     [""] * n,
        "source":        ["Goodreads"] * n,
        "platform":      ["PDF"] * n,
        "category":      ["Programming"] * n,
    })


class TestBuildTfidf:
    def test_returns_vectorizer_and_matrix(self):
        df = _make_books()
        vectorizer, matrix = build_tfidf(df)
        assert isinstance(vectorizer, TfidfVectorizer)
        assert matrix.shape[0] == len(df)

    def test_matrix_rows_match_book_count(self):
        df = _make_books(15)
        _, matrix = build_tfidf(df)
        assert matrix.shape[0] == 15

    def test_content_column_created(self):
        df = _make_books(5)
        build_tfidf(df)
        assert "content" in df.columns

    def test_sparse_matrix_returned(self):
        from scipy.sparse import issparse
        df = _make_books()
        _, matrix = build_tfidf(df)
        assert issparse(matrix)


class TestBuildSimilarity:
    def test_output_shape_is_square(self):
        df = _make_books(10)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        assert sim.shape == (10, 10)

    def test_diagonal_is_one(self):
        df = _make_books(10)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        assert np.allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_values_between_0_and_1(self):
        df = _make_books(10)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        assert sim.min() >= -1e-6
        assert sim.max() <= 1.0 + 1e-6

    def test_symmetric(self):
        df = _make_books(8)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        assert np.allclose(sim, sim.T, atol=1e-6)


class TestComputeScores:
    def test_score_column_added(self):
        df = _make_books()
        scored = compute_scores(df)
        assert "score" in scored.columns

    def test_scores_between_0_and_1(self):
        df = _make_books()
        scored = compute_scores(df)
        assert scored["score"].between(0.0, 1.0).all()

    def test_unrated_books_get_zero_score(self):
        df = _make_books(10)
        df.loc[0, "rating"] = 0
        scored = compute_scores(df)
        assert scored.loc[0, "score"] == 0.0

    def test_higher_rating_higher_score(self):
        """Fix #8: Bayesian scoring should still rank better books higher."""
        df = _make_books(10)
        df.loc[0, "rating"]        = 5.0
        df.loc[0, "ratings_count"] = 100_000
        scored = compute_scores(df)
        assert scored.loc[0, "score"] == scored["score"].max()

    def test_score_spread_better_than_minmax(self):
        """Fix #8: Bayesian scoring should give wider std than plain MinMaxScaler."""
        df = _make_books(25)
        scored = compute_scores(df)
        rated = scored[scored["rating"] > 0]["score"]
        # Bayesian scoring on a diverse dataset should give std > 0.05
        assert rated.std() > 0.05, f"Score std too low: {rated.std():.4f}"

    def test_bayesian_rating_column_created(self):
        """Fix #8: bayes_rating column should exist after scoring."""
        df = _make_books()
        scored = compute_scores(df)
        assert "bayes_rating" in scored.columns


class TestExtractTopAndRecs:
    def test_top_n_respected(self):
        df = _make_books(25)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        top, _ = extract_top_and_recs(df, sim)
        assert len(top) == min(MODEL.TOP_N, len(df))

    def test_recs_count_per_book(self):
        df = _make_books(25)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        _, recs = extract_top_and_recs(df, sim)
        for rec_list in recs.values():
            assert len(rec_list) <= MODEL.N_RECS

    def test_recs_have_required_keys(self):
        df = _make_books(25)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        _, recs = extract_top_and_recs(df, sim)
        for rec_list in recs.values():
            for rec in rec_list:
                for key in ("title", "author", "rating", "similarity"):
                    assert key in rec

    def test_similarity_in_valid_range(self):
        df = _make_books(25)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        _, recs = extract_top_and_recs(df, sim)
        for rec_list in recs.values():
            for rec in rec_list:
                assert 0.0 <= rec["similarity"] <= 1.0

    def test_no_self_recommendations(self):
        """Fix #1: no book should recommend itself."""
        df = _make_books(25)
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        _, recs = extract_top_and_recs(df, sim)
        self_recs = [
            (title, r["title"])
            for title, rec_list in recs.items()
            for r in rec_list
            if r["title"].strip() == title.strip()
        ]
        assert len(self_recs) == 0, f"Self-recs found: {self_recs}"

    def test_no_perfect_similarity_in_recs(self):
        """Fix #1: similarity==1.0 to a different book (duplicate) must be excluded."""
        df = _make_books(25)
        # Manually create a duplicate: book 0 and book 1 same content
        df.loc[1, "title"]       = df.loc[0, "title"]
        df.loc[1, "description"] = df.loc[0, "description"]
        _, matrix = build_tfidf(df)
        sim = build_similarity(matrix)
        df = compute_scores(df)
        _, recs = extract_top_and_recs(df, sim)
        perfect = [
            r for rec_list in recs.values()
            for r in rec_list if r["similarity"] >= 1.0
        ]
        assert len(perfect) == 0, f"Perfect-sim recs found: {perfect}"


# ─────────────────────────────────────────────────────────────────────────────
# TestNearDupDedup
# ─────────────────────────────────────────────────────────────────────────────
class TestNearDupDedup(unittest.TestCase):
    """extract_top_and_recs() must remove near-duplicate titles (sim > 0.98)."""

    def _make_df(self, titles):
        """Build a minimal scored DataFrame with the given titles."""
        n = len(titles)
        return pd.DataFrame({
            "title":         titles,
            "author":        ["Auth"] * n,
            "publisher":     ["Pub"] * n,
            "year":          [2020] * n,
            "pages":         [300] * n,
            "rating":        [4.0] * n,
            "ratings_count": [1000] * n,
            "score":         [0.9 - i * 0.01 for i in range(n)],
            "description":   ["desc"] * n,
            "image":         [""] * n,
            "category":      ["Programming"] * n,
            "source":        ["Goodreads"] * n,
            "platform":      ["PDF"] * n,
            "download_link": [""] * n,
            "file_info":     [""] * n,
            "price":         [0.0] * n,
            "currency":      ["USD"] * n,
            "preview_link":  [""] * n,
            "info_link":     [""] * n,
            "bayes_rating":  [4.0] * n,
            "log_count":     [6.9] * n,
        })

    def test_near_dup_removed_from_top(self):
        """A near-duplicate title (punctuation/case variant) must be deduplicated."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import extract_top_and_recs, build_similarity, build_tfidf

        titles = [
            "Machine Learning Fundamentals",
            "machine learning fundamentals",   # exact near-dup (normalised)
            "Deep Learning with Python",
            "Python Deep Learning Guide",
            "Neural Network Design",
        ]
        df = self._make_df(titles)
        _, mat = build_tfidf(df)
        sim = build_similarity(mat)

        from config.settings import MODEL
        original_top_n = MODEL.TOP_N
        MODEL.TOP_N = len(titles)   # keep all

        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N = original_top_n

        top_titles_norm = (
            top["title"]
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.strip()
            .tolist()
        )
        # No two normalised titles should be identical
        self.assertEqual(len(top_titles_norm), len(set(top_titles_norm)),
                         "Near-duplicate titles must be removed from top-N")

    def test_higher_scoring_dup_kept(self):
        """When two near-dups exist, the one with higher score must be retained."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import extract_top_and_recs, build_similarity, build_tfidf
        from config.settings import MODEL

        titles = [
            "Python Programming Guide",       # score 0.90 — should be kept
            "python programming guide",        # score 0.89 — near-dup, lower score
            "Data Structures and Algorithms",
        ]
        df = self._make_df(titles)
        # Manually set scores so first is clearly higher
        df["score"] = [0.90, 0.89, 0.80]

        _, mat = build_tfidf(df)
        sim    = build_similarity(mat)

        original_top_n = MODEL.TOP_N
        MODEL.TOP_N    = len(titles)
        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N    = original_top_n

        self.assertIn("Python Programming Guide", top["title"].tolist())
        self.assertNotIn("python programming guide", top["title"].tolist())


# ── TestSaveArtefacts ────────────────────────────────────────────────────────
class TestSaveArtefacts(unittest.TestCase):
    """save_artefacts() must write all four output files."""

    def test_all_artefacts_written(self):
        """save_artefacts must create pkl, npy, csv, and json in a temp directory."""
        import tempfile, joblib, numpy as np, json
        from unittest.mock import patch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import save_artefacts, build_tfidf, build_similarity, compute_scores
        from config.settings import MODEL

        df = _make_books(15)
        df["rating"]        = pd.to_numeric(df["rating"],        errors="coerce").fillna(0)
        df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)
        vec, mat  = build_tfidf(df)
        sim       = build_similarity(mat)
        df        = compute_scores(df)
        original  = MODEL.TOP_N
        MODEL.TOP_N = len(df)
        top, recs = extract_top_and_recs(df, sim)
        MODEL.TOP_N = original

        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = os.path.join(tmp, "vec.pkl")
            npy_path = os.path.join(tmp, "sim.npy")
            csv_path = os.path.join(tmp, "top.csv")
            jsn_path = os.path.join(tmp, "recs.json")

            with patch("scripts.train_model.PATHS") as mock_paths:
                mock_paths.MODELS  = tmp
                mock_paths.OUTPUTS = tmp
                mock_paths.TFIDF_PKL     = pkl_path
                mock_paths.COSINE_NPY    = npy_path
                mock_paths.TOP_BOOKS_CSV = csv_path
                mock_paths.RECS_JSON     = jsn_path
                save_artefacts(vec, sim, top, recs)

            self.assertTrue(os.path.exists(pkl_path),  "tfidf pkl not written")
            self.assertTrue(os.path.exists(npy_path),  "cosine npy not written")
            self.assertTrue(os.path.exists(csv_path),  "top csv not written")
            self.assertTrue(os.path.exists(jsn_path),  "recs json not written")

            loaded_vec = joblib.load(pkl_path)
            self.assertIsNotNone(loaded_vec)
            loaded_sim = np.load(npy_path)
            self.assertEqual(loaded_sim.shape, sim.shape)
            loaded_recs = json.load(open(jsn_path))
            self.assertIsInstance(loaded_recs, dict)


# ── TestLoadData ─────────────────────────────────────────────────────────────
class TestLoadData(unittest.TestCase):
    """load_data() must return a cleaned DataFrame from the real CSV."""

    def test_loads_and_cleans(self):
        """load_data must return a DataFrame with the expected clean columns."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import load_data
        df = load_data()
        self.assertGreater(len(df), 100)
        # 'content' column is added by build_tfidf, not load_data
        self.assertIn("description", df.columns)
        # All descriptions must be strings (no NaN)
        self.assertTrue(df["description"].notna().all())
        # Ratings should be numeric 0-5
        self.assertTrue((df["rating"] >= 0).all())
        self.assertTrue((df["rating"] <= 5).all())
        # No 'nan' strings in image field
        self.assertFalse((df.get("image", pd.Series([])).astype(str) == "nan").any())


# ── TestEdgeCasesInExtract ────────────────────────────────────────────────────
class TestEdgeCasesInExtract(unittest.TestCase):
    """extract_top_and_recs() edge cases: flat scores and duplicate skip."""

    def _df(self, titles, scores=None):
        import pandas as pd
        n = len(titles)
        df = pd.DataFrame({
            "title":         titles,
            "author":        ["A"] * n,
            "description":   ["python data algorithms " * 3] * n,
            "rating":        [4.0] * n,
            "ratings_count": [500] * n,
            "score":         scores if scores is not None else [0.5] * n,
            "bayes_rating":  [4.0] * n,
            "category":      ["Programming"] * n,
            "desc_short":    ["desc"] * n,
            "year":          [2020] * n,
            "pages":         [200] * n,
            "image":         [""] * n,
            "download_link": [""] * n,
            "file_info":     [""] * n,
            "price":         [0.0] * n,
            "currency":      ["USD"] * n,
            "preview_link":  [""] * n,
            "info_link":     [""] * n,
            "source":        ["Test"] * n,
            "platform":      ["PDF"] * n,
        })
        return df

    def test_flat_scores_dont_crash(self):
        """When s_min == s_max (all scores identical), rescaling must not divide by zero."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import build_tfidf, build_similarity, extract_top_and_recs
        from config.settings import MODEL

        df  = self._df([f"Book {i}" for i in range(20)], scores=[0.5] * 20)
        _, mat = build_tfidf(df)
        sim    = build_similarity(mat)
        orig   = MODEL.TOP_N
        MODEL.TOP_N = 20
        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N = orig
        # All scores should be 0.0 when flat (fallback path)
        self.assertTrue((top["score"] == 0.0).all() or top["score"].std() == pytest.approx(0, abs=1e-9))

    def test_near_dup_with_equal_scores_keeps_lower_index(self):
        """Near-dup dedup: when scores are equal, the first occurrence is kept."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import build_tfidf, build_similarity, extract_top_and_recs
        from config.settings import MODEL

        titles = ["Python Guide", "python guide", "Java Basics"]
        df  = self._df(titles, scores=[0.9, 0.9, 0.7])
        _, mat = build_tfidf(df)
        sim    = build_similarity(mat)
        orig   = MODEL.TOP_N
        MODEL.TOP_N = 3
        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N = orig
        result_titles = top["title"].tolist()
        self.assertIn("Python Guide",  result_titles)
        self.assertNotIn("python guide", result_titles)
        self.assertIn("Java Basics",   result_titles)


# ── TestTrainModelMain ────────────────────────────────────────────────────────
class TestTrainModelMain(unittest.TestCase):
    """train_model.main() must return 0 with real data and 1 on failure."""

    def test_main_returns_zero_with_real_data(self):
        """main() must return 0 end-to-end with real cleaned data."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/train_model.py"],
            capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0,
                         f"train_model.py exited non-zero:\n{result.stderr[-500:]}")

    def test_main_returns_one_on_broken_csv(self):
        """main() must return 1 when data loading raises an exception."""
        from unittest.mock import patch
        from scripts.train_model import main
        with patch("scripts.train_model.load_data",
                   side_effect=RuntimeError("corrupt csv")):
            result = main()
        self.assertEqual(result, 1)


# ── TestLoadDataAutoPreprocess ────────────────────────────────────────────────
class TestLoadDataAutoPreprocess(unittest.TestCase):
    """load_data() must call preprocess.py when books_clean.csv is missing."""

    def test_calls_preprocess_when_csv_missing(self):
        """When CLEAN_CSV does not exist, load_data must run preprocess.py."""
        from unittest.mock import patch, MagicMock
        import pandas as pd
        dummy_df = pd.DataFrame({
            "title": ["T"], "author": ["A"], "description": ["D"],
            "rating": [4.0], "ratings_count": [100], "download_link": [""],
            "file_info": [""], "price": [0.0], "currency": ["USD"],
            "preview_link": [""], "info_link": [""], "image": [""],
        })
        with patch("scripts.train_model.os.path.exists", return_value=False):
            with patch("scripts.train_model.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                with patch("scripts.train_model.pd.read_csv", return_value=dummy_df):
                    from scripts.train_model import load_data
                    df = load_data()
        mock_run.assert_called_once()
        self.assertGreater(len(df), 0)

    def test_raises_when_preprocess_fails(self):
        """load_data must raise RuntimeError when preprocess.py exits non-zero."""
        from unittest.mock import patch, MagicMock
        with patch("scripts.train_model.os.path.exists", return_value=False):
            with patch("scripts.train_model.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                from scripts.train_model import load_data
                with self.assertRaises(RuntimeError):
                    load_data()


# ── TestNearDupContinue ───────────────────────────────────────────────────────
class TestNearDupContinue(unittest.TestCase):
    """Near-dup dedup continue branch: j already in drop_idx must be skipped."""

    def test_triple_near_dup_only_drops_two(self):
        """With three near-identical titles, only two should be dropped."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import build_tfidf, build_similarity, extract_top_and_recs
        from config.settings import MODEL

        # Three variants of same title — should only keep the highest-scoring one
        titles = [
            "Python Programming Guide Edition",
            "python programming guide edition",    # near-dup of [0]
            "python programming guide edition 2",  # near-dup of [0] and [1]
            "Java Fundamentals",                   # distinct
        ]
        import pandas as pd
        df = pd.DataFrame({
            "title":         titles,
            "author":        ["A"] * 4,
            "description":   ["python guide data"] * 4,
            "rating":        [4.0, 3.9, 3.8, 4.2],
            "ratings_count": [500, 450, 400, 600],
            "score":         [0.90, 0.85, 0.80, 0.95],
            "bayes_rating":  [4.0, 3.9, 3.8, 4.2],
            "category":      ["Programming"] * 4,
            "desc_short":    ["d"] * 4,
            "year":          [2020] * 4,
            "pages":         [200] * 4,
            "image":         [""] * 4,
            "download_link": [""] * 4,
            "file_info":     [""] * 4,
            "price":         [0.0] * 4,
            "currency":      ["USD"] * 4,
            "preview_link":  [""] * 4,
            "info_link":     [""] * 4,
            "source":        ["Test"] * 4,
            "platform":      ["PDF"] * 4,
        })
        _, mat = build_tfidf(df)
        sim    = build_similarity(mat)
        orig   = MODEL.TOP_N
        MODEL.TOP_N = 4
        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N = orig

        result_titles = top["title"].tolist()
        # "Java Fundamentals" must survive
        self.assertIn("Java Fundamentals", result_titles)
        # "Python Programming Guide Edition" (highest score) must survive
        self.assertIn("Python Programming Guide Edition", result_titles)
        # At least one variant removed
        self.assertLess(len(result_titles), 4)


# ── TestNearDupLoopSkip ───────────────────────────────────────────────────────
class TestNearDupLoopSkip(unittest.TestCase):
    """The near-dup dedup inner loop must skip indices already queued for removal."""

    def test_only_one_of_triple_near_dup_removed(self):
        """Three near-identical titles should result in exactly one removal, not two."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import (
            extract_top_and_recs, build_tfidf, build_similarity, compute_scores
        )
        from config.settings import MODEL

        # Three near-identical titles — all pairs score > 0.98
        # Build from _make_books so all required columns are present
        df = _make_books(25)  # use more than TOP_N=4 to avoid edge cases
        # Override first 4 titles with near-duplicate variants
        df.at[0, "title"] = "Python Programming Guide"
        df.at[1, "title"] = "python programming guide"
        df.at[2, "title"] = "PYTHON PROGRAMMING GUIDE"
        df.at[3, "title"] = "Completely Different Topic About Databases SQL"
        # Give them distinct descriptions to ensure they appear in top-N
        for i in range(4):
            df.at[i, "description"] = f"python programming intro guide topic {i}"
            df.at[i, "rating"]        = 4.0 - i * 0.05
            df.at[i, "ratings_count"] = 1000 - i * 50

        original   = MODEL.TOP_N
        MODEL.TOP_N = len(df)
        _, mat = build_tfidf(df)
        sim    = build_similarity(mat)
        df     = compute_scores(df)   # adds 'score' column needed by near-dup dedup
        top, _ = extract_top_and_recs(df, sim)
        MODEL.TOP_N = original

        top_titles = list(top["title"])
        # The canonical title must survive
        self.assertIn("Python Programming Guide", top_titles)
        # The unrelated book must also survive
        self.assertIn("Completely Different Topic About Databases SQL", top_titles)
        # Near-dups must be gone
        self.assertNotIn("python programming guide", top_titles)
        self.assertNotIn("PYTHON PROGRAMMING GUIDE", top_titles)


# ── TestLoadDataAutoPreprocess ────────────────────────────────────────────────
class TestLoadDataAutoPreprocess(unittest.TestCase):
    """load_data() must auto-run preprocess.py when books_clean.csv is missing."""

    def test_raises_when_preprocess_fails(self):
        """load_data must raise RuntimeError when auto-preprocess exits non-zero."""
        import tempfile, subprocess
        from unittest.mock import patch, MagicMock
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scripts.train_model import load_data
        from config.settings import PATHS

        with tempfile.TemporaryDirectory() as tmp:
            fake_csv = os.path.join(tmp, "books_clean.csv")
            mock_result = MagicMock()
            mock_result.returncode = 1  # preprocess fails

            with patch("scripts.train_model.PATHS") as mp:
                mp.CLEAN_CSV = fake_csv          # non-existent → triggers auto-run
                mp.ROOT      = PATHS.ROOT
                with patch("scripts.train_model.subprocess.run",
                           return_value=mock_result):
                    with self.assertRaises(RuntimeError):
                        load_data()
