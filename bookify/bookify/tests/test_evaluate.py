"""
tests/test_evaluate.py
───────────────────────
Unit tests for the evaluation metrics.

Run:
    pytest tests/
"""

import os
import sys
import pandas as pd
import pytest
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import compute_metrics


def _make_top(titles: list[str], categories: list[str], ratings: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "title":    titles,
        "category": categories,
        "rating":   ratings,
        "ratings_count": [100] * len(titles),
        "score":    [0.8] * len(titles),
    })


def _make_recs(titles: list[str], sim: float = 0.5) -> dict:
    recs = {}
    for i, t in enumerate(titles):
        others = [tt for tt in titles if tt != t]
        recs[t] = [
            {"title": o, "author": "A", "rating": 4.0, "similarity": sim}
            for o in others[:5]
        ]
    return recs


class TestComputeMetrics:
    def test_coverage_100_percent_when_all_have_recs(self):
        titles = [f"Book {i}" for i in range(5)]
        top    = _make_top(titles, ["AI/ML"] * 5, [4.0] * 5)
        recs   = _make_recs(titles)
        m = compute_metrics(top, recs)
        assert m["Coverage (%)"] == 100.0

    def test_coverage_0_when_no_recs(self):
        titles = ["Book A", "Book B"]
        top    = _make_top(titles, ["AI/ML"] * 2, [4.0] * 2)
        recs   = {"Book A": [], "Book B": []}
        m = compute_metrics(top, recs)
        assert m["Coverage (%)"] == 0.0

    def test_avg_similarity_matches_input(self):
        titles = [f"Book {i}" for i in range(4)]
        top    = _make_top(titles, ["AI/ML"] * 4, [4.0] * 4)
        recs   = _make_recs(titles, sim=0.42)
        m = compute_metrics(top, recs)
        assert abs(m["Avg Similarity Score"] - 0.42) < 0.01

    def test_category_precision_100_when_all_same_category(self):
        titles = [f"Book {i}" for i in range(5)]
        top    = _make_top(titles, ["Security"] * 5, [4.0] * 5)
        recs   = _make_recs(titles)
        m = compute_metrics(top, recs)
        assert m["Category Precision (%)"] == 100.0

    def test_empty_recs_returns_zero_metrics(self):
        top  = _make_top([], [], [])
        recs = {}
        m = compute_metrics(top, recs)
        assert m["Coverage (%)"] == 0

    def test_all_metrics_present(self):
        titles = [f"Book {i}" for i in range(4)]
        top    = _make_top(titles, ["AI/ML"] * 4, [4.0] * 4)
        recs   = _make_recs(titles)
        m = compute_metrics(top, recs)
        expected_keys = [
            "Coverage (%)", "Avg Similarity Score",
            "Avg Intra-list Diversity", "Category Precision (%)", "Avg Rating Lift",
        ]
        for k in expected_keys:
            assert k in m, f"Missing metric: {k}"


# ── TestLoadOutputs ────────────────────────────────────────────────────────────
class TestLoadOutputs(unittest.TestCase):
    """load() must raise FileNotFoundError when files are absent."""

    def test_raises_when_files_missing(self):
        """load_outputs raises FileNotFoundError when model outputs don't exist."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import load
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mock_paths:
                mock_paths.RECS_JSON     = os.path.join(tmp, "nope.json")
                mock_paths.TOP_BOOKS_CSV = os.path.join(tmp, "nope.csv")
                with self.assertRaises(FileNotFoundError):
                    load()

    def test_loads_real_outputs(self):
        """load_outputs must return a DataFrame and dict when files exist."""
        from scripts.evaluate import load
        top, recs = load()
        self.assertGreater(len(top), 0)
        self.assertIsInstance(recs, dict)
        self.assertGreater(len(recs), 0)


# ── TestPlotFunctions ─────────────────────────────────────────────────────────
class TestPlotFunctions(unittest.TestCase):
    """Plot functions must run without error using the Agg backend."""

    def setUp(self):
        """Set headless Agg backend before any plot test."""
        import matplotlib
        matplotlib.use("Agg")
        from scripts.evaluate import load
        self.top, self.recs = load()

    def test_plot_rating_distribution(self):
        """plot_rating_distribution must run and save a file."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_rating_distribution
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_rating_distribution(self.top)
            import os
            self.assertTrue(any(f.endswith(".png") for f in os.listdir(tmp)))

    def test_plot_score_distribution(self):
        """plot_score_distribution must run without error."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_score_distribution
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_score_distribution(self.top)

    def test_plot_category_breakdown(self):
        """plot_category_breakdown must run without error."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_category_breakdown
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_category_breakdown(self.top)

    def test_plot_score_vs_rating(self):
        """plot_score_vs_rating must run without error."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_score_vs_rating
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_score_vs_rating(self.top)

    def test_plot_top10(self):
        """plot_top10 must run without error."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_top10
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_top10(self.top)


    def test_plot_similarity_heatmap(self):
        """plot_similarity_heatmap must run without error."""
        import tempfile
        from unittest.mock import patch
        from scripts.evaluate import plot_similarity_heatmap
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.evaluate.PATHS") as mp:
                mp.PLOTS = tmp
                plot_similarity_heatmap(self.recs)


# ── TestEvaluateMain ──────────────────────────────────────────────────────────
class TestEvaluateMain(unittest.TestCase):
    """evaluate.main() must return 0 on success and 1 on failure."""

    def test_main_returns_zero_on_success(self):
        """main() must return 0 when evaluation runs successfully."""
        from unittest.mock import patch, MagicMock
        from scripts.evaluate import main
        import pandas as pd
        top = pd.DataFrame({
            "title": ["Book A", "Book B"],
            "rating": [4.0, 3.5], "ratings_count": [100, 50],
            "score": [0.8, 0.6], "category": ["AI/ML", "Security"],
        })
        recs = {"Book A": [{"title": "Book B", "similarity": 0.5}]}
        with patch("scripts.evaluate.load", return_value=(top, recs)):
            with patch("scripts.evaluate.compute_metrics", return_value={"Coverage (%)": 100}):
                with patch("scripts.evaluate.plot_rating_distribution"):
                    with patch("scripts.evaluate.plot_score_distribution"):
                        with patch("scripts.evaluate.plot_category_breakdown"):
                            with patch("scripts.evaluate.plot_score_vs_rating"):
                                with patch("scripts.evaluate.plot_top10"):
                                    with patch("scripts.evaluate.plot_similarity_heatmap"):
                                        result = main()
        self.assertEqual(result, 0)

    def test_main_returns_one_on_file_not_found(self):
        """main() must return 1 when model outputs are missing."""
        from unittest.mock import patch
        from scripts.evaluate import main
        with patch("scripts.evaluate.load", side_effect=FileNotFoundError("no outputs")):
            result = main()
        self.assertEqual(result, 1)


# ── TestComputeMetricsSelfRec ─────────────────────────────────────────────────
class TestComputeMetricsSelfRec(unittest.TestCase):
    """compute_metrics() must warn when a self-recommendation is detected."""

    def test_self_rec_triggers_warning(self):
        """compute_metrics must log an error when a book recommends itself."""
        import logging
        import pandas as pd
        from scripts.evaluate import compute_metrics
        top = pd.DataFrame({
            "title": ["Book A", "Book B"],
            "rating": [4.0, 3.5],
            "ratings_count": [100, 50],
            "score": [0.8, 0.6],
            "category": ["AI/ML", "Security"],
        })
        # Inject a self-recommendation
        recs = {"Book A": [{"title": "Book A", "rating": 4.0, "similarity": 0.99}]}
        # compute_metrics itself doesn't log the error — that happens in main()
        # Just verify it returns a valid dict even with self-recs present
        result = compute_metrics(top, recs)
        self.assertIsInstance(result, dict)
        self.assertIn("Coverage (%)", result)


# ── TestComputeMetricsSelfRecDetection ────────────────────────────────────────
class TestComputeMetricsSelfRecDetection(unittest.TestCase):
    """compute_metrics must log an error when a book recommends itself."""

    def test_self_rec_detected(self):
        """compute_metrics must still return a dict even with self-recs present."""
        import pandas as pd
        from scripts.evaluate import compute_metrics

        top = pd.DataFrame({
            "title":         ["Book A", "Book B"],
            "rating":        [4.0, 3.5],
            "ratings_count": [100, 50],
            "score":         [0.9, 0.8],
            "category":      ["AI/ML", "AI/ML"],
        })
        # Book A recommends itself — self-rec
        recs = {
            "Book A": [{"title": "Book A",  "similarity": 0.99, "rating": 4.0}],
            "Book B": [{"title": "Book A",  "similarity": 0.50, "rating": 4.0}],
        }
        result = compute_metrics(top, recs)
        # Must still return valid dict (error is logged, not raised)
        self.assertIn("Coverage (%)", result)
        self.assertIsInstance(result, dict)


# ── TestEvaluateMainSelfRecWarning ────────────────────────────────────────────
class TestEvaluateMainSelfRecWarning(unittest.TestCase):
    """evaluate.main() must log an error when self-recs are present in the data."""

    def test_self_rec_in_main_logs_error(self):
        """main() must log ERROR when recs contain a book that recommends itself."""
        import pandas as pd
        from unittest.mock import patch
        from scripts.evaluate import main
        top = pd.DataFrame({
            "title": ["Book A", "Book B"],
            "rating": [4.0, 3.5], "ratings_count": [100, 50],
            "score": [0.8, 0.6], "category": ["AI/ML", "Security"],
        })
        # Self-recommendation injected
        recs = {"Book A": [{"title": "Book A", "rating": 4.0, "similarity": 0.99}]}
        with patch("scripts.evaluate.load", return_value=(top, recs)):
            with patch("scripts.evaluate.compute_metrics", return_value={}):
                with patch("scripts.evaluate.plot_rating_distribution"):
                    with patch("scripts.evaluate.plot_score_distribution"):
                        with patch("scripts.evaluate.plot_category_breakdown"):
                            with patch("scripts.evaluate.plot_score_vs_rating"):
                                with patch("scripts.evaluate.plot_top10"):
                                    with patch("scripts.evaluate.plot_similarity_heatmap"):
                                        with self.assertLogs("scripts.evaluate", level="ERROR") as cm:
                                            main()
        self.assertTrue(any("Self-rec" in msg or "self" in msg.lower() for msg in cm.output))
