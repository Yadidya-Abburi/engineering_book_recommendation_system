"""
tests/test_build_app.py
────────────────────────
Tests for build_app.py — covers the 3 scenarios found in the audit.

Fix #9: 3 new tests for:
  1. Missing optional columns do not crash load_model_outputs()
  2. 'nan' strings are cleaned from link fields
  3. Output HTML contains all required JS fields

Run:
    pytest tests/test_build_app.py -v
"""

import json
import os
import sys
import tempfile

import pandas as pd
import pytest
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import PATHS


def _make_top(n=10, include_optional=True) -> pd.DataFrame:
    base = {
        "title":         [f"Book {i}" for i in range(n)],
        "author":        ["Author A"] * n,
        "publisher":     ["Pub"] * n,
        "year":          [2020] * n,
        "pages":         [300] * n,
        "rating":        [4.0] * n,
        "ratings_count": [500] * n,
        "score":         [0.85] * n,
        "desc_short":    ["A description."] * n,
        "image":         [""] * n,
        "category":      ["Programming"] * n,
        "source":        ["Goodreads"] * n,
    }
    if include_optional:
        base.update({
            "platform":      ["PDF"] * n,
            "download_link": ["https://example.com/dl"] * n,
            "file_info":     ["PDF, 5MB"] * n,
            "price":         [0.0] * n,
            "currency":      ["USD"] * n,
            "preview_link":  [""] * n,
            "info_link":     [""] * n,
        })
    return pd.DataFrame(base)


def _make_recs(titles) -> dict:
    return {t: [{"title": titles[(i+1) % len(titles)], "author": "A",
                 "rating": 4.0, "similarity": 0.4}]
            for i, t in enumerate(titles)}


class TestLoadModelOutputs:
    """Fix #2 and Fix #9: build_app must not crash on missing optional columns."""

    def test_all_columns_present_loads_cleanly(self, tmp_path, monkeypatch):
        top = _make_top(5, include_optional=True)
        recs = _make_recs(top["title"].tolist())
        top_path = tmp_path / "top_books.csv"
        rec_path = tmp_path / "recommendations.json"
        top.to_csv(top_path, index=False)
        with open(rec_path, "w") as f:
            json.dump(recs, f)
        monkeypatch.setattr(PATHS, "TOP_BOOKS_CSV", str(top_path))
        monkeypatch.setattr(PATHS, "RECS_JSON", str(rec_path))
        from scripts.build_app import load_model_outputs
        books, loaded_recs = load_model_outputs()
        assert len(books) == 5
        assert len(loaded_recs) == 5

    def test_missing_price_column_does_not_crash(self, tmp_path, monkeypatch):
        """Fix #2: optional column 'price' absent → default 0.0, no crash."""
        top = _make_top(5, include_optional=True)
        top = top.drop(columns=["price", "download_link"])
        recs = _make_recs(top["title"].tolist())
        top_path = tmp_path / "top_books.csv"
        rec_path = tmp_path / "recommendations.json"
        top.to_csv(top_path, index=False)
        with open(rec_path, "w") as f:
            json.dump(recs, f)
        monkeypatch.setattr(PATHS, "TOP_BOOKS_CSV", str(top_path))
        monkeypatch.setattr(PATHS, "RECS_JSON", str(rec_path))
        from scripts.build_app import load_model_outputs
        books, _ = load_model_outputs()   # must NOT raise KeyError
        assert all("price" in b for b in books)
        assert all(b["price"] == 0.0 for b in books)
        assert all("download_link" in b for b in books)
        assert all(b["download_link"] == "" for b in books)

    def test_all_optional_columns_missing_does_not_crash(self, tmp_path, monkeypatch):
        """Fix #2: no optional columns at all → all use safe defaults."""
        top = _make_top(5, include_optional=False)
        recs = _make_recs(top["title"].tolist())
        top_path = tmp_path / "top_books.csv"
        rec_path = tmp_path / "recommendations.json"
        top.to_csv(top_path, index=False)
        with open(rec_path, "w") as f:
            json.dump(recs, f)
        monkeypatch.setattr(PATHS, "TOP_BOOKS_CSV", str(top_path))
        monkeypatch.setattr(PATHS, "RECS_JSON", str(rec_path))
        from scripts.build_app import load_model_outputs
        books, _ = load_model_outputs()
        for col, default in [("price", 0.0), ("download_link", ""), ("platform", ""),
                              ("file_info", ""), ("currency", "USD"),
                              ("preview_link", ""), ("info_link", "")]:
            assert all(b[col] == default for b in books), f"Wrong default for {col}"

    def test_nan_strings_cleaned_from_link_fields(self, tmp_path, monkeypatch):
        """Fix #2: 'nan' strings (from pandas CSV round-trip) are stripped."""
        top = _make_top(5, include_optional=True)
        top["preview_link"] = "nan"     # simulate pandas NaN → string "nan"
        top["info_link"]    = "nan"
        top["image"]        = "nan"
        recs = _make_recs(top["title"].tolist())
        top_path = tmp_path / "top_books.csv"
        rec_path = tmp_path / "recommendations.json"
        top.to_csv(top_path, index=False)
        with open(rec_path, "w") as f:
            json.dump(recs, f)
        monkeypatch.setattr(PATHS, "TOP_BOOKS_CSV", str(top_path))
        monkeypatch.setattr(PATHS, "RECS_JSON", str(rec_path))
        from scripts.build_app import load_model_outputs
        books, _ = load_model_outputs()
        assert all(b["preview_link"] != "nan" for b in books), "nan string in preview_link"
        assert all(b["info_link"]    != "nan" for b in books), "nan string in info_link"
        assert all(b["image"]        != "nan" for b in books), "nan string in image"

    def test_missing_recs_file_raises_file_not_found(self, tmp_path, monkeypatch):
        top = _make_top(5)
        top.to_csv(tmp_path / "top.csv", index=False)
        monkeypatch.setattr(PATHS, "TOP_BOOKS_CSV", str(tmp_path / "top.csv"))
        monkeypatch.setattr(PATHS, "RECS_JSON", str(tmp_path / "nonexistent.json"))
        from scripts.build_app import load_model_outputs
        with pytest.raises(FileNotFoundError):
            load_model_outputs()


class TestValidateOutput:
    """Fix #9: output HTML must contain all required markers."""

    def test_all_markers_present_passes(self):
        from scripts.build_app import validate_output
        html = "const BOOKS=[];const RECS={};download_link file_info price platform getPlatforms"
        assert validate_output(html) is True

    def test_missing_books_const_fails(self):
        from scripts.build_app import validate_output
        html = "const RECS={};download_link file_info price platform getPlatforms"
        assert validate_output(html) is False

    def test_missing_download_link_fails(self):
        from scripts.build_app import validate_output
        html = "const BOOKS=[];const RECS={};file_info price platform getPlatforms"
        assert validate_output(html) is False

    def test_actual_app_html_passes(self):
        """Fix #9 integration: the real app/index.html must pass validation."""
        if not os.path.exists(PATHS.APP_HTML):
            pytest.skip("app/index.html not found — run build_app.py first")
        from scripts.build_app import validate_output
        html = open(PATHS.APP_HTML, encoding="utf-8").read()
        assert validate_output(html) is True, "app/index.html is missing required JS fields"


# ── TestReadTemplate ──────────────────────────────────────────────────────────
class TestReadTemplate(unittest.TestCase):
    """read_template() must find index.html or raise FileNotFoundError."""

    def test_reads_existing_index_html(self):
        """read_template must return a non-empty string when index.html exists."""
        from scripts.build_app import read_template
        result = read_template()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 1000)

    def test_raises_when_no_files_found(self):
        """read_template must raise FileNotFoundError if neither template nor index exists."""
        import tempfile
        from unittest.mock import patch
        from scripts.build_app import read_template
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.build_app.PATHS") as mock_paths:
                mock_paths.APP      = tmp
                mock_paths.APP_HTML = os.path.join(tmp, "index.html")
                with self.assertRaises(FileNotFoundError):
                    read_template()


# ── TestInjectDataGuard ───────────────────────────────────────────────────────
class TestInjectDataGuard(unittest.TestCase):
    """inject_data() must warn when a pattern appears more than once."""

    def test_single_occurrence_no_warning(self):
        """inject_data must succeed silently when patterns appear exactly once."""
        from scripts.build_app import inject_data
        template = "const BOOKS=[];const RECS={};"
        result   = inject_data(template, [], {})
        self.assertIn("const BOOKS=", result)
        self.assertIn("const RECS=", result)

    def test_inject_replaces_books_and_recs(self):
        """inject_data must replace BOOKS and RECS with serialised JSON."""
        from scripts.build_app import inject_data
        template = "const BOOKS=[];const RECS={};"
        books    = [{"title": "Test Book", "rating": 4.2}]
        recs     = {"Test Book": [{"title": "Other", "similarity": 0.5}]}
        result   = inject_data(template, books, recs)
        self.assertIn('"Test Book"', result)
        self.assertIn('"Other"', result)


# ── TestLoadOutputsMissing ────────────────────────────────────────────────────
class TestLoadOutputsMissing(unittest.TestCase):
    """load_model_outputs() must raise FileNotFoundError when outputs are absent."""

    def test_raises_when_top_books_missing(self):
        """FileNotFoundError when top_books.csv does not exist."""
        import tempfile
        from unittest.mock import patch
        from scripts.build_app import load_model_outputs
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.build_app.PATHS") as mock_paths:
                mock_paths.TOP_BOOKS_CSV = os.path.join(tmp, "missing.csv")
                mock_paths.RECS_JSON     = os.path.join(tmp, "missing.json")
                with self.assertRaises(FileNotFoundError):
                    load_model_outputs()


# ── TestReadTemplateFile ──────────────────────────────────────────────────────
class TestReadTemplateFile(unittest.TestCase):
    """read_template() must read template.html when it exists (preferred over index.html)."""

    def test_reads_template_html_when_present(self):
        """If template.html exists it must be returned over index.html."""
        import tempfile
        from unittest.mock import patch
        from scripts.build_app import read_template
        with tempfile.TemporaryDirectory() as tmp:
            tpl_path = os.path.join(tmp, "template.html")
            idx_path = os.path.join(tmp, "index.html")
            with open(tpl_path, "w") as f:
                f.write("TEMPLATE_CONTENT")
            with open(idx_path, "w") as f:
                f.write("INDEX_CONTENT")
            with patch("scripts.build_app.PATHS") as mp:
                mp.APP      = tmp
                mp.APP_HTML = idx_path
                result = read_template()
            self.assertEqual(result, "TEMPLATE_CONTENT")


# ── TestInjectDataMultiMatchWarning ───────────────────────────────────────────
class TestInjectDataMultiMatchWarning(unittest.TestCase):
    """inject_data() must log a warning when a pattern appears more than once."""

    def test_warning_logged_on_duplicate_pattern(self):
        """inject_data must log a warning when const BOOKS= appears twice."""
        import logging
        from scripts.build_app import inject_data
        # Two BOOKS declarations — should trigger the multi-match warning
        template = "const BOOKS=[];var x=1;const BOOKS=[];const RECS={};"
        with self.assertLogs("scripts.build_app", level="WARNING") as cm:
            inject_data(template, [], {})
        self.assertTrue(any("multiple times" in msg or "BOOKS" in msg for msg in cm.output))


# ── TestFetchCoverUrl ─────────────────────────────────────────────────────────
class TestFetchCoverUrl(unittest.TestCase):
    """fetch_cover_url() must return correct URL across all API response paths."""

    def _mock_requests(self, json_data, ok=True):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.ok = ok
        mock_resp.json.return_value = json_data
        mock_req = MagicMock()
        mock_req.get.return_value = mock_resp
        mock_req.utils.quote = lambda s: s.replace(" ", "+")
        return mock_req

    def test_returns_cover_id_url_when_present(self):
        """fetch_cover_url returns covers.openlibrary.org/b/id/... when cover_i exists."""
        from scripts.build_app import fetch_cover_url
        import sys
        mock_req = self._mock_requests({"docs": [{"cover_i": 12345}]})
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
                sys.modules, {"requests": mock_req}):
            url = fetch_cover_url("Python Book", "Author")
        self.assertIn("12345", url)
        self.assertIn("openlibrary.org", url)

    def test_falls_back_to_isbn_url(self):
        """fetch_cover_url falls back to ISBN cover URL when cover_i is absent."""
        from scripts.build_app import fetch_cover_url
        import sys
        mock_req = self._mock_requests({"docs": [{"isbn": ["9780134685991"]}]})
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
                sys.modules, {"requests": mock_req}):
            url = fetch_cover_url("Some Book", "Author")
        self.assertIn("9780134685991", url)

    def test_returns_empty_on_no_docs(self):
        """fetch_cover_url returns empty string when API response has no docs."""
        from scripts.build_app import fetch_cover_url
        import sys
        mock_req = self._mock_requests({"docs": []})
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
                sys.modules, {"requests": mock_req}):
            url = fetch_cover_url("Unknown Book", "Author")
        self.assertEqual(url, "")

    def test_returns_empty_on_http_error(self):
        """fetch_cover_url returns empty string when API returns non-200."""
        from scripts.build_app import fetch_cover_url
        import sys
        mock_req = self._mock_requests({}, ok=False)
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
                sys.modules, {"requests": mock_req}):
            url = fetch_cover_url("Bad Request Book", "Author")
        self.assertEqual(url, "")

    def test_returns_empty_on_connection_error(self):
        """fetch_cover_url returns empty string when network raises."""
        from scripts.build_app import fetch_cover_url
        from unittest.mock import MagicMock
        import sys
        mock_req = MagicMock()
        mock_req.get.side_effect = ConnectionError("timeout")
        mock_req.utils.quote = lambda s: s
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
                sys.modules, {"requests": mock_req}):
            url = fetch_cover_url("Offline Book", "Author")
        self.assertEqual(url, "")


# ── TestPrefetchCovers ────────────────────────────────────────────────────────
class TestPrefetchCovers(unittest.TestCase):
    """prefetch_covers() must skip books that already have images."""

    def test_all_have_images_no_fetch(self):
        """prefetch_covers must not call fetch_cover_url when all books have images."""
        from unittest.mock import patch
        from scripts.build_app import prefetch_covers
        books = [{"title": "A", "author": "X", "image": "https://example.com/cover.jpg"},
                 {"title": "B", "author": "Y", "image": "https://example.com/cover2.jpg"}]
        with patch("scripts.build_app.fetch_cover_url") as mock_fetch:
            result = prefetch_covers(books, delay=0)
        mock_fetch.assert_not_called()
        self.assertEqual(len(result), 2)

    def test_missing_image_triggers_fetch(self):
        """prefetch_covers must call fetch_cover_url for books without images."""
        from unittest.mock import patch
        from scripts.build_app import prefetch_covers
        books = [{"title": "A", "author": "X", "image": ""},
                 {"title": "B", "author": "Y", "image": "https://example.com/ok.jpg"}]
        with patch("scripts.build_app.fetch_cover_url", return_value="https://new.jpg") as mock_fetch:
            result = prefetch_covers(books, delay=0)
        mock_fetch.assert_called_once_with("A", "X")
        self.assertEqual(result[0]["image"], "https://new.jpg")
        self.assertEqual(result[1]["image"], "https://example.com/ok.jpg")


# ── TestBuildAppMain ──────────────────────────────────────────────────────────
class TestBuildAppMain(unittest.TestCase):
    """main() must return 0 on success and 1 on failure."""

    def test_main_returns_zero_on_success(self):
        """main() must return 0 when build succeeds."""
        from unittest.mock import patch, MagicMock
        from scripts.build_app import main
        with patch("scripts.build_app.load_model_outputs",
                   return_value=([{"title": "T", "image": ""}], {"T": []})):
            with patch("scripts.build_app.prefetch_covers", side_effect=lambda b, **_: b):
                with patch("scripts.build_app.read_template", return_value="const BOOKS=[];const RECS={};getPlatforms download_link platform"):
                    with patch("scripts.build_app.inject_data",
                               return_value="const BOOKS=[];const RECS={};getPlatforms download_link platform"):
                        with patch("scripts.build_app.validate_output", return_value=True):
                            with patch("builtins.open", unittest.mock.mock_open()):
                                with patch("os.path.getsize", return_value=5000):
                                    import sys
                                    sys.argv = ["build_app.py", "--skip-covers"]
                                    result = main()
        self.assertEqual(result, 0)

    def test_main_returns_one_on_exception(self):
        """main() must return 1 when an exception is raised."""
        from unittest.mock import patch
        from scripts.build_app import main
        with patch("scripts.build_app.load_model_outputs",
                   side_effect=FileNotFoundError("no file")):
            import sys
            sys.argv = ["build_app.py", "--skip-covers"]
            result = main()
        self.assertEqual(result, 1)


# ── TestPrefetchCoversEdgeCases ───────────────────────────────────────────────
class TestPrefetchCoversEdgeCases(unittest.TestCase):
    """Edge cases: failed fetches and progress logging."""

    def test_failed_fetch_increments_failed_counter(self):
        """When fetch_cover_url returns empty string, 'failed' must be counted."""
        from unittest.mock import patch
        from scripts.build_app import prefetch_covers
        books = [{"title": f"Book {i}", "author": "A", "image": ""} for i in range(3)]
        with patch("scripts.build_app.fetch_cover_url", return_value="") as mock_fetch:
            result = prefetch_covers(books, delay=0)
        self.assertEqual(mock_fetch.call_count, 3)
        # All images remain empty (all failed)
        self.assertTrue(all(b["image"] == "" for b in result))

    def test_progress_logged_every_25(self):
        """prefetch_covers must log progress after every 25th fetch."""
        import logging
        from unittest.mock import patch
        from scripts.build_app import prefetch_covers
        books = [{"title": f"Book {i}", "author": "A", "image": ""} for i in range(26)]
        with patch("scripts.build_app.fetch_cover_url", return_value="https://ok.jpg"):
            with self.assertLogs("scripts.build_app", level="INFO") as cm:
                prefetch_covers(books, delay=0)
        # Should have logged a "25 / 26" progress message
        self.assertTrue(any("25" in msg and "26" in msg for msg in cm.output))


# ── TestBuildAppMainValidationFails ──────────────────────────────────────────
class TestBuildAppMainValidationFails(unittest.TestCase):
    """main() must return 1 when validate_output returns False."""

    def test_main_returns_one_when_validation_fails(self):
        """main() must return 1 when the built HTML fails validation."""
        from unittest.mock import patch
        from scripts.build_app import main
        import sys
        sys.argv = ["build_app.py", "--skip-covers"]
        with patch("scripts.build_app.load_model_outputs",
                   return_value=([{"title": "T", "image": ""}], {})):
            with patch("scripts.build_app.prefetch_covers", side_effect=lambda b, **_: b):
                with patch("scripts.build_app.read_template", return_value="X"):
                    with patch("scripts.build_app.inject_data", return_value="X"):
                        with patch("scripts.build_app.validate_output", return_value=False):
                            result = main()
        self.assertEqual(result, 1)


# ── TestBuildAppMainValidationFailure ─────────────────────────────────────────
class TestBuildAppMainValidationFailure(unittest.TestCase):
    """main() must return 1 when validate_output detects missing markers."""

    def test_main_returns_1_on_bad_html(self):
        """main() must return 1 when the generated HTML fails validation."""
        import tempfile, json
        from unittest.mock import patch
        from scripts.build_app import main

        with tempfile.TemporaryDirectory() as tmp:
            # Write minimal valid top_books.csv and recs.json
            import pandas as pd
            top = pd.DataFrame({
                "title": ["Book A"], "author": ["Auth"], "publisher": ["Pub"],
                "year": [2020], "pages": [200], "rating": [4.0],
                "ratings_count": [100], "score": [0.9],
                "desc_short": ["desc"], "image": [""], "category": ["AI/ML"],
                "source": ["test"], "platform": ["PDF"],
                "download_link": [""], "file_info": [""], "price": [0.0],
                "currency": ["USD"], "preview_link": [""], "info_link": [""],
            })
            csv_path  = os.path.join(tmp, "top.csv")
            json_path = os.path.join(tmp, "recs.json")
            html_path = os.path.join(tmp, "index.html")
            top.to_csv(csv_path, index=False)
            json.dump({"Book A": []}, open(json_path, "w"))
            # Write a template that will FAIL validation (no getPlatforms etc.)
            open(html_path, "w").write("const BOOKS=[];const RECS={};")

            with patch("scripts.build_app.PATHS") as mp:
                mp.TOP_BOOKS_CSV = csv_path
                mp.RECS_JSON     = json_path
                mp.APP_HTML      = html_path
                mp.APP           = tmp
                import sys
                with patch.object(sys, "argv", ["build_app.py", "--skip-covers"]):
                    result = main()

            # With a bare template, validation fails → return 1
            self.assertEqual(result, 1)


# ── TestBuildAppMainWithCovers ────────────────────────────────────────────────
class TestBuildAppMainWithCovers(unittest.TestCase):
    """main() without --skip-covers must call prefetch_covers."""

    def test_main_calls_prefetch_when_covers_not_skipped(self):
        """main() without --skip-covers flag must call prefetch_covers."""
        from unittest.mock import patch, call
        from scripts.build_app import main
        import sys
        sys.argv = ["build_app.py"]   # no --skip-covers
        books_with_covers = [{"title": "T", "image": "https://ok.jpg"}]
        with patch("scripts.build_app.load_model_outputs",
                   return_value=([{"title": "T", "image": ""}], {})):
            with patch("scripts.build_app.prefetch_covers",
                       return_value=books_with_covers) as mock_prefetch:
                with patch("scripts.build_app.read_template",
                           return_value="const BOOKS=[];const RECS={};getPlatforms download_link platform"):
                    with patch("scripts.build_app.inject_data",
                               return_value="const BOOKS=[];const RECS={};getPlatforms download_link platform"):
                        with patch("scripts.build_app.validate_output", return_value=True):
                            with patch("builtins.open", unittest.mock.mock_open()):
                                with patch("os.path.getsize", return_value=5000):
                                    result = main()
        mock_prefetch.assert_called_once()
        self.assertEqual(result, 0)
