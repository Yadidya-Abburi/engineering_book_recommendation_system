"""
evaluate.py
───────────
Evaluates the trained model and generates plots.

Fixes applied:
  #11 Imports directly from config.settings

Run:
    python scripts/evaluate.py
Exit code 0 = success, 1 = failure.
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#181818",
    "axes.facecolor":   "#1e1e1e",
    "text.color":       "#ffffff",
    "axes.labelcolor":  "#b3b3b3",
    "xtick.color":      "#b3b3b3",
    "ytick.color":      "#b3b3b3",
    "grid.color":       "#2a2a2a",
})
ACCENT = "#1db954"


def load() -> tuple[pd.DataFrame, dict]:
    missing = [p for p in (PATHS.RECS_JSON, PATHS.TOP_BOOKS_CSV) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Model outputs missing: {missing}\n"
            "Run  python scripts/train_model.py  first."
        )
    top = pd.read_csv(PATHS.TOP_BOOKS_CSV)
    top["rating"] = pd.to_numeric(top["rating"], errors="coerce").fillna(0)
    with open(PATHS.RECS_JSON, encoding="utf-8") as f:
        recs = json.load(f)
    return top, recs


def compute_metrics(top: pd.DataFrame, recs: dict) -> dict:
    if top.empty or not recs:
        return {
            "Coverage (%)": 0, "Avg Similarity Score": 0,
            "Avg Intra-list Diversity": 0, "Category Precision (%)": 0,
            "Avg Rating Lift": 0,
        }

    title_to_cat = dict(zip(top["title"].astype(str).str.strip(), top["category"]))
    coverage_count, sim_scores, diversity_scores = 0, [], []
    cat_precisions, rating_lifts = [], []

    for title, rec_list in recs.items():
        if not rec_list:
            continue
        coverage_count += 1
        sims = [r["similarity"] for r in rec_list]
        sim_scores.append(float(np.mean(sims)))

        if len(sims) > 1:
            pairwise = [abs(sims[i] - sims[j])
                        for i in range(len(sims))
                        for j in range(i + 1, len(sims))]
            diversity_scores.append(1.0 - float(np.mean(pairwise)))

        query_cat = title_to_cat.get(title.strip(), "")
        rec_cats  = [title_to_cat.get(r["title"].strip(), "") for r in rec_list]
        cat_precisions.append(sum(c == query_cat for c in rec_cats) / len(rec_cats))

        match = top[top["title"].astype(str).str.strip() == title.strip()]
        if not match.empty:
            q_r    = match.iloc[0]["rating"]
            rec_avg = float(np.mean([r["rating"] for r in rec_list if r["rating"] > 0] or [0]))
            rating_lifts.append(rec_avg - q_r)

    return {
        "Coverage (%)":             round(coverage_count / len(recs) * 100, 1) if recs else 0,
        "Avg Similarity Score":     round(float(np.mean(sim_scores)), 4)        if sim_scores else 0,
        "Avg Intra-list Diversity": round(float(np.mean(diversity_scores)), 4)  if diversity_scores else 0,
        "Category Precision (%)":   round(float(np.mean(cat_precisions)) * 100, 1) if cat_precisions else 0,
        "Avg Rating Lift":          round(float(np.mean(rating_lifts)), 3)      if rating_lifts else 0,
    }


def _save(fig, name: str) -> None:
    os.makedirs(PATHS.PLOTS, exist_ok=True)
    path = os.path.join(PATHS.PLOTS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#181818")
    plt.close(fig)
    log.info("  Saved → %s", path)


def plot_rating_distribution(top: pd.DataFrame):
    rated = top[top["rating"] > 0]["rating"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rated, bins=20, color=ACCENT, edgecolor="#121212", alpha=0.9)
    ax.axvline(rated.mean(), color="#f0c040", linestyle="--", linewidth=1.5,
               label=f"Mean {rated.mean():.2f}")
    ax.set_title("Rating Distribution (Top Books)", fontsize=13, pad=12)
    ax.set_xlabel("Rating"); ax.set_ylabel("Count")
    ax.legend(facecolor="#232323", edgecolor="#2a2a2a", labelcolor="#fff")
    _save(fig, "rating_distribution.png")


def plot_score_distribution(top: pd.DataFrame):
    """New plot: shows the improved Bayesian score spread."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(top["score"], bins=25, color="#8b7cf8", edgecolor="#121212", alpha=0.9)
    ax.axvline(top["score"].mean(), color="#f0c040", linestyle="--", linewidth=1.5,
               label=f"Mean {top['score'].mean():.3f}  std={top['score'].std():.3f}")
    ax.set_title("ML Score Distribution (Top Books — Bayesian Scoring)", fontsize=13, pad=12)
    ax.set_xlabel("Score"); ax.set_ylabel("Count")
    ax.legend(facecolor="#232323", edgecolor="#2a2a2a", labelcolor="#fff")
    _save(fig, "score_distribution.png")


def plot_category_breakdown(top: pd.DataFrame):
    counts = top["category"].value_counts()
    colors = ["#1db954","#a78bfa","#60a5fa","#f87171","#34d399",
              "#fb923c","#f472b6","#fbbf24","#94a3b8","#c084fc","#67e8f9","#4ade80","#f97316"]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(counts.index, counts.values,
                   color=colors[:len(counts)], edgecolor="#121212")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color="#b3b3b3", fontsize=10)
    ax.set_title("Top Books by Category", fontsize=13, pad=12)
    ax.set_xlabel("Count")
    _save(fig, "category_breakdown.png")


def plot_score_vs_rating(top: pd.DataFrame):
    rated = top[top["rating"] > 0]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(rated["rating"], rated["score"],
                    c=np.log1p(rated["ratings_count"]), cmap="YlGn",
                    alpha=0.8, edgecolors="#121212", linewidths=0.4, s=60)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log(ratings count)", color="#b3b3b3")
    cb.ax.yaxis.set_tick_params(color="#b3b3b3")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#b3b3b3")
    ax.set_title("ML Score vs Rating (Bayesian)", fontsize=13, pad=12)
    ax.set_xlabel("Rating"); ax.set_ylabel("Hybrid ML Score")
    _save(fig, "score_vs_rating.png")


def plot_top10(top: pd.DataFrame):
    top10 = top.nlargest(10, "score")[["title", "score"]].copy()
    top10["short"] = top10["title"].str[:40] + "…"
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10["short"][::-1], top10["score"][::-1],
                   color=ACCENT, edgecolor="#121212")
    for bar, val in zip(bars, top10["score"][::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#b3b3b3", fontsize=9)
    ax.set_title("Top 10 Books by ML Score", fontsize=13, pad=12)
    ax.set_xlabel("Score")
    _save(fig, "top10_books.png")


def plot_similarity_heatmap(recs: dict):
    titles = list(recs.keys())[:30]
    n = len(titles)
    mat = np.zeros((n, n))
    for i, t in enumerate(titles):
        for r in recs.get(t, []):
            if r["title"] in titles:
                j = titles.index(r["title"])
                mat[i][j] = r["similarity"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(mat, ax=ax, cmap="Greens", linewidths=0,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Cosine Similarity Heatmap (sample 30 books)", fontsize=12, pad=12)
    _save(fig, "similarity_heatmap.png")


def main() -> int:
    log.info("=" * 56)
    log.info("  Bookify – Model Evaluation")
    log.info("=" * 56)

    try:
        top, recs = load()
        log.info("Loaded: %d top books, %d rec entries", len(top), len(recs))

        log.info("Computing metrics…")
        metrics = compute_metrics(top, recs)
        for k, v in metrics.items():
            log.info("  %-35s %s", k, v)

        # Self-rec check
        self_recs = sum(1 for t, rl in recs.items()
                        for r in rl if r["title"].strip() == t.strip())
        log.info("  %-35s %s", "Self-recommendations (must be 0)", self_recs)
        if self_recs:
            log.error("Self-recommendations detected — retrain with fixed train_model.py")

        log.info("Generating plots…")
        plot_rating_distribution(top)
        plot_score_distribution(top)
        plot_category_breakdown(top)
        plot_score_vs_rating(top)
        plot_top10(top)
        plot_similarity_heatmap(recs)

    except Exception as exc:
        log.exception("Evaluation failed: %s", exc)
        return 1

    log.info("Evaluation complete. Plots → %s", PATHS.PLOTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
