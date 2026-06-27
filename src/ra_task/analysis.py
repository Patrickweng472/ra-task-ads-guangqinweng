from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def quadratic_weighted_kappa(a: list[int], b: list[int], categories: int = 4) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be nonempty and equally sized")
    observed = np.zeros((categories, categories), dtype=float)
    for left, right in zip(a, b):
        observed[left, right] += 1
    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) / (categories - 1)) ** 2, (categories, categories))
    denominator = (weights * expected).sum()
    return 1.0 if denominator == 0 else float(1 - (weights * observed).sum() / denominator)


def annual_summary(ads: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    data = ads[["canonical_id", "year"]].merge(labels[["canonical_id", "score"]], on="canonical_id", validate="one_to_one")
    rows = []
    for year, group in data.groupby("year", sort=True):
        n = len(group)
        headline = int((group["score"] >= 2).sum())
        low, high = wilson_interval(headline, n)
        rows.append({"year": int(year), "n_ads": n, "n_score_ge_2": headline, "share_score_ge_2": headline / n, "wilson_low": low, "wilson_high": high, "share_score_ge_1": float((group["score"] >= 1).mean()), "share_score_eq_3": float((group["score"] == 3).mean())})
    return pd.DataFrame(rows)


def plot_annual(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    years = summary["year"].to_numpy()
    shares = summary["share_score_ge_2"].to_numpy()
    yerr = np.vstack([shares - summary["wilson_low"].to_numpy(), summary["wilson_high"].to_numpy() - shares])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    axes[0].errorbar(years, shares, yerr=yerr, marker="o", capsize=3, color="#2457A6")
    axes[0].set_ylabel("Share (score >= 2)")
    axes[0].set_ylim(0, max(0.5, float(summary["wilson_high"].max()) + 0.05))
    axes[0].grid(alpha=0.25)
    axes[1].bar(years, summary["n_ads"], color="#82A6D9")
    axes[1].set_ylabel("Number of ads")
    axes[1].set_xlabel("Publication year")
    axes[1].set_xticks(years)
    fig.suptitle("AI/digital intensity and annual sample size")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
