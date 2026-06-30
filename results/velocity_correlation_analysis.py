"""
Velocity vs Pitching Mechanics Correlation Analysis
====================================================
This script reads per-pitch data from pitching_data.csv and tests whether
hip-shoulder separation, stride length, and release extension correlate
with pitch velocity.

This was the original hypothesis behind the whole project:
    "Pitchers with higher velocity have larger hip-shoulder separation,
    stride length, and release extension." This script is how that gets
    tested against real data instead of just assumed.

Usage:
    1. Open pitching_data.csv and fill in real per-pitch data
       (values come from running pitch_summary_v2.py on each pitch video).
    2. Run: python velocity_correlation_analysis.py

Requirements:
    pip install pandas numpy matplotlib scipy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
CSV_PATH = "pitching_data.csv"
OUTPUT_DIR = Path(".")

METRICS = {
    "max_hip_shoulder_separation_deg": "Hip-Shoulder Separation (deg)",
    "stride_length_m": "Stride Length (m)",
    "release_extension_m": "Release Extension (m)",
}
VELOCITY_COL = "velocity_mph"
VELOCITY_LABEL = "Velocity (mph)"

MIN_PITCHES_FOR_CORRELATION = 5  # minimum sample size before a correlation
                                   # coefficient starts to mean anything


# ──────────────────────────────────────────
# Load & clean data
# ──────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Force numeric columns to numeric type (blanks become NaN)
    numeric_cols = [VELOCITY_COL] + list(METRICS.keys())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_valid_rows(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Keep only rows where both velocity and the given metric are filled in."""
    return df.dropna(subset=[VELOCITY_COL, metric_col])


# ──────────────────────────────────────────
# Correlation
# ──────────────────────────────────────────
def compute_correlation(df: pd.DataFrame, metric_col: str) -> dict:
    """
    Compute Pearson correlation between velocity and a given metric.
    Handles small/empty samples gracefully instead of erroring out.
    """
    valid = get_valid_rows(df, metric_col)
    n = len(valid)

    result = {"n": n, "r": None, "p_value": None, "status": ""}

    if n == 0:
        result["status"] = "No data yet — fill in pitching_data.csv"
        return result

    if n == 1:
        result["status"] = "Only 1 pitch — correlation requires at least 2"
        return result

    if n < MIN_PITCHES_FOR_CORRELATION:
        # Still compute it, but flag the small sample size
        r, p = stats.pearsonr(valid[VELOCITY_COL], valid[metric_col])
        result["r"] = r
        result["p_value"] = p
        result["status"] = (f"Preliminary only (n={n} — recommend at least "
                              f"{MIN_PITCHES_FOR_CORRELATION} pitches for a reliable result)")
        return result

    r, p = stats.pearsonr(valid[VELOCITY_COL], valid[metric_col])
    result["r"] = r
    result["p_value"] = p
    if p < 0.05:
        result["status"] = "Statistically significant (p < 0.05)"
    else:
        result["status"] = "Not statistically significant (p >= 0.05)"
    return result


def interpret_r(r: float) -> str:
    """Translate a correlation coefficient into plain language."""
    if r is None:
        return "-"
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    return f"{strength} {direction} correlation"


# ──────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, metric_col: str, metric_label: str,
                  corr_result: dict, save_path: Path):
    valid = get_valid_rows(df, metric_col)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    if valid.empty:
        ax.text(0.5, 0.5, "No data yet\nFill in pitching_data.csv to see results",
                ha="center", va="center", fontsize=13, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.scatter(valid[metric_col], valid[VELOCITY_COL],
                   s=70, color="#1D9E75", alpha=0.8, edgecolors="#0F6E56", linewidth=1)

        # Add a trend line once there are at least 2 points
        if len(valid) >= 2:
            z = np.polyfit(valid[metric_col], valid[VELOCITY_COL], 1)
            trend_x = np.linspace(valid[metric_col].min(), valid[metric_col].max(), 50)
            trend_y = np.polyval(z, trend_x)
            ax.plot(trend_x, trend_y, color="#D85A30", linestyle="--", linewidth=1.5,
                    label="Trend line")
            ax.legend(loc="best", fontsize=9)

        ax.set_xlabel(metric_label, fontsize=11)
        ax.set_ylabel(VELOCITY_LABEL, fontsize=11)

    r = corr_result["r"]
    n = corr_result["n"]
    title = f"{metric_label} vs {VELOCITY_LABEL}"
    if r is not None:
        title += f"\nr = {r:.3f}  (n = {n})  -  {interpret_r(r)}"
    else:
        title += f"\n(n = {n})  -  {corr_result['status']}"

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {save_path}")


def plot_combined_summary(results: dict, save_path: Path):
    """Bar chart comparing all three correlation coefficients at a glance."""
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = []
    values = []
    colors = []

    for metric_col, label in METRICS.items():
        r = results[metric_col]["r"]
        labels.append(label)
        values.append(r if r is not None else 0)
        if r is None:
            colors.append("#CCCCCC")
        elif r > 0:
            colors.append("#1D9E75")
        else:
            colors.append("#D85A30")

    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson correlation coefficient (r)", fontsize=11)
    ax.set_title("Correlation with velocity, by metric", fontsize=12)

    for bar, metric_col in zip(bars, METRICS.keys()):
        n = results[metric_col]["n"]
        r = results[metric_col]["r"]
        label = f"n={n}" if r is None else f"r={r:.2f} (n={n})"
        x_pos = bar.get_width() + (0.03 if bar.get_width() >= 0 else -0.03)
        ha = "left" if bar.get_width() >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary chart: {save_path}")


# ──────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "=" * 55)
    print("  Velocity Correlation Analysis Results")
    print("=" * 55)

    for metric_col, label in METRICS.items():
        res = results[metric_col]
        print(f"\n[{label}]")
        print(f"  Sample size: {res['n']}")
        if res["r"] is not None:
            print(f"  Correlation (r): {res['r']:.3f}")
            print(f"  p-value: {res['p_value']:.4f}")
            print(f"  Interpretation: {interpret_r(res['r'])}")
        print(f"  Status: {res['status']}")

    print("\n" + "=" * 55)

    total_n = max(r["n"] for r in results.values())
    if total_n < MIN_PITCHES_FOR_CORRELATION:
        print(f"\nNote: Current sample size is small ({total_n} pitches),")
        print(f"so results above are preliminary. At least {MIN_PITCHES_FOR_CORRELATION}")
        print("pitches (ideally 30+) are recommended for a reliable conclusion.")
    print("=" * 55)


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("Starting velocity correlation analysis...\n")

    if not Path(CSV_PATH).exists():
        print(f"Error: could not find '{CSV_PATH}'.")
        print("Make sure pitching_data.csv is in the same folder.")
        exit(1)

    df = load_data(CSV_PATH)
    print(f"Read {len(df)} row(s) from CSV (including rows with blank values).\n")

    results = {}
    for metric_col, label in METRICS.items():
        corr_result = compute_correlation(df, metric_col)
        results[metric_col] = corr_result

        save_path = OUTPUT_DIR / f"correlation_{metric_col}.png"
        plot_scatter(df, metric_col, label, corr_result, save_path)

    plot_combined_summary(results, OUTPUT_DIR / "correlation_summary.png")

    print_summary(results)
