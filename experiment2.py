"""
experiment2.py — End-to-End Scheduling Comparison (Experiment 2)

Reads the saved CSV results from your baseline and ours runs in saved_results/,
computes mean and p90 e2e-time per scheduler, computes percentage improvement
over FIFO, saves results to CSV, and generates a bar chart.

Run your experiments first using run.py, then run this script.

Expected files in --results_dir (one per seed per scheduler):
    exp_k<K>_seed<SEED>_baseline.csv   — FIFO results
    exp_k<K>_seed<SEED>_ours.csv       — IconqSched results

Usage:
    python experiment2.py \
        --results_dir saved_results \
        --k 4 \
        --seeds 11 12 13 \
        --output_dir results/experiment2
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv_results(path: str):
    """
    Load a result CSV and return (e2e_times, exec_times, n_timeouts, n_errors).
    Filters out rows with errors or timeouts before computing metrics.
    """
    df = pd.read_csv(path)

    # filter out errors and timeouts
    if "error" in df.columns:
        df = df[df["error"] != True]
        df = df[df["error"] != "True"]
    if "timeout" in df.columns:
        df = df[df["timeout"] != True]
        df = df[df["timeout"] != "True"]

    # e2e-time is run_time_s, system runtime is exec_time
    e2e  = df["run_time_s"].values.astype(float)
    exec_t = df["exec_time"].values.astype(float) if "exec_time" in df.columns else e2e

    # count before filtering for reporting
    n_timeouts = int((df.get("timeout", pd.Series(dtype=bool)) == True).sum()) if "timeout" in df.columns else 0
    n_errors   = int((df.get("error",   pd.Series(dtype=bool)) == True).sum()) if "error"   in df.columns else 0

    # keep only positive runtimes
    valid = (e2e > 0) & (exec_t > 0)
    return e2e[valid], exec_t[valid], n_timeouts, n_errors


def collect_seed_results(results_dir: str, k: int, seeds: list, mode: str):
    """
    Collect all seed CSVs for a given K and mode (baseline or ours).
    Returns a flat array of all e2e_times across all seeds.
    """
    all_e2e = []
    found = []
    for seed in seeds:
        pattern = os.path.join(results_dir, f"exp_k{k}_seed{seed}_{mode}.csv")
        matches = glob.glob(pattern)
        if not matches:
            # also try the default naming from run.py
            pattern2 = os.path.join(results_dir, f"clients_{k}_timeout_*_{mode}.csv")
            matches = glob.glob(pattern2)
        if matches:
            path = matches[0]
            e2e, _, n_to, n_err = load_csv_results(path)
            all_e2e.append(e2e)
            found.append(f"seed {seed}: {len(e2e)} queries, {n_to} timeouts, {n_err} errors")
        else:
            print(f"  Warning: no file found for k={k} seed={seed} mode={mode}")
    if found:
        for msg in found:
            print(f"  {mode}: {msg}")
    return np.concatenate(all_e2e) if all_e2e else np.array([])


def compute_e2e_stats(e2e: np.ndarray, name: str):
    """Print and return stats dict."""
    if len(e2e) == 0:
        print(f"  {name}: no data")
        return {}
    stats = {
        "mean":   float(np.mean(e2e)),
        "p50":    float(np.percentile(e2e, 50)),
        "p90":    float(np.percentile(e2e, 90)),
        "p95":    float(np.percentile(e2e, 95)),
        "n":      len(e2e),
    }
    print(f"  {name}: mean={stats['mean']:.3f}s  "
          f"p50={stats['p50']:.3f}s  "
          f"p90={stats['p90']:.3f}s  "
          f"p95={stats['p95']:.3f}s  "
          f"(n={stats['n']})")
    return stats


def plot_results(stats: dict, output_dir: str):
    """
    stats = {
        scheduler_name: {"mean": ..., "p50": ..., "p90": ..., "p95": ...}
    }
    Generates two plots: mean e2e-time and p90 e2e-time.
    """
    os.makedirs(output_dir, exist_ok=True)
    schedulers = list(stats.keys())
    colors     = ["#888780", "#2d4a8a", "#1D9E75"][:len(schedulers)]

    for metric, label in [("mean", "Mean e2e-time (s)"), ("p90", "p90 tail e2e-time (s)")]:
        vals = [stats[s].get(metric, 0) for s in schedulers]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(schedulers, vals, color=colors, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(vals),
                f"{val:.2f}s",
                ha="center", va="bottom", fontsize=9
            )
        # add improvement annotation over FIFO if FIFO is present
        if "FIFO" in stats and len(schedulers) > 1:
            fifo_val = stats["FIFO"].get(metric, 0)
            for i, s in enumerate(schedulers):
                if s != "FIFO" and fifo_val > 0:
                    improvement = (fifo_val - vals[i]) / fifo_val * 100
                    sign = "+" if improvement < 0 else ""
                    # negative improvement means worse, positive means better
                    ax.text(
                        i, vals[i] + 0.04 * max(vals),
                        f"{improvement:.1f}% vs FIFO",
                        ha="center", fontsize=8,
                        color="#2d4a8a" if improvement > 0 else "#A32D2D"
                    )
        ax.set_ylabel(label)
        ax.set_title(f"Experiment 2 — {label}")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f"experiment2_{metric}_e2e.pdf")
        plt.savefig(path)
        plt.close()
        print(f"Plot saved to {path}")


def save_summary_csv(stats: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    fifo_mean = stats.get("FIFO", {}).get("mean", None)
    fifo_p90  = stats.get("FIFO", {}).get("p90",  None)
    for scheduler, s in stats.items():
        row = {"scheduler": scheduler}
        row.update({k: round(v, 4) for k, v in s.items() if isinstance(v, float)})
        if fifo_mean and scheduler != "FIFO" and s.get("mean"):
            row["pct_improvement_mean"] = round((fifo_mean - s["mean"]) / fifo_mean * 100, 2)
        if fifo_p90 and scheduler != "FIFO" and s.get("p90"):
            row["pct_improvement_p90"]  = round((fifo_p90  - s["p90"])  / fifo_p90  * 100, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "experiment2_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSummary saved to {path}")
    return df


# ── statistical test: paired t-test ─────────────────────────────────────────────────────────────────────
def run_paired_ttest(results_dir: str, k: int, seeds: list, output_dir: str):
    """
    Run paired t-test comparing FIFO vs IconqSched mean e2e-time per seed.
    Saves results to experiment2_ttest.csv
    """
    fifo_means = []
    ours_means = []
 
    for seed in seeds:
        fifo_path = os.path.join(results_dir, f"exp_k{k}_seed{seed}_baseline.csv")
        ours_path = os.path.join(results_dir, f"exp_k{k}_seed{seed}_ours.csv")
        if os.path.exists(fifo_path) and os.path.exists(ours_path):
            f_e2e, _, _, _ = load_csv_results(fifo_path)
            o_e2e, _, _, _ = load_csv_results(ours_path)
            if len(f_e2e) > 0 and len(o_e2e) > 0:
                fifo_means.append(float(np.mean(f_e2e)))
                ours_means.append(float(np.mean(o_e2e)))
 
    print("\n" + "="*60)
    print("  PAIRED T-TEST — FIFO vs IconqSched")
    print("="*60)
 
    if len(fifo_means) < 2:
        print("  Not enough seeds for t-test (need at least 2)")
        return
 
    t_stat, p_value = scipy_stats.ttest_rel(fifo_means, ours_means)
    significant = p_value < 0.05
 
    print(f"  Seeds used:       {len(fifo_means)}")
    print(f"  FIFO means:       {[round(x,3) for x in fifo_means]}")
    print(f"  IconqSched means: {[round(x,3) for x in ours_means]}")
    print(f"  t-statistic:      {t_stat:.4f}")
    print(f"  p-value:          {p_value:.4f}")
    print(f"  Significant:      {'Yes (p < 0.05)' if significant else 'No (p >= 0.05)'}")
 
    if not significant:
        print("  Note: improvement exists but not statistically significant")
        print("        with only 3 seeds. More seeds would strengthen this result.")
 
    ttest_df = pd.DataFrame([{
        "comparison":   "FIFO vs IconqSched",
        "n_seeds":      len(fifo_means),
        "t_statistic":  round(t_stat, 4),
        "p_value":      round(p_value, 4),
        "significant":  significant,
        "fifo_means":   str([round(x,3) for x in fifo_means]),
        "ours_means":   str([round(x,3) for x in ours_means]),
    }])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "experiment2_ttest.csv")
    ttest_df.to_csv(path, index=False)
    print(f"  T-test results saved to {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = args.seeds

    print(f"\nExperiment 2 — End-to-End Scheduling Comparison")
    print(f"K={args.k}  seeds={seeds}  results_dir={args.results_dir}")
    print("="*60)

    stats = {}

    # ── FIFO (baseline) ───────────────────────────────────────────────────────
    print(f"\nLoading FIFO (baseline) results...")
    fifo_e2e = collect_seed_results(args.results_dir, args.k, seeds, "baseline")
    if len(fifo_e2e) > 0:
        stats["FIFO"] = compute_e2e_stats(fifo_e2e, "FIFO")
    else:
        print("  No FIFO results found.")

    # ── IconqSched (ours) ─────────────────────────────────────────────────────
    print(f"\nLoading IconqSched (ours) results...")
    ours_e2e = collect_seed_results(args.results_dir, args.k, seeds, "ours")
    if len(ours_e2e) > 0:
        stats["IconqSched"] = compute_e2e_stats(ours_e2e, "IconqSched")
    else:
        print("  No IconqSched results found.")

    # ── SQF if present ────────────────────────────────────────────────────────
    print(f"\nLooking for SQF results (optional)...")
    sqf_e2e = collect_seed_results(args.results_dir, args.k, seeds, "sqf")
    if len(sqf_e2e) > 0:
        stats["SQF"] = compute_e2e_stats(sqf_e2e, "SQF")
        print("  SQF results found and loaded.")
    else:
        print("  No SQF results found — skipping.")

    if not stats:
        print("\nNo results found. Make sure you have run the experiments first.")
        print("Expected files like: saved_results/exp_k4_seed11_baseline.csv")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  SUMMARY TABLE")
    print("="*70)
    header = f"{'Scheduler':<20} {'Mean':>10} {'p50':>10} {'p90':>10} {'p95':>10} {'Improvement':>14}"
    print(header)
    print("-"*70)
    fifo_mean = stats.get("FIFO", {}).get("mean", None)
    fifo_p90  = stats.get("FIFO", {}).get("p90",  None)
    for scheduler, s in stats.items():
        if not s:
            continue
        imp_mean = ""
        if fifo_mean and scheduler != "FIFO":
            pct = (fifo_mean - s.get("mean", 0)) / fifo_mean * 100
            imp_mean = f"{pct:+.1f}% (mean)"
        row = (
            f"{scheduler:<20} "
            f"{s.get('mean',0):>10.3f}s "
            f"{s.get('p50', 0):>10.3f}s "
            f"{s.get('p90', 0):>10.3f}s "
            f"{s.get('p95', 0):>10.3f}s "
            f"{imp_mean:>14}"
        )
        print(row)

    if fifo_mean and "IconqSched" in stats:
        pct_mean = (fifo_mean - stats["IconqSched"]["mean"]) / fifo_mean * 100
        pct_p90  = (fifo_p90  - stats["IconqSched"]["p90"])  / fifo_p90  * 100 if fifo_p90 else 0
        print(f"\n  IconqSched vs FIFO:")
        print(f"    Mean e2e improvement: {pct_mean:.1f}%")
        print(f"    p90  e2e improvement: {pct_p90:.1f}%")

    # ── Paired t-test ─────────────────────────────────────────────────────────
    run_paired_ttest(args.results_dir, args.k, seeds, args.output_dir)

    # ── Save and plot ──────────────────────────────────────────────────────────
    summary_df = save_summary_csv(stats, args.output_dir)
    print(summary_df.to_string(index=False))
    plot_results(stats, args.output_dir)
    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2 — End-to-End Scheduling")
    parser.add_argument("--results_dir", default="saved_results", type=str,
                        help="Directory containing experiment result CSVs")
    parser.add_argument("--k",    default=4, type=int,
                        help="Number of concurrent clients used in experiments")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13],
                        help="Seeds used in experiments e.g. --seeds 11 12 13")
    parser.add_argument("--output_dir", default="results/experiment2", type=str)
    args = parser.parse_args()
    main(args)
  
