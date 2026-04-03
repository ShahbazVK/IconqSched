"""
experiment1.py — Predictor Accuracy Evaluation (Experiment 1)

Loads the trained Iconq model and Stage baseline, runs both on the
held-out test split, computes Q-error and absolute error at p50/p90/p95,
saves results to CSV, and generates a bar chart.

Usage:
    python experiment1.py \
        --model_name postgres_brad \
        --target_path models/_checkpoints \
        --directory saved_results \
        --parsed_queries_path workloads/postgres/brad_parsed_query_plans.json \
        --rnn_type bilstm \
        --output_dir results/experiment1
"""

import os
import sys
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.load_trace import load_all_csv_from_dir, create_concurrency_dataset
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray):
    """Return q_error and abs_error arrays."""
    preds  = np.maximum(preds,  1e-6)
    labels = np.maximum(labels, 1e-6)
    q_error  = np.maximum(preds / labels, labels / preds)
    abs_error = np.abs(preds - labels)
    return q_error, abs_error


def print_metrics(name: str, q_error: np.ndarray, abs_error: np.ndarray):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Q-error   — p50: {np.percentile(q_error,  50):.4f}  "
          f"p90: {np.percentile(q_error,  90):.4f}  "
          f"p95: {np.percentile(q_error,  95):.4f}")
    print(f"  Abs-error — p50: {np.percentile(abs_error, 50):.4f}s  "
          f"p90: {np.percentile(abs_error, 90):.4f}s  "
          f"p95: {np.percentile(abs_error, 95):.4f}s")


def heuristic_predict(df: pd.DataFrame) -> np.ndarray:
    """Predict average runtime per query template (heuristic baseline)."""
    avg_by_idx = df.groupby("query_idx")["runtime"].mean().to_dict()
    overall_avg = df["runtime"].mean()
    preds = np.array([
        avg_by_idx.get(int(row["query_idx"]), overall_avg)
        for _, row in df.iterrows()
    ])
    return preds


def save_results_csv(results: dict, output_dir: str):
    """Save per-query results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for method, (q_err, abs_err) in results.items():
        for i in range(len(q_err)):
            rows.append({
                "method":    method,
                "q_error":   round(float(q_err[i]),   4),
                "abs_error": round(float(abs_err[i]), 4),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "experiment1_results.csv")
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")


def plot_results(summary: dict, output_dir: str):
    """
    summary = {
        method_name: {
            "q_p50": ..., "q_p90": ..., "q_p95": ...,
            "a_p50": ..., "a_p90": ..., "a_p95": ...
        }
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    methods   = list(summary.keys())
    percentiles = ["p50", "p90", "p95"]
    x = np.arange(len(percentiles))
    width = 0.25
    colors = ["#2d4a8a", "#1D9E75", "#BA7517"]

    # ── Q-error plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [summary[method][f"q_{p}"] for p in percentiles]
        bars = ax.bar(x + i * width, vals, width, label=method, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8
            )
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Q-error (lower is better)")
    ax.set_title("Experiment 1 — Predictor Q-error")
    ax.set_xticks(x + width)
    ax.set_xticklabels(percentiles)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    qerr_path = os.path.join(output_dir, "experiment1_qerror.pdf")
    plt.savefig(qerr_path)
    plt.close()
    print(f"Q-error plot saved to {qerr_path}")

    # ── Absolute-error plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (method, color) in enumerate(zip(methods, colors)):
        vals = [summary[method][f"a_{p}"] for p in percentiles]
        bars = ax.bar(x + i * width, vals, width, label=method, color=color)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}s",
                ha="center", va="bottom", fontsize=8
            )
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error in seconds (lower is better)")
    ax.set_title("Experiment 1 — Predictor Absolute Error")
    ax.set_xticks(x + width)
    ax.set_xticklabels(percentiles)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    abserr_path = os.path.join(output_dir, "experiment1_abserror.pdf")
    plt.savefig(abserr_path)
    plt.close()
    print(f"Absolute error plot saved to {abserr_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load traces and build concurrency dataset ──────────────────────────
    print("\nLoading traces...")
    all_trace = load_all_csv_from_dir(args.directory)
    all_concurrency_df = []
    for trace in all_trace:
        cdf = create_concurrency_dataset(trace, engine=None, pre_exec_interval=None)
        all_concurrency_df.append(cdf)
    concurrency_df = pd.concat(all_concurrency_df, ignore_index=True)

    # ── 2. Chronological train / test split (70 / 30) ────────────────────────
    print(f"Total samples: {len(concurrency_df)}")
    split_idx  = int(0.7 * len(concurrency_df))
    train_df   = concurrency_df.iloc[:split_idx].copy()
    test_df    = concurrency_df.iloc[split_idx:].copy()
    # only evaluate on queries that have at least one concurrent query
    test_df    = test_df[test_df["num_concurrent_queries"] > 0].copy()
    print(f"Train: {len(train_df)}  |  Test (concurrent only): {len(test_df)}")

    # ── 3. Load Stage model ───────────────────────────────────────────────────
    print("\nLoading Stage model...")
    stage_path = os.path.join(args.target_path, f"{args.model_name}_stage_model.pkl")
    with open(stage_path, "rb") as f:
        ss: SingleStage = pkl.load(f)
    print(f"Stage model loaded from {stage_path}")

    # ── 4. Load Iconq bi-LSTM ─────────────────────────────────────────────────
    print("\nLoading Iconq bi-LSTM...")
    rnn = ConcurrentRNN(
        ss,
        model_prefix=args.model_name,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_separation=False,
        ignore_short_running=args.ignore_short_running,
        short_running_threshold=args.short_running_threshold,
    )
    rnn.load_model(args.target_path)
    print("Iconq loaded.")

    # ── 5. Iconq predictions on test set ─────────────────────────────────────
    print("\nRunning Iconq predictions on test set...")
    iconq_preds, iconq_labels = rnn.predict(test_df, return_per_query=False)
    iconq_q, iconq_a = compute_metrics(iconq_preds, iconq_labels)
    print_metrics("Iconq (bi-LSTM)", iconq_q, iconq_a)

    # ── 6. Stage predictions on test set ─────────────────────────────────────
    # Stage.predict() takes a DataFrame with a "features" column.
    # We need to featurize the test_df first using the loaded stage model.
    print("\nRunning Stage (single-query MLP) predictions on test set...")
    # add features column to test_df using the already-loaded stage model
    all_query_idx = test_df["query_idx"].values
    features_list = [ss.all_feature[int(q)] for q in all_query_idx]
    test_df_stage = test_df.copy()
    test_df_stage["features"] = features_list
    stage_preds, stage_labels = ss.evaluate(test_df_stage)
    stage_q, stage_a = compute_metrics(stage_preds, stage_labels)
    print_metrics("Stage (single-query MLP)", stage_q, stage_a)

    # ── 7. Heuristic baseline ─────────────────────────────────────────────────
    print("\nRunning heuristic baseline (average runtime per template)...")
    heur_preds  = heuristic_predict(test_df)
    heur_labels = test_df["runtime"].values
    heur_q, heur_a = compute_metrics(heur_preds, heur_labels)
    print_metrics("Heuristic (avg per template)", heur_q, heur_a)

    # ── 8. Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    header = f"{'Method':<30} {'Q-p50':>8} {'Q-p90':>8} {'Q-p95':>8} {'A-p50':>8} {'A-p90':>8} {'A-p95':>8}"
    print(header)
    print("-"*60)
    for name, (q, a) in [
        ("Iconq (bi-LSTM)",           (iconq_q, iconq_a)),
        ("Stage (single-query MLP)",  (stage_q, stage_a)),
        ("Heuristic",                 (heur_q,  heur_a)),
    ]:
        row = (
            f"{name:<30} "
            f"{np.percentile(q,50):>8.3f} "
            f"{np.percentile(q,90):>8.3f} "
            f"{np.percentile(q,95):>8.3f} "
            f"{np.percentile(a,50):>8.3f}s "
            f"{np.percentile(a,90):>8.3f}s "
            f"{np.percentile(a,95):>8.3f}s"
        )
        print(row)

    # ── 9. Save CSV ───────────────────────────────────────────────────────────
    save_results_csv({
        "Iconq":      (iconq_q, iconq_a),
        "Stage":      (stage_q, stage_a),
        "Heuristic":  (heur_q,  heur_a),
    }, args.output_dir)

    # ── 10. Save summary CSV ──────────────────────────────────────────────────
    summary = {
        "Iconq":     {"q_p50": np.percentile(iconq_q,50), "q_p90": np.percentile(iconq_q,90), "q_p95": np.percentile(iconq_q,95),
                      "a_p50": np.percentile(iconq_a,50), "a_p90": np.percentile(iconq_a,90), "a_p95": np.percentile(iconq_a,95)},
        "Stage":     {"q_p50": np.percentile(stage_q,50), "q_p90": np.percentile(stage_q,90), "q_p95": np.percentile(stage_q,95),
                      "a_p50": np.percentile(stage_a,50), "a_p90": np.percentile(stage_a,90), "a_p95": np.percentile(stage_a,95)},
        "Heuristic": {"q_p50": np.percentile(heur_q,50),  "q_p90": np.percentile(heur_q,90),  "q_p95": np.percentile(heur_q,95),
                      "a_p50": np.percentile(heur_a,50),  "a_p90": np.percentile(heur_a,90),  "a_p95": np.percentile(heur_a,95)},
    }
    summary_rows = []
    for method, vals in summary.items():
        row = {"method": method}
        row.update({k: round(v, 4) for k, v in vals.items()})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "experiment1_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # ── 11. Plot ──────────────────────────────────────────────────────────────
    plot_results(summary, args.output_dir)
    print("\nExperiment 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1 — Predictor Accuracy")
    parser.add_argument("--model_name",    default="postgres_brad", type=str)
    parser.add_argument("--target_path",   default="models/_checkpoints", type=str)
    parser.add_argument("--directory",     required=True, type=str,
                        help="Directory containing saved_results CSV files for training traces")
    parser.add_argument("--parsed_queries_path", required=True, type=str)
    parser.add_argument("--rnn_type",      default="bilstm", type=str)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_size",   default=256, type=int)
    parser.add_argument("--num_layers",    default=2,   type=int)
    parser.add_argument("--ignore_short_running",   action="store_true")
    parser.add_argument("--short_running_threshold", default=5.0, type=float)
    parser.add_argument("--output_dir",    default="results/experiment1", type=str)
    args = parser.parse_args()
    main(args)
    