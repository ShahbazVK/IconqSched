# CS 6360 Project - IconqSched Reproduction: Bi-LSTM Based Concurrent Query Scheduling

Reproduction of IconqSched from Z. Wu et al. 2025 (VLDB). 

Our project evaluates Iconq bi-LSTM prediction accuracy and IconqSched greedy scheduling performance on local PostgreSQL 16 using the IMDB JOB dataset with BRAD queries at a reduced scale of ~10 GB compared to the paper's 160 GB configuration. 

---

### Standard Workflow

1. Setup environment and database.
2. Warmup once.
3. Run baseline seeds (K=4).
4. Build clean training folder from baseline CSVs only.
5. Train checkpoints.
6. Run ours on the same seeds and same config.
7. Run Exp1 and Exp2.

Core rule: use one checkpoint set for both Experiment 1 and Experiment 2 in one report cycle.

---

## Setup

### Requirements
- Python 3.11
- PostgreSQL 16
- IMDB JOB dataset loaded into local Postgres

### Environment Setup: Install dependencies

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Thread stability (recommended on macOS):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

## Postgres + Dataset Setup

Check DB:

```bash
psql "host=127.0.0.1 port=5432 dbname=imdb user=<user>" -c "\conninfo"
```

Download JOB dataset:

```bash
cd <download-dir>
curl -fL -o imdb.tgz "https://event.cwi.nl/da/job/imdb.tgz"
tar -xzf imdb.tgz
ls <data-dir>/*.csv | wc -l
```

Create schema:

```bash
cd <repo-root>
python utils/load_database.py --schema-only --no-drop --user <user> --password <password> --database imdb
```

Load CSV data:

```bash
python utils/load_imdb_csvs_python.py --host 127.0.0.1 --port 5432 --user <user> --password <password> --database imdb --data-dir <data-dir>
```

---

## Query File Normalization (Only If Needed)

```bash
python3 - <<'PY'
src = "workloads/postgres/brad_queries.sql"
dst = "workloads/postgres/brad_queries_normalized.sql"
txt = open(src).read()
parts = [q.strip() for q in txt.split(";") if q.strip()]
with open(dst, "w") as f:
    for q in parts:
        f.write(q + ";\n\n")
print("wrote", dst, "queries", len(parts))
PY
```

---

## Running the Pipeline

Follow these steps in order on one designated machine.

### 1. Warmup (Run once)

```bash
mkdir -p saved_results
python run.py --warmup_run --database postgres --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <user> --password <password> --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --timeout_s 1000
```

Expected: `saved_results/_timeout_1000_warmup_run.csv`

### 2. Baseline seeds (K=4)

Use the same seeds each cycle:

```bash
for SEED in 11 12 13 21 22 23 24 25; do
  echo "=== Baseline seed ${SEED} ==="
  python -u run.py --run_k_client_in_parallel --baseline --scheduler_type None --database postgres --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <user> --password <password> --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 4 --timeout_s 120 --exec_for_s 300 --seed "${SEED}"
done
```

Auto-snapshots are created automatically (`exp_k4_seed*_baseline.csv`).

### 3. Build clean training folder

```bash
rm -rf training_traces_baseline_k4
mkdir -p training_traces_baseline_k4
for SEED in 11 12 13 21 22 23 24 25; do
  cp "saved_results/exp_k4_seed${SEED}_baseline.csv" training_traces_baseline_k4/
done
```

Optional sanity check (exact sample counts before training):

```bash
python3 - <<'PY'
import glob, os
import pandas as pd
from utils.load_trace import create_concurrency_dataset

trace_dir = "training_traces_baseline_k4"
files = sorted(glob.glob(os.path.join(trace_dir, "*.csv")))
print(f"trace_files={len(files)}")
for f in files:
    print(" -", os.path.basename(f))

all_concurrency = []
for f in files:
    df = pd.read_csv(f)
    all_concurrency.append(create_concurrency_dataset(df, engine=None, pre_exec_interval=None))

concurrency_df = pd.concat(all_concurrency, ignore_index=True)
n = len(concurrency_df)
train_n = int(0.8 * n)
eval_n = n - train_n
eval_concurrent_only = len(concurrency_df.iloc[train_n:][concurrency_df.iloc[train_n:]["num_concurrent_queries"] > 0])

print(f"total_concurrency_samples={n}")
print(f"approx_train_samples={train_n}")
print(f"approx_eval_samples={eval_n}")
print(f"approx_eval_samples_concurrent_only={eval_concurrent_only}")
PY
```

### 4. Train Checkpoints

```bash
mkdir -p models/_checkpoints
python -u run.py --train_concurrent_rnn --model_name postgres_brad --directory training_traces_baseline_k4 --parsed_queries_path workloads/postgres/brad_parsed_query_plans.json --target_path models/_checkpoints --rnn_type bilstm --use_size --use_log --use_table_features --epochs 30
```

Expected artifacts:

- `models/_checkpoints/postgres_brad_stage_model.pkl`
- `models/_checkpoints/postgres_brad_bilstm_256_2_q_loss_wo_sep`

### 5. Run IconqSched seeds 

Ours runs with same seeds and same configuration.
Do not add `--debug` for timed comparisons.

```bash
for SEED in 11 12 13 21 22 23 24 25; do
  echo "=== Ours seed ${SEED} ==="
  python -u run.py --run_k_client_in_parallel --database postgres --model_name postgres_brad --target_path models/_checkpoints --rnn_type bilstm --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <user> --password <password> --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 4 --timeout_s 120 --exec_for_s 300 --scheduler_type greedy --seed "${SEED}"
done
```

Auto-snapshots are created automatically (`exp_k4_seed*_ours.csv`).

### 6. Run experiment scripts

Experiment 1:

```bash
python experiment1.py --model_name postgres_brad --target_path models/_checkpoints --directory training_traces_baseline_k4 --rnn_type bilstm --output_dir results/experiment1_clean_k4
```

Experiment 2:

```bash
python experiment2.py --results_dir saved_results --k 4 --seeds 11 12 13 21 22 23 24 25 --output_dir results/experiment2_clean_k4
```

Note: `--seeds` must be space-separated integers (not comma-separated).

---

## Metrics

**Experiment 1 — Predictor Accuracy**
- Q-error: max(predicted/actual, actual/predicted) — closer to 1 is better
- Absolute error: |predicted − actual| in seconds — closer to 0 is better
- Both reported at p50, p90, and p95

**Experiment 2 — End-to-End Scheduling**
- Mean e2e-time: queuing time + system runtime per query
- p90 e2e-time: tail latency at 90th percentile
- Percentage improvement over FIFO: (FIFO − IconqSched) / FIFO × 100
- Statistical test: paired t-test across 8 seeds, p < 0.05 threshold
- Supplemental: p50 and p95 e2e-time

---

## Notes

- Keep `num_clients=4`, `timeout_s=120`, `exec_for_s=300` identical for baseline and ours.
- Use the same seed list for both baseline and ours runs.
- Do not mix old CSVs into `training_traces_baseline_k4`.
- Clear Postgres cache between runs for consistent results.
- Do not mix checkpoint sets across Experiment1 and Experiment2 in one report cycle.
