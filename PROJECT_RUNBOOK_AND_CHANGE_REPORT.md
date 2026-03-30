# IconqSched: Final Change Report and Runbook

This document explains exactly what changed, why it changed, how to run the project from scratch, what outputs to expect, and how to fix common errors.

It is written for team members with little or no background in this repo.

---

## 1) What Changed

### A. Modified files

1. `models/concurrency/complex_models.py`

- Changed training logic to use the full `stage_model.predictions` map (when available) instead of only `cache.running_average`.

1. `utils/load_database.py`

- Converted from a hardcoded helper script into a runnable CLI utility.
- Added flags/options for local usage (`--host`, `--port`, `--user`, `--password`, `--database`, `--data-dir`, etc.).
- Added safety and control flags (`--no-drop`, `--schema-only`, `--skip-schema`, `--skip-index-print`, `--reset-public-schema`).
- Added clearer errors and guidance.
- Added direct-script import handling so it works when run from repo root.

1. `workloads/postgres/imdb_schema.py`

- Fixed index typo: `movi44e_companies` -> `movie_companies`.
- Updated `IMDB_LOAD_TEMPLATE` to use CSV format consistent with your downloaded JOB files.

### B. New file added

1. `utils/load_imdb_csvs_python.py` (new)

- Added robust Python-based loader for IMDB CSVs when direct `psql \copy` fails on malformed lines.

1. `requirements.txt` (new)

- Added reproducible Python dependency list for `.venv` setup (`pip install -r requirements.txt`).

### C. Removed/temporary files cleanup

Temporary artifacts were removed during cleanup (generated logs, temporary traces, checkpoint test outputs, etc.).

Important: runtime outputs (like `saved_results/*`) are expected to be regenerated when you run experiments.

### D. Final git-visible state (code-level)

Current expected custom code deltas:

- `models/concurrency/complex_models.py` (modified)
- `utils/load_database.py` (modified)
- `workloads/postgres/imdb_schema.py` (modified)
- `utils/load_imdb_csvs_python.py` (new)
- `requirements.txt` (new)

Also present in your tree:

- `tpch-dbgen` (untracked repo artifact; optional for current IMDB workflow)

---

## 2) Why These Changes Were Needed

### Why the original repo failed in your setup

The repository is research-oriented, not a plug-and-play production package. Multiple assumptions in the original code did not hold in your local environment:

1. **Data loading script was not operational for local setup by default**

- `utils/load_database.py` had cloud placeholder credentials and was not integrated into `run.py`.
- It was more of an instruction script than a ready CLI tool.

1. **IMDB CSV import path was fragile**

- Your JOB CSV dump contained malformed/problematic rows (quotes/newlines/field-width mismatches).
- Direct `\copy` failed repeatedly on those rows.

1. **Training bug with sparse prediction map**

- During RNN featurization, concurrent context referenced query IDs not present in `cache.running_average`.
- This caused `KeyError` (for example, query id `158`).

1. **Query file formatting mismatch**

- Parts of the code parse SQL banks by splitting on `;\n\n`.
- Raw `brad_queries.sql` formatting may not always match this expectation, depending on path.

1. **Research runtime behavior can look like “hang”**

- Some paths are silent for long intervals.
- Replay mode can pause based on trace timing.
- Debug visibility is limited unless explicit logging is enabled.

### Why it may work for others but not for you

Different users may have:

- different IMDB file variants,
- pre-cleaned datasets,
- prebuilt DB snapshots,
- precomputed artifacts/checkpoints,
- different PostgreSQL/OS behavior,
- different pipeline scripts not committed in this repo.

So a workflow that worked for one environment can fail in another.

---

## 3) What Each Fix Does

### Fix 1: `complex_models.py`

**Broken behavior**

- Training crashed with `KeyError` for missing query IDs during concurrency featurization.

**Fix**

- Use `self.stage_model.predictions` (full query prediction map) when available.
- Fallback to `cache.running_average` only if full map is not available.

**Why this is correct**

- Concurrent featurization can reference more query IDs than those in the immediate training subset.
- Full prediction map is the right source for complete query-id coverage.

---

### Fix 2: `load_database.py`

**Broken behavior**

- Script was hardcoded and awkward for local execution.

**Fix**

- Added proper CLI and run modes:
  - create schema,
  - print load SQL,
  - avoid destructive drop,
  - reset only schema when needed.

**Why this is correct**

- Makes setup reproducible and explicit, without editing source every time.

---

### Fix 3: `imdb_schema.py`

**Broken behavior**

- Index DDL typo broke index creation.
- Load template did not match your CSV characteristics.

**Fix**

- Corrected table name typo.
- Set `IMDB_LOAD_TEMPLATE` to CSV mode compatible with your JOB archive files.

**Why this is correct**

- Typo was objectively invalid SQL.
- CSV mode is appropriate for your source format.

---

### Fix 4: `load_imdb_csvs_python.py` (new)

**Broken behavior**

- Raw `\copy` failed on malformed rows.

**Fix**

- Added Python parsing + controlled loading path with error tolerance and row-skip reporting.

**Why this is correct**

- Provides a deterministic import path for imperfect research data dumps.

---

## 4) How to Run the Project (Step-by-Step)

This is the beginner path that worked in your environment.

### Step 0: Environment

Python version required for this runbook:

- **Python 3.11.x** (tested with 3.11)

If this is a fresh machine, create and use a local virtual environment first.

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This project now includes a `requirements.txt` for reproducible setup.

Important dependency note:

- `requirements.txt` pins `numpy<2` because some PyTorch wheels used in this workflow are built against NumPy 1.x.
- If you install NumPy 2.x, you may see errors like:
  - `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

Optional (stability):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

---

### Step 1: Start/verify local Postgres

Example connectivity check:

```bash
psql "host=127.0.0.1 port=5432 dbname=imdb user=nizarnoorani" -c "\conninfo"
```

Placeholder note:

- Replace placeholders with real values before running commands.
- Do **not** paste `nizarnoorani`, `<K>`, `<SEED>` literally in zsh.

---

### Step 2: Download and extract IMDB JOB data

```bash
cd <download-dir>
curl -fL -o imdb.tgz "https://event.cwi.nl/da/job/imdb.tgz"
tar -xzf imdb.tgz
ls <data-dir>/*.csv | wc -l
```

Expected CSV count: around 21 files.

---

### Step 3: Create schema

```bash
cd <repo-root>
python utils/load_database.py \
  --schema-only \
  --no-drop \
  --user nizarnoorani \
  --password '' \
  --database imdb
```

If tables already exist, either keep them or reset public schema intentionally:

```bash
python utils/load_database.py --schema-only --no-drop --reset-public-schema --user nizarnoorani --password '' --database imdb
```

---

### Step 4: Load data (robust method)

```bash
python utils/load_imdb_csvs_python.py \
  --host 127.0.0.1 --port 5432 --user nizarnoorani --password '' \
  --database imdb \
  --data-dir <data-dir>
```

Verify:

```bash
psql "host=127.0.0.1 port=5432 dbname=imdb user=nizarnoorani" -c "SELECT COUNT(*) FROM title;"
```

---

### Step 5: Normalize query bank (if needed)

```bash
python3 - <<'PY'
src="workloads/postgres/brad_queries.sql"
dst="workloads/postgres/brad_queries_normalized.sql"
txt=open(src).read()
parts=[q.strip() for q in txt.split(";") if q.strip()]
with open(dst,"w") as f:
    for q in parts:
        f.write(q+";\n\n")
print("wrote", dst, "queries", len(parts))
PY
```

---

### Step 6: Warmup

```bash
mkdir -p saved_results
python run.py \
  --warmup_run \
  --database postgres \
  --save_result_dir saved_results \
  --host 127.0.0.1 --port 5432 --user nizarnoorani --password '' \
  --db_name imdb \
  --query_bank_path workloads/postgres/brad_queries_normalized.sql \
  --timeout_s 1000
```

---

### Step 7: Collect baseline training trace

```bash
python run.py \
  --run_k_client_in_parallel \
  --baseline \
  --scheduler_type None \
  --database postgres \
  --save_result_dir saved_results \
  --host 127.0.0.1 --port 5432 --user nizarnoorani --password '' \
  --db_name imdb \
  --query_bank_path workloads/postgres/brad_queries_normalized.sql \
  --num_clients 4 \
  --timeout_s 120 \
  --exec_for_s 900
```

---

### Step 8: Train model

```bash
mkdir -p models/_checkpoints
python run.py \
  --train_concurrent_rnn \
  --model_name postgres_brad \
  --directory saved_results \
  --parsed_queries_path workloads/postgres/brad_parsed_query_plans.json \
  --target_path models/_checkpoints \
  --rnn_type bilstm \
  --use_size \
  --use_log \
  --use_table_features \
  --ignore_short_running \
  --epochs 20
```

What Step 8 produces:

- Stage model file:
  - `models/_checkpoints/postgres_brad_stage_model.pkl`
- Concurrent model file:
  - `models/_checkpoints/postgres_brad_bilstm_256_2_q_loss_wo_sep`

Where this trained model is used:

- It is loaded in **Ours** runs (Step 9, `--scheduler_type greedy`) using:
  - `--model_name postgres_brad`
  - `--target_path models/_checkpoints`
- Baseline runs do **not** use trained model files.

---

### Step 9: Run baseline vs ours experiments (stable path)

Use `run_k_client_in_parallel` for comparisons (it was more stable than replay mode on your machine).

Quick command meaning (cheat sheet):

- `--train_concurrent_rnn` = trains and saves model files (Step 8).
- `--run_k_client_in_parallel --baseline --scheduler_type None` = baseline experiment only (no learned model).
- `--run_k_client_in_parallel --scheduler_type greedy --model_name ... --target_path ...` = ours experiment (loads trained model and schedules with it).

Baseline template:

```bash
python run.py \
  --run_k_client_in_parallel \
  --baseline \
  --scheduler_type None \
  --database postgres \
  --save_result_dir saved_results \
  --host 127.0.0.1 --port 5432 --user nizarnoorani --password '' \
  --db_name imdb \
  --query_bank_path workloads/postgres/brad_queries_normalized.sql \
  --num_clients <K> \
  --timeout_s 120 \
  --exec_for_s 180 \
  --seed <SEED>
```

Ours template:

```bash
python run.py \
  --run_k_client_in_parallel \
  --database postgres \
  --model_name postgres_brad \
  --target_path models/_checkpoints \
  --rnn_type bilstm \
  --save_result_dir saved_results \
  --host 127.0.0.1 --port 5432 --user nizarnoorani --password '' \
  --db_name imdb \
  --query_bank_path workloads/postgres/brad_queries_normalized.sql \
  --num_clients <K> \
  --timeout_s 120 \
  --exec_for_s 180 \
  --ignore_short_running \
  --scheduler_type greedy \
  --seed <SEED> \
  --debug
```

Replace `<K>` and `<SEED>` with real numbers (no angle brackets in zsh).

#### Step 9A: Keep each run result (important)

By default, each new run overwrites:

- `saved_results/clients_<K>_timeout_120_baseline.csv`
- `saved_results/clients_<K>_timeout_120_ours.csv`

So after each run, copy it to a seed-specific filename.

Example for `K=1`, `SEED=11`:

```bash
# after baseline run
cp saved_results/clients_1_timeout_120_baseline.csv saved_results/exp_k1_seed11_baseline.csv

# after ours run
cp saved_results/clients_1_timeout_120_ours.csv saved_results/exp_k1_seed11_ours.csv
```

Example for `K=4`, `SEED=12`:

```bash
# after baseline run
cp saved_results/clients_4_timeout_120_baseline.csv saved_results/exp_k4_seed12_baseline.csv

# after ours run
cp saved_results/clients_4_timeout_120_ours.csv saved_results/exp_k4_seed12_ours.csv
```

#### Step 9B: Recommended comparison loop

Run this pair for each seed you want (for example 11, 12, 13):

1. baseline command
2. copy baseline CSV to `exp_k..._seed..._baseline.csv`
3. ours command
4. copy ours CSV to `exp_k..._seed..._ours.csv`

This guarantees you keep every run and can compare fairly.

#### Step 9C: Quick command block (K=1, seeds 11/12/13)

```bash
# Seed 11 baseline
python run.py --run_k_client_in_parallel --baseline --scheduler_type None --database postgres --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --seed 11
cp saved_results/clients_1_timeout_120_baseline.csv saved_results/exp_k1_seed11_baseline.csv

# Seed 11 ours
python run.py --run_k_client_in_parallel --database postgres --model_name postgres_brad --target_path models/_checkpoints --rnn_type bilstm --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --ignore_short_running --scheduler_type greedy --seed 11 --debug
cp saved_results/clients_1_timeout_120_ours.csv saved_results/exp_k1_seed11_ours.csv

# Seed 12 baseline
python run.py --run_k_client_in_parallel --baseline --scheduler_type None --database postgres --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --seed 12
cp saved_results/clients_1_timeout_120_baseline.csv saved_results/exp_k1_seed12_baseline.csv

# Seed 12 ours
python run.py --run_k_client_in_parallel --database postgres --model_name postgres_brad --target_path models/_checkpoints --rnn_type bilstm --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --ignore_short_running --scheduler_type greedy --seed 12 --debug
cp saved_results/clients_1_timeout_120_ours.csv saved_results/exp_k1_seed12_ours.csv

# Seed 13 baseline
python run.py --run_k_client_in_parallel --baseline --scheduler_type None --database postgres --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --seed 13
cp saved_results/clients_1_timeout_120_baseline.csv saved_results/exp_k1_seed13_baseline.csv

# Seed 13 ours
python run.py --run_k_client_in_parallel --database postgres --model_name postgres_brad --target_path models/_checkpoints --rnn_type bilstm --save_result_dir saved_results --host 127.0.0.1 --port 5432 --user <db-user> --password '' --db_name imdb --query_bank_path workloads/postgres/brad_queries_normalized.sql --num_clients 1 --timeout_s 120 --exec_for_s 180 --ignore_short_running --scheduler_type greedy --seed 13 --debug
cp saved_results/clients_1_timeout_120_ours.csv saved_results/exp_k1_seed13_ours.csv
```

After all runs:

```bash
ls -1 saved_results/exp_k1_seed*.csv | sort
```

#### Step 9D: Result comparison (baseline vs ours)

Use this script to summarize one experiment group (example: `K=1`):

```bash
python3 - <<'PY'
import csv,glob,os,statistics as st
base='saved_results'
prefix='exp_k1_seed'   # change to exp_k4_seed for K=4
files=sorted(glob.glob(os.path.join(base,f'{prefix}*_*.csv')))
if not files:
    print('No files found for prefix:', prefix)
    raise SystemExit(1)

rows=[]
for p in files:
    name=os.path.basename(p).replace('.csv','')  # exp_k1_seed11_baseline
    parts=name.split('_')
    seed=parts[2].replace('seed','')
    mode=parts[3]
    data=list(csv.DictReader(open(p)))
    rt=[float(r['run_time_s']) for r in data if r.get('run_time_s')]
    rows.append({
        'seed':seed, 'mode':mode, 'n':len(data),
        'mean':st.mean(rt) if rt else float('nan'),
        'p50':st.median(rt) if rt else float('nan'),
        'p90':sorted(rt)[int(0.9*(len(rt)-1))] if rt else float('nan'),
        'timeouts':sum(r.get('timeout','').lower()=='true' for r in data),
        'errors':sum(r.get('error','').lower()=='true' for r in data),
    })

print('seed,mode,n,mean_run_s,p50_s,p90_s,timeouts,errors')
for r in sorted(rows, key=lambda x:(x['seed'],x['mode'])):
    print(f"{r['seed']},{r['mode']},{r['n']},{r['mean']:.4f},{r['p50']:.4f},{r['p90']:.4f},{r['timeouts']},{r['errors']}")

seeds=sorted(set(r['seed'] for r in rows))
print('\\npaired_mean_delta_s (ours - baseline):')
for s in seeds:
    b=[r for r in rows if r['seed']==s and r['mode']=='baseline']
    o=[r for r in rows if r['seed']==s and r['mode']=='ours']
    if b and o:
        print(f"seed {s}: {o[0]['mean']-b[0]['mean']:+.4f}s")

b_means=[r['mean'] for r in rows if r['mode']=='baseline']
o_means=[r['mean'] for r in rows if r['mode']=='ours']
if b_means and o_means:
    print(f"\naggregate mean of seed means: baseline={st.mean(b_means):.4f}s, ours={st.mean(o_means):.4f}s, delta={st.mean(o_means)-st.mean(b_means):+.4f}s")
PY
```

How to interpret:

- Lower `mean_run_s` is better.
- Check `timeouts` and `errors` first (must be low/zero).
- Prefer multiple seeds; do not conclude from a single seed.
- If row counts differ between modes, treat conclusions as directional and run longer windows for stronger confidence.

---

## 5) Expected Outputs

### Common generated files

- Warmup:
  - `saved_results/_timeout_1000_warmup_run.csv`
- Baseline trace:
  - `saved_results/clients_4_timeout_120_baseline.csv`
- Trained models:
  - `models/_checkpoints/postgres_brad_stage_model.pkl`
  - `models/_checkpoints/postgres_brad_bilstm_256_2_q_loss_wo_sep`
- Experiment outputs:
  - `saved_results/clients_<K>_timeout_120_baseline.csv`
  - `saved_results/clients_<K>_timeout_120_ours.csv`
  - Optional copied snapshots:
    - `saved_results/exp_k<K>_seed<SEED>_baseline.csv`
    - `saved_results/exp_k<K>_seed<SEED>_ours.csv`
- Debug logs (ours):
  - `saved_results/verbose_logs/ours_<id>.log`

### What success looks like

- Commands return to prompt without traceback.
- Output CSV exists and has >1 line.
- `timeout` and `error` columns mostly/fully false.

### Why warmup file can show 239 lines

If `saved_results/_timeout_1000_warmup_run.csv` shows `239` lines, that is expected:

- `1` line is CSV header.
- `238` lines are warmup query results.

Those `238` query rows come from the normalized BRAD query bank (`brad_queries_normalized.sql`), which contains 238 SQL statements.
So `239` total lines means warmup completed correctly for that query bank.

---

## 6) What We Actually Ran vs Not Run

This is a factual checklist from this project session.

### Strategies used in this session

- Baseline runs used:
  - `--baseline`
  - `--scheduler_type None`
- Ours runs used:
  - `--scheduler_type greedy`
  - trained model: `postgres_brad` from `models/_checkpoints`
- Strategies **not used** in final experiments:
  - `lp`
  - `qshuffler`
  - `pgm`

### What we ran successfully

1. Local Postgres connectivity checks.
2. IMDB JOB download + extraction.
3. IMDB schema creation.
4. IMDB data load into Postgres (using `utils/load_imdb_csvs_python.py`).
5. Warmup execution (`--warmup_run`) with BRAD normalized query file.
6. Baseline close-loop runs (`--run_k_client_in_parallel --baseline`).
7. Model training (`--train_concurrent_rnn`) and checkpoint generation.
8. Ours close-loop runs (`--run_k_client_in_parallel` with trained model).
9. Multi-seed comparisons for:
  - `K=1` (seeds 11/12/13)
  - `K=4` (seeds 11/12/13)

### What we did not rely on for final results

1. We did not rely on `--replay_workload` for final comparisons because this path repeatedly showed long silent waits/stalls in this local setup.
2. We did not rely on direct raw `psql \\copy` for final data load because malformed rows in the downloaded CSVs caused parser failures.
3. We did not rely on Redshift flow; this runbook validates the Postgres path.

---

## 7) Troubleshooting Guide

### A) NumPy/PyTorch ABI error on startup

Symptoms:

- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
- import failures around `torch` / NumPy initialization.

Fix:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
python --version  # should be 3.11.x
```

Ensure `numpy<2` is installed in the active venv.

### B) `FileNotFoundError: ... brad_queries_normalized.sql`

- Regenerate normalized file (Step 5), or use `brad_queries.sql` if parser path permits.

### C) `Cannot save file into a non-existent directory: 'saved_results'`

- Create directory:

```bash
mkdir -p saved_results
```

### D) `KeyError` during training (query id)

- Ensure `complex_models.py` fix is present (full prediction map use).

### E) CSV import fails (`extra data after last expected column`, quote errors)

- Use `utils/load_imdb_csvs_python.py` instead of direct `\copy`.

### F) zsh parse errors (`<K>`, `<SEED>`)

- Do not use placeholders literally. Replace with real values.

### G) Looks “stuck”

- Check if output files are growing:

```bash
ls -lt saved_results | head
```

- For `--warmup_run`, the process is often **silent** and can take a long time.
  - It does not print per-query progress to terminal.
  - It writes CSV progress periodically (every ~40 queries in this code path).
  - With `--timeout_s 1000` and ~238 queries, worst-case wall time can be very long.
  - Use a smaller timeout (for example `--timeout_s 120`) for faster local validation.
- To confirm warmup progress specifically:

```bash
wc -l saved_results/_timeout_1000_warmup_run.csv
```

If line count increases over time, warmup is progressing.

- Check DB activity:

```bash
psql "host=127.0.0.1 port=5432 dbname=imdb user=nizarnoorani" -c "select now(),state,wait_event_type,wait_event,left(query,120) from pg_stat_activity where datname='imdb' and state<>'idle';"
```

- Use single-thread env vars for stability:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

### H) Directory vs file argument confusion in training

- `--directory` is expected as directory in current code path. Use folder, not direct file path.

---

## 8) Important Notes / Gotchas

1. This is **research code**, not a polished production framework.
2. Some scripts are helper-style and assume manual/operator judgment.
3. Data import quality may vary by source dump; malformed rows can exist.
4. Benchmarks are sensitive to environment (CPU threads, DB cache state, OS, trace timing).
5. Short experiments can have varying row counts between modes; use multiple seeds and averages.

---

## 9) Final Clean Repo State

### Keep (required for your stable workflow)

- `models/concurrency/complex_models.py`
- `utils/load_database.py`
- `workloads/postgres/imdb_schema.py`
- `utils/load_imdb_csvs_python.py`
- `requirements.txt`

### Optional/runtime-generated (can regenerate anytime)

- `saved_results/`*
- `models/_checkpoints/postgres_brad_`*
- `workloads/postgres/brad_queries_normalized.sql`

### Ignore for this IMDB path

- `tpch-dbgen` (unless you plan TPC-H experiments)

---

## Quick Executive Summary

- The original repository failed locally because setup/import/training assumptions did not hold in your environment.
- We fixed one real training bug, hardened DB/data loading, fixed schema typo, and added a robust CSV loader.
- You successfully completed baseline + ours experiment sets and generated comparable result files.
- With these changes and runbook steps, a new team member can reproduce your workflow end-to-end.

