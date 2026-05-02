# run.py
import argparse
import os.path
import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # Windows + psycopg3; uncomment if needed.
import shutil
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
import copy
import pickle as pkl
from utils.load_trace import create_concurrency_dataset, load_all_csv_from_dir
from models.single.stage import SingleStage
from models.concurrency.complex_models import ConcurrentRNN
from scheduler.greedy_scheduler import GreedyScheduler
from executor.executor import Executor
from utils.logging import create_custom_logger

np.set_printoptions(suppress=True)


def load_workload(
    train_test_split: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    all_trace = load_all_csv_from_dir(args.directory)
    all_concurrency_df = []
    for trace in all_trace:
        concurrency_df = create_concurrency_dataset(
            trace, engine=None, pre_exec_interval=None
        )
        all_concurrency_df.append(concurrency_df)
    concurrency_df = pd.concat(all_concurrency_df, ignore_index=True)

    if train_test_split:
        np.random.seed(0)
        train_idx = np.random.choice(
            len(concurrency_df), size=int(0.8 * len(concurrency_df)), replace=False
        )
        test_idx = [i for i in range(len(concurrency_df)) if i not in train_idx]
        train_trace_df = copy.deepcopy(concurrency_df.iloc[train_idx])
        eval_trace_df = concurrency_df.iloc[test_idx]
        eval_trace_df = copy.deepcopy(
            eval_trace_df[eval_trace_df["num_concurrent_queries"] > 0]
        )
        print(len(train_trace_df), len(eval_trace_df))
        return train_trace_df, eval_trace_df
    return concurrency_df


def train_concurrent_rnn() -> None:
    train_trace_df, eval_trace_df = load_workload(train_test_split=True)
    print("Starting stage model featurization and training...")
    ss = SingleStage(
        use_size=args.true_card,
        use_log=args.use_log,
        true_card=args.true_card,
        use_table_features=args.use_table_features,
        use_table_selectivity=args.use_table_selectivity,
        num_operators=args.num_operators,
    )
    df = ss.featurize_data(train_trace_df, args.parsed_queries_path)
    ss.train(df)
    print("Stage model training complete.")
    with open(
        os.path.join(args.target_path, f"{args.model_name}_stage_model.pkl"), "wb"
    ) as f:
        pkl.dump(ss, f)
    print(
        f"Saved stage model to {os.path.join(args.target_path, f'{args.model_name}_stage_model.pkl')}"
    )

    print("Starting concurrent RNN training...")
    rnn = ConcurrentRNN(
        ss,
        model_prefix=args.model_name,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_separation=args.use_separation,
        ignore_short_running=args.ignore_short_running,
        short_running_threshold=args.short_running_threshold,
    )
    rnn.train(
        train_trace_df,
        eval_trace_df,
        lr=args.lr,
        loss_function=args.loss_function,
        val_on_test=args.val_on_test,
        epochs=args.epochs,
    )
    print("Concurrent RNN training complete.")
    if args.target_path is not None:
        rnn.save_model(args.target_path)
        print(f"Saved concurrent RNN model under {args.target_path}")


def load_concurrent_rnn_stage_model() -> Tuple[SingleStage, ConcurrentRNN]:
    with open(
        os.path.join(args.target_path, f"{args.model_name}_stage_model.pkl"), "rb"
    ) as f:
        ss = pkl.load(f)

    model = ConcurrentRNN(
        ss,
        model_prefix=args.model_name,
        input_size=len(ss.all_feature[0]) * 2 + 7,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        use_separation=args.use_separation,
        ignore_short_running=args.ignore_short_running,
        short_running_threshold=args.short_running_threshold,
    )
    model.load_model(args.target_path)
    return ss, model


def warmup_run(query_bank_path: str) -> None:
    database_kwargs = {
        "host": args.host,
        "dbname": args.db_name,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }
    executor = Executor(
        database_kwargs, timeout=args.timeout_s, database=args.database, scheduler=None
    )
    executor.warmup_run(
        query_bank_path, args.save_result_dir, args.selected_query_idx_path
    )


def run_k_client_in_parallel(
    query_bank_path: str,
    num_clients: int,
    save_result_dir: str,
    selected_query_idx_path: Optional[str] = None,
) -> None:
    if args.debug:
        verbose_log_dir = os.path.join(save_result_dir, "verbose_logs")
        if not os.path.exists(verbose_log_dir):
            os.mkdir(verbose_log_dir)
        if args.baseline:
            log_name = "baseline"
        else:
            log_name = "ours"
        run_id = np.random.randint(100000)
        log_file_path = os.path.join(verbose_log_dir, f"{log_name}_{run_id}.log")
        verbose_logger = create_custom_logger(log_name, log_file_path)
        print(f"Debug log save to: {log_file_path}")
    else:
        verbose_logger = None
    if args.scheduler_type == "greedy":
        ss, rnn = load_concurrent_rnn_stage_model()
        scheduler = GreedyScheduler(
            ss,
            rnn,
            debug=args.debug,
            logger=verbose_logger,
            ignore_short_running=args.ignore_short_running,
            starve_penalty=args.starve_penalty,
            alpha=args.alpha,
            short_running_threshold=args.short_running_threshold,
            steps_into_future=args.steps_into_future,
        )
    else:
        scheduler = None
        assert args.baseline, (
            f"{args.scheduler_type} scheduler not implemented and not in a baseline run"
        )

    database_kwargs = {
        "host": args.host,
        "dbname": args.db_name,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }
    executor = Executor(
        database_kwargs,
        timeout=args.timeout_s,
        database=args.database,
        scheduler=scheduler,
        pause_wait_s=0.1,
        debug=args.debug,
        logger=verbose_logger,
    )
    asyncio.run(
        executor.run_k_client_in_parallel(
            query_bank_path,
            num_clients,
            args.baseline,
            save_result_dir,
            selected_query_idx_path=selected_query_idx_path,
            exec_for_s=args.exec_for_s,
            seed=args.seed,
        )
    )
    mode = "baseline" if args.baseline else "ours"
    src = os.path.join(
        save_result_dir, f"clients_{num_clients}_timeout_{args.timeout_s}_{mode}.csv"
    )
    dst = os.path.join(
        save_result_dir, f"exp_k{num_clients}_seed{args.seed}_{mode}.csv"
    )
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Saved seed snapshot: {dst}")
    else:
        print(
            f"Warning: expected result file not found, skip snapshot copy: {src}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_concurrent_rnn", action="store_true")
    parser.add_argument("--warmup_run", action="store_true")
    parser.add_argument("--run_k_client_in_parallel", action="store_true")

    parser.add_argument("--target_path", type=str)

    parser.add_argument("--use_size", action="store_true")
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--true_card", action="store_true")
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--num_operators", type=int, default=20)
    parser.add_argument("--use_separation", action="store_true")
    parser.add_argument("--use_table_features", action="store_true")
    parser.add_argument("--use_table_selectivity", action="store_true")

    parser.add_argument("--num_clients", default=8, type=int)
    parser.add_argument("--parsed_queries_path", type=str)
    parser.add_argument("--directory", type=str)

    parser.add_argument("--model_name", default="postgres", type=str)
    parser.add_argument("--embedding_dim", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--loss_function", default="q_loss", type=str)
    parser.add_argument("--val_on_test", action="store_true")

    parser.add_argument("--scheduler_type", default="greedy", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ignore_short_running", action="store_true")
    parser.add_argument("--steps_into_future", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--short_running_threshold", type=float, default=5.0)
    parser.add_argument("--starve_penalty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--exec_for_s", type=int, default=24 * 3600)
    parser.add_argument("--num_clients_list", type=str, default=None)
    parser.add_argument("--selected_query_idx_path", type=str)
    parser.add_argument("--database", default="postgres", type=str)
    parser.add_argument("--save_result_dir", type=str)
    parser.add_argument("--query_bank_path", type=str)
    parser.add_argument("--timeout_s", default=200, type=int)
    parser.add_argument("--host", type=str)
    parser.add_argument("--db_name", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)

    args = parser.parse_args()

    if args.train_concurrent_rnn:
        train_concurrent_rnn()

    if args.warmup_run:
        warmup_run(args.query_bank_path)

    if args.run_k_client_in_parallel:
        if args.num_clients_list is not None:
            num_clients_list = list(map(int, args.num_clients_list.split(",")))
            for num_clients in num_clients_list:
                print(num_clients)
                run_k_client_in_parallel(
                    args.query_bank_path,
                    num_clients,
                    args.save_result_dir,
                    args.selected_query_idx_path,
                )
        else:
            run_k_client_in_parallel(
                args.query_bank_path,
                args.num_clients,
                args.save_result_dir,
                args.selected_query_idx_path,
            )
