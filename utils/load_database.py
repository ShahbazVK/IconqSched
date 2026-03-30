import argparse
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import psycopg2
from psycopg2 import sql

from workloads.postgres.imdb_schema import IMDB_SCHEMA, IMDB_FK_INDEX, IMDB_LOAD_TEMPLATE, IMDB_TABLE_NAMES
from workloads.postgres.tpc_schema import TPC_SCHEMA, TPC_FK_INDEX, TPC_LOAD_TEMPLATE, TPC_TABLE_NAMES
from workloads.redshift.imdb_schema import REDSHIFT_IMDB_SCHEMA, REDSHIFT_IMDB_LOAD_TEMPLATE, REDSHIFT_IMDB_TABLE_NAMES
from workloads.redshift.tpc_schema import REDSHIFT_TPC_SCHEMA, REDSHIFT_TPC_TABLE_NAMES, REDSHIFT_TPC_LOAD_TEMPLATE


def _postgres_workload_parts(db_name: str):
    if db_name == "imdb":
        return IMDB_SCHEMA, IMDB_LOAD_TEMPLATE, IMDB_TABLE_NAMES, IMDB_FK_INDEX
    if db_name == "tpc":
        return TPC_SCHEMA, TPC_LOAD_TEMPLATE, TPC_TABLE_NAMES, TPC_FK_INDEX
    raise ValueError(f"unrecognized db_name {db_name}")


def load_database_postgres(
    data_dir: str,
    db_name: str = "imdb",
    host: str = "imdb-postgres.xxx.us-east-1.rds.amazonaws.com",
    port: str = "5432",
    user: str = "postgres",
    password: str = "xxxx",
    drop_database: bool = True,
    reset_public_schema: bool = False,
    create_schema: bool = True,
    print_copy_commands: bool = True,
    print_indexes: bool = True,
) -> None:
    admin_conn_kwargs = dict(
        host=host, port=port, database="postgres", user=user, password=password
    )
    if drop_database:
        conn = psycopg2.connect(**admin_conn_kwargs)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
        cur.execute(f"CREATE DATABASE {db_name};")
        cur.close()
        conn.close()

    schema, load_template, table_names, fk_index = _postgres_workload_parts(db_name)

    conn = psycopg2.connect(
        host=host, port=port, database=db_name, user=user, password=password
    )
    conn.autocommit = True
    cur = conn.cursor()
    if reset_public_schema:
        cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
        cur.execute("CREATE SCHEMA public")
        cur.execute(
            sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(sql.Identifier(user))
        )
        cur.execute("GRANT ALL ON SCHEMA public TO PUBLIC")

    # Emit load SQL before CREATE TABLE so redirects (e.g. > imdb_load.sql) still get
    # \copy lines when tables already exist and create_schema would error.
    if print_copy_commands:
        print("-- Run these inside psql connected to your database (\\copy runs on the client, reads local files):")
        for table_name in table_names:
            load_query = load_template.format(
                table_name=table_name,
                path=os.path.join(data_dir, f"{table_name}.csv"),
            )
            print(load_query)
    if print_indexes:
        print(fk_index)
    sys.stdout.flush()

    if create_schema:
        cur.execute(schema)
    conn.commit()
    cur.close()
    conn.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create IMDB/TPC schema on Postgres and print \\copy commands for local CSV files."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="5432")
    p.add_argument("--user", default=os.environ.get("USER", "postgres"))
    p.add_argument(
        "--password",
        default=os.environ.get("PGPASSWORD", ""),
        help="Empty string is OK for local trust auth (default: env PGPASSWORD or '').",
    )
    p.add_argument("--database", default="imdb", choices=("imdb", "tpc"))
    p.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing <table_name>.csv files. Required unless --schema-only.",
    )
    p.add_argument(
        "--schema-only",
        action="store_true",
        help="Only create tables (no \\copy lines). Use when you do not have CSVs yet.",
    )
    p.add_argument(
        "--no-drop",
        action="store_true",
        help="Do not DROP/CREATE the database; only connect and create tables (use for existing empty DB).",
    )
    p.add_argument(
        "--reset-public-schema",
        action="store_true",
        help="Before CREATE TABLE: DROP SCHEMA public CASCADE (wipes all tables in this DB). Implies --no-drop.",
    )
    p.add_argument(
        "--skip-schema",
        action="store_true",
        help="Only print \\copy and index SQL; do not run CREATE TABLE.",
    )
    p.add_argument(
        "--skip-index-print",
        action="store_true",
        help="Do not print secondary index DDL after \\copy lines.",
    )
    return p


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _build_arg_parser().parse_args(argv)
    if not args.schema_only and args.data_dir is None:
        print("Error: pass --data-dir or use --schema-only.", file=sys.stderr)
        return 1
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ""
    no_drop = args.no_drop or args.reset_public_schema
    try:
        load_database_postgres(
            data_dir=data_dir,
            db_name=args.database,
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            drop_database=not no_drop,
            reset_public_schema=args.reset_public_schema,
            create_schema=not args.skip_schema,
            print_copy_commands=bool(args.data_dir) and not args.schema_only,
            print_indexes=not args.skip_index_print and not args.schema_only,
        )
    except Exception as e:
        err = str(e).lower()
        print(f"Error: {e}", file=sys.stderr)
        if "already exists" in err:
            print(
                "If you meant to recreate from scratch: add --reset-public-schema (wipes all tables in this DB).",
                file=sys.stderr,
            )
            print(
                'Or list tables: psql ... -c "\\dt public.*"',
                file=sys.stderr,
            )
            print(
                "If you redirected stdout to a .sql file, it should still contain the \\copy lines — run that file with psql.",
                file=sys.stderr,
            )
        return 1
    if args.schema_only:
        print(
            "\nSchema created. Download IMDB CSVs (e.g. Join Order Benchmark), then run:\n"
            "  python utils/load_database.py --no-drop --user YOUR_USER --data-dir /path/to/csv\n"
            "and execute the printed \\copy lines in psql, then the index DDL.",
            file=sys.stderr,
        )
    else:
        print(
            "\nNext: run the printed \\copy commands in psql, then the index DDL after loads.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def load_database_redshift(s3_path: str, db_name: str = "imdb"):
    # s3_path of format s3://{s3_bucket}/{s3_obj}
    host = "redshift-tpc-h.xxx.us-east-1.rds.amazonaws.com"
    port = "5439"
    user = "awsuser"
    token = "xxxx"
    iam_role = 'arn:aws:iam::xxx:role/RedshiftS3'
    conn = psycopg2.connect(host=host, port=port, database="postgres", user=user, password=token)
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
    cur.execute(f"CREATE DATABASE {db_name};")
    cur.close()
    conn.close()

    if db_name == "imdb":
        schema = REDSHIFT_IMDB_SCHEMA
        if type(schema) == list:
            schema = '\n'.join(schema)
        load_template = REDSHIFT_IMDB_LOAD_TEMPLATE
        table_names = REDSHIFT_IMDB_TABLE_NAMES
    elif db_name == "tpc":
        schema = REDSHIFT_TPC_SCHEMA
        load_template = REDSHIFT_TPC_LOAD_TEMPLATE
        table_names = REDSHIFT_TPC_TABLE_NAMES
    else:
        assert False, f"unrecognized db_name {db_name}"

    conn = psycopg2.connect(host=host, port=port, database=db_name, user=user, password=token)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(schema)

    for table_name in table_names:
        load_query = load_template.format(
            table_name=table_name,
            s3_path=os.path.join(s3_path, table_name, f"{table_name}.csv"),
            s3_iam_role=iam_role
        )
        cur.execute(load_query)
    conn.commit()
    cur.close()
    conn.close()






