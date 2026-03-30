#!/usr/bin/env python3
"""
Load IMDB JOB CSVs into Postgres when psql \\copy fails (malformed quotes, newlines in fields).

1. Parse rows with csv.reader (handles most multiline quoted fields).
2. Skip rows whose column count != header (unrecoverable garbage in the dump).
3. Stream COPY using FORMAT text + tab delimiter + backslash escapes — avoids Postgres CSV quirks.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import psycopg2
from psycopg2 import sql

from workloads.postgres.imdb_schema import IMDB_TABLE_NAMES

# Keep batches under ~64MB string buffer to limit memory.
_BATCH_BYTES = 48 * 1024 * 1024


def _truncate_all(cur, tables: list[str]) -> None:
    if not tables:
        return
    parts = sql.SQL(", ").join(sql.Identifier(t) for t in tables)
    cur.execute(sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY").format(parts))


def _get_table_columns(cur, table_name: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    cols = [r[0] for r in cur.fetchall()]
    if not cols:
        raise ValueError(f"Table '{table_name}' has no columns in schema public.")
    return cols


def _escape_text_copy_field(val: str | None) -> str:
    """PostgreSQL COPY TEXT: \\t \\n \\r \\ must be escaped; empty -> NULL."""
    if val is None or val == "":
        return "\\N"
    s = str(val)
    s = s.replace("\\", "\\\\")
    s = s.replace("\t", "\\t")
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    return s


def _row_to_text_copy_line(cols: list[str]) -> str:
    return "\t".join(_escape_text_copy_field(c) for c in cols) + "\n"


def _load_one_table(
    cur, conn, table_name: str, csv_path: str, encoding: str
) -> tuple[int, int]:
    rows_loaded = 0
    rows_skipped = 0
    columns = _get_table_columns(cur, table_name)
    ncols = len(columns)
    col_list = sql.SQL(", ").join(sql.Identifier(h) for h in columns)
    copy_stmt = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')"
    ).format(sql.Identifier(table_name), col_list)
    copy_sql_str = copy_stmt.as_string(conn)

    with open(csv_path, "r", encoding=encoding, errors="replace", newline="") as inf:
        reader = csv.reader(inf)

        buf = io.StringIO()

        def flush() -> None:
            nonlocal buf
            buf.seek(0)
            payload = buf.getvalue()
            if not payload.strip():
                buf = io.StringIO()
                return
            stream = io.StringIO(payload)
            cur.copy_expert(copy_sql_str, stream)
            buf = io.StringIO()

        first_row_checked = False
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                rows_skipped += 1
                if rows_skipped <= 5:
                    print("  skip row (csv parse error)", file=sys.stderr)
                continue
            # Some files include a header, some do not. Skip it if present.
            if not first_row_checked:
                first_row_checked = True
                row_norm = [x.strip().lower() for x in row]
                if row_norm == [c.lower() for c in columns]:
                    continue
            if len(row) != ncols:
                rows_skipped += 1
                if rows_skipped <= 5:
                    print(
                        f"  skip row (expected {ncols} cols, got {len(row)})",
                        file=sys.stderr,
                    )
                continue
            buf.write(_row_to_text_copy_line(row))
            rows_loaded += 1
            if buf.tell() >= _BATCH_BYTES:
                flush()
                conn.commit()

        flush()
        conn.commit()
    return rows_loaded, rows_skipped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Load IMDB JOB CSVs via Python csv + COPY STDIN.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="5432")
    p.add_argument("--user", default=os.environ.get("USER", "postgres"))
    p.add_argument("--password", default=os.environ.get("PGPASSWORD", ""))
    p.add_argument("--database", default="imdb")
    p.add_argument("--data-dir", required=True, help="Directory with aka_name.csv, ...")
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="Try utf-8 first; script does not auto-fallback per file (use latin-1 if needed).",
    )
    p.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not TRUNCATE tables first (you must have empty tables or accept duplicates).",
    )
    args = p.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    missing = [
        t
        for t in IMDB_TABLE_NAMES
        if not os.path.isfile(os.path.join(data_dir, f"{t}.csv"))
    ]
    if missing:
        print("Missing CSV files:", ", ".join(missing), file=sys.stderr)
        return 1

    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.user,
            password=args.password,
        )
        conn.autocommit = False
        cur = conn.cursor()
        if not args.no_truncate:
            print("Truncating IMDB tables...", file=sys.stderr)
            _truncate_all(cur, IMDB_TABLE_NAMES)
            conn.commit()

        for t in IMDB_TABLE_NAMES:
            path = os.path.join(data_dir, f"{t}.csv")
            print(f"Loading {t} ...", file=sys.stderr, flush=True)
            try:
                n, skipped = _load_one_table(cur, conn, t, path, args.encoding)
                extra = f", skipped {skipped} bad-width rows" if skipped else ""
                print(f"  done ({n} rows{extra})", file=sys.stderr, flush=True)
            except Exception as e:
                conn.rollback()
                print(f"Error loading {t}: {e}", file=sys.stderr)
                return 1
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print("All tables loaded.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
