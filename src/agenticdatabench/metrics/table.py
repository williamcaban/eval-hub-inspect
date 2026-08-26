"""
Table Metrics Module - CSV and SQLite comparison metrics for evaluation.

This module compares tabular data:
- CSV file comparison with column matching and tolerance support
- SQLite database comparison by converting to CSVs
- Handles multiple potential gold answers
- Supports specified columns and ignore_order options

Ported from DA-Code (MIT) — https://github.com/yiyihum/da-code
"""

from __future__ import annotations

import logging
import math
import os.path
import random
import sqlite3
from typing import Any

import pandas as pd


def compare_csv(
    output_file_name: str, gold_file_name: str | list[str], **options
) -> dict[str, Any]:
    """
    @args:
        output_file_name(str): the pred csv file
        gold_file_name(str|list[str]): the gold csv file or csv files, maybe multiple potential
            answers, not there are two answers
        option(dict): the configuration dictionary
            - specified_columns(list|list[list]): the column name that should be used to compare
            - ignore_order(bool|list(bool)): whether to ignore the order of the rows
            - thresholds(dict): column index to tolerance mapping, e.g., {0: None, 1: 0.02}
                               None means default tolerance (0.01), otherwise use specified value
    @return:
        dict: {'score': float, 'errors': list[str]}
            - score: the similarity score (0.0 to 1.0)
            - errors: detailed error messages including missing columns
    """
    if set(options.keys()) > {"specified_columns", "ignore_order", "thresholds"}:
        raise ValueError(f"Unexpected options: {options.keys()}")
    if isinstance(gold_file_name, list):
        specified_columns = options.get("specified_columns", [[]] * len(gold_file_name))
        ignore_order = options.get("ignore_order", [True] * len(gold_file_name))
    elif isinstance(gold_file_name, str):
        specified_columns = [options.get("specified_columns", [])]
        ignore_order = [options.get("ignore_order", True)]
        gold_file_name = [gold_file_name]
    thresholds = options.get("thresholds", {})
    default_abs_tol = 1e-2  # default absolute tolerance

    def resolve_threshold_keys(thresholds_dict: dict, gold_df: pd.DataFrame) -> dict:
        """Convert threshold keys from column names to column indices."""
        resolved: dict[int, Any] = {}
        for key, value in thresholds_dict.items():
            if isinstance(key, int):
                resolved[key] = value
            elif isinstance(key, str):
                if key in gold_df.columns:
                    resolved[gold_df.columns.get_loc(key)] = value
                else:
                    logging.warning(
                        f"Threshold column '{key}' not found in gold file, "
                        f"available columns: {list(gold_df.columns)}"
                    )
            else:
                logging.warning(f"Invalid threshold key type: {type(key)}, expected int or str")
        return resolved

    def get_tolerance_info(col_idx: int, thresholds_dict: dict) -> tuple[float, bool]:
        """Get tolerance info for a specific column index."""
        if col_idx in thresholds_dict:
            tol = thresholds_dict[col_idx]
            if tol is None:
                return (0.01, True)
            return (tol, True)
        return (default_abs_tol, False)

    def normalize_value(val: Any, tol: float, is_relative: bool = False) -> Any:
        """Normalize value for hashing - round floats for tolerance, lowercase strings."""
        if pd.isna(val):
            return "__NA__"
        elif isinstance(val, (int, float)):
            if is_relative:
                if val == 0 or tol <= 0:
                    return ("rel", 0.0 if val == 0 else float(val))
                sign = 1 if val > 0 else -1
                return ("rel", sign, round(math.log(abs(val)) / math.log1p(tol)))
            actual_tol = tol
            return round(val / actual_tol) * actual_tol
        elif isinstance(val, str):
            return val.lower().strip()
        return val

    def vector_to_hashable(
        v: list, tol: float, is_relative: bool = False, do_sort: bool = False
    ) -> tuple:
        """Convert vector to hashable tuple for fast comparison."""
        normalized = [normalize_value(x, tol, is_relative) for x in v]
        if do_sort:
            normalized = sorted(normalized, key=lambda x: (x == "__NA__", str(x)))
        return tuple(normalized)

    def vectors_match(
        v1: list,
        v2: list,
        col_idx: int,
        ignore_order_: bool = False,
        thresholds_dict: dict | None = None,
    ) -> bool:
        if thresholds_dict is None:
            thresholds_dict = {}
        tol, is_relative = get_tolerance_info(col_idx, thresholds_dict)
        if ignore_order_:
            set_v1 = {normalize_value(x, tol, is_relative) for x in v1}
            set_v2 = {normalize_value(x, tol, is_relative) for x in v2}
            return set_v1.issubset(set_v2)
        else:
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2, strict=False):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if is_relative:
                        if abs(a) < 1e-12 and abs(b) < 1e-12:
                            continue
                        if abs(a) < 1e-12 or abs(b) < 1e-12:
                            return False
                        rel_error = abs(a - b) / max(abs(a), abs(b))
                        if rel_error > tol:
                            return False
                    else:
                        if not math.isclose(float(a), float(b), abs_tol=tol):
                            return False
                elif isinstance(a, str) and isinstance(b, str):
                    if a.lower().strip() != b.lower().strip():
                        return False
                elif a != b:
                    return False
            return True

    def csv_score(
        pred: pd.DataFrame,
        gold: pd.DataFrame,
        specified_columns_: list | None = None,
        score_rule_: str = "divide",
        ignore_order_: bool = False,
        total_scores_: int = 1,
        thresholds_: dict | None = None,
    ) -> tuple[float, list[str], str]:
        """Compare CSV files and return detailed score, errors, and meaning."""
        if specified_columns_ is None:
            specified_columns_ = []
        if thresholds_ is None:
            thresholds_ = {}
        # Build mapping from filtered column index to original column index
        if specified_columns_:
            gold_cols = gold.loc[:, specified_columns_]
            original_col_indices = [gold.columns.get_loc(col) for col in specified_columns_]
        else:
            gold_cols = gold
            original_col_indices = list(range(len(gold.columns)))
        pred_cols = pred

        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred_cols.transpose().values.tolist()

        max_elements = 10000
        if t_gold_list and len(t_gold_list[0]) > max_elements:
            t_gold_list = [col[:max_elements] for col in t_gold_list]
        if not ignore_order_ and t_pred_list and len(t_pred_list[0]) > max_elements:
            t_pred_list = [col[:max_elements] for col in t_pred_list]

        # Pre-compute hashes for pred columns for O(1) lookup
        pred_hashes: dict[tuple, int] = {}
        for j, pred_col in enumerate(t_pred_list):
            tol, is_rel = get_tolerance_info(j, thresholds_)
            h = vector_to_hashable(pred_col, tol, is_relative=is_rel, do_sort=ignore_order_)
            if h not in pred_hashes:
                pred_hashes[h] = j

        errors: list[str] = []
        pre_score: float
        if score_rule_ == "all":
            pre_score = total_scores_
            for i, gold_col in enumerate(t_gold_list):
                orig_idx = original_col_indices[i]
                tol, is_rel = get_tolerance_info(orig_idx, thresholds_)
                gold_hash = vector_to_hashable(
                    gold_col, tol, is_relative=is_rel, do_sort=ignore_order_
                )
                if gold_hash in pred_hashes:
                    continue
                found = False
                for pred_col in t_pred_list:
                    if vectors_match(
                        gold_col,
                        pred_col,
                        orig_idx,
                        ignore_order_=ignore_order_,
                        thresholds_dict=thresholds_,
                    ):
                        found = True
                        break
                if not found:
                    pre_score = 0
                    col_name = specified_columns_[i] if specified_columns_ else f"Column index {i}"
                    pred_normalized_sets = []
                    for pred_col in t_pred_list:
                        normalized = [normalize_value(x, tol, is_relative=is_rel) for x in pred_col]
                        pred_normalized_sets.append(set(normalized))

                    shuffled_gold = list(gold_col)
                    random.shuffle(shuffled_gold)
                    unmatched = []
                    max_count = 50
                    max_str_len = 5000
                    for val in shuffled_gold:
                        val_norm = normalize_value(val, tol, is_relative=is_rel)
                        found_val = any(val_norm in ps for ps in pred_normalized_sets)
                        if not found_val:
                            unmatched.append(val)
                            if len(unmatched) >= max_count:
                                break
                            sample_str = ", ".join([str(v) for v in unmatched])
                            if len(sample_str) >= max_str_len:
                                break

                    if unmatched:
                        if len(unmatched) == len(gold_col) and all(
                            isinstance(x, str) for x in gold_col
                        ):
                            logging.warning(
                                f"Column '{col_name}': match score 0.0/1.0, all values unmatched"
                            )
                        full_str = ", ".join([str(v) for v in unmatched])
                        sample_str = (
                            full_str[:max_str_len] + "..."
                            if len(full_str) > max_str_len
                            else full_str
                        )
                        error_msg = (
                            f"Column '{col_name}' does not match: expected values {sample_str} "
                            f"but none found in prediction "
                            f"(score_rule='all' requires all columns to match)"
                        )
                    else:
                        error_msg = (
                            f"Column '{col_name}' does not match: values are completely different "
                            f"from expected (score_rule='all' requires all columns to match)"
                        )
                    errors.append(error_msg)
        elif score_rule_ == "divide":
            matches = 0
            total = len(t_gold_list) if t_gold_list else 1

            for i, gold_col in enumerate(t_gold_list):
                orig_idx = original_col_indices[i]
                tol, is_rel = get_tolerance_info(orig_idx, thresholds_)
                gold_hash = vector_to_hashable(
                    gold_col, tol, is_relative=is_rel, do_sort=ignore_order_
                )
                if gold_hash in pred_hashes:
                    matches += total_scores_
                    continue
                found = False
                for pred_col in t_pred_list:
                    if vectors_match(
                        gold_col,
                        pred_col,
                        orig_idx,
                        ignore_order_=ignore_order_,
                        thresholds_dict=thresholds_,
                    ):
                        found = True
                        matches += total_scores_
                        break
                if not found:
                    col_name = specified_columns_[i] if specified_columns_ else f"Column index {i}"
                    pred_normalized_sets = []
                    for pred_col in t_pred_list:
                        normalized = [normalize_value(x, tol, is_relative=is_rel) for x in pred_col]
                        pred_normalized_sets.append(set(normalized))

                    shuffled_gold = list(gold_col)
                    random.shuffle(shuffled_gold)
                    unmatched = []
                    max_count = 50
                    max_str_len = 5000
                    for val in shuffled_gold:
                        val_norm = normalize_value(val, tol, is_relative=is_rel)
                        found_val = any(val_norm in ps for ps in pred_normalized_sets)
                        if not found_val:
                            unmatched.append(val)
                            if len(unmatched) >= max_count:
                                break
                            sample_str = ", ".join([str(v) for v in unmatched])
                            if len(sample_str) >= max_str_len:
                                break

                    if unmatched:
                        if len(unmatched) == len(gold_col) and all(
                            isinstance(x, str) for x in gold_col
                        ):
                            logging.warning(
                                f"{col_name}: no values matched, scoring 0 for entire column"
                            )
                        full_str = ", ".join([str(v) for v in unmatched])
                        sample_str = (
                            full_str[:max_str_len] + "..."
                            if len(full_str) > max_str_len
                            else full_str
                        )
                        error_msg = (
                            f"Column '{col_name}' does not match: expected values {sample_str} "
                            f"but none found in prediction "
                            f"(score_rule='divide' scores each column independently)"
                        )
                    else:
                        error_msg = (
                            f"Column '{col_name}' does not match: values are completely different "
                            f"from expected (score_rule='divide' scores each column independently)"
                        )
                    errors.append(error_msg)

            pre_score = matches / total
        else:
            pre_score = 0.0

        if specified_columns_:
            meaning = (
                f"Compared {len(t_gold_list)} specified columns: {', '.join(specified_columns_)}"
            )
        else:
            meaning = f"Compared all {len(t_gold_list)} columns in the CSV files"

        if ignore_order_:
            meaning += " (ignoring row order)"

        if score_rule_ == "all":
            meaning += " with score rule 'all' (all columns must match for full score)"
        else:
            meaning += " with score rule 'divide' (score = matched columns / total columns)"

        meaning += f". Score: {pre_score:.2f}/{total_scores_}"

        return pre_score, errors, meaning

    output: list[float] = []
    output_errors: list[list[str]] = []
    output_meanings: list[str] = []
    output_data: str | None = None
    gold_data: str | None = None

    if not os.path.exists(output_file_name):
        return {
            "score": 0,
            "errors": ["Output file does not exist"],
            "meaning": (
                f"Failed to compare CSV files: output file "
                f"'{os.path.basename(output_file_name)}' does not exist"
            ),
            "output_data": None,
            "gold_data": None,
        }

    try:
        with open(output_file_name) as fh:
            output_total_rows = sum(1 for _ in fh) - 1
    except Exception:
        output_total_rows = 0

    max_gold_rows = 0
    for gold_file in gold_file_name:  # type: ignore[union-attr]
        if os.path.exists(gold_file):
            try:
                with open(gold_file) as fh:
                    gold_rows = sum(1 for _ in fh) - 1
            except Exception:
                gold_rows = 0
            max_gold_rows = max(max_gold_rows, gold_rows)

    if output_total_rows < max_gold_rows:
        return {
            "score": 0,
            "errors": [
                f"Row count mismatch: output has {output_total_rows} rows, "
                f"gold has {max_gold_rows} rows (output too short)"
            ],
            "meaning": (
                f"Failed: output ({output_total_rows} rows) has fewer rows "
                f"than gold ({max_gold_rows} rows)"
            ),
            "output_data": None,
            "gold_data": None,
        }

    all_ignore_order_false = all(not io for io in ignore_order)  # type: ignore[union-attr]
    try:
        df1 = pd.read_csv(
            output_file_name,
            low_memory=False,
            nrows=10000 if (all_ignore_order_false and output_total_rows > 10000) else None,
        )
        if df1.empty:
            return {
                "score": 0,
                "errors": ["Output file is empty"],
                "meaning": (
                    f"Failed to compare CSV files: output file "
                    f"'{os.path.basename(output_file_name)}' is empty"
                ),
                "output_data": None,
                "gold_data": None,
            }
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.warning(f"Failed to read result CSV {output_file_name}: {e}")
        return {
            "score": 0,
            "errors": [f"Failed to read result CSV: {e}"],
            "meaning": (
                f"Failed to compare CSV files: error reading output file "
                f"'{os.path.basename(output_file_name)}': {e}"
            ),
            "output_data": None,
            "gold_data": None,
        }

    for i in range(len(gold_file_name)):  # type: ignore[arg-type]
        try:
            df2 = pd.read_csv(gold_file_name[i], low_memory=False, nrows=10000)  # type: ignore[index]
            if df2.empty:
                output.append(0)
                output_errors.append(["Gold file is empty"])
                output_meanings.append(
                    f"Gold file '{os.path.basename(gold_file_name[i])}' is empty"
                )  # type: ignore[index]
                continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.warning(f"Failed to read expected CSV {gold_file_name[i]}: {e}")  # type: ignore[index]
            output.append(0)
            output_errors.append([f"Failed to read gold CSV: {e}"])
            output_meanings.append(
                f"Error reading gold file '{os.path.basename(gold_file_name[i])}': {e}"  # type: ignore[index]
            )
            continue

        for head_n in [10, 100, 1000, None]:
            output_str = df1.head(head_n).to_string() if head_n else df1.to_string()
            if len(output_str) >= 1000 or head_n is None:
                break

        output_data = output_str[:1000] + "\n..." if len(output_str) >= 1000 else output_str

        for head_n in [10, 100, 1000, None]:
            gold_str = df2.head(head_n).to_string() if head_n else df2.to_string()
            if len(gold_str) >= 1000 or head_n is None:
                break

        gold_data = gold_str[:1000] + "\n..." if len(gold_str) >= 1000 else gold_str

        resolved_thresholds = resolve_threshold_keys(thresholds, df2)

        pre_score, errors, meaning = csv_score(
            df1,
            df2,
            specified_columns_=specified_columns[i],  # type: ignore[index]
            ignore_order_=ignore_order[i],  # type: ignore[index]
            thresholds_=resolved_thresholds,
        )
        output.append(pre_score)
        output_errors.append(errors)
        output_meanings.append(meaning)

    max_idx = output.index(max(output)) if output else 0
    best_meaning = output_meanings[max_idx] if output else "Failed to compare: no valid gold files"
    return {
        "score": max(output) if output else 0,
        "errors": output_errors[max_idx] if output else [],
        "meaning": best_meaning,
        "output_data": output_data,
        "gold_data": gold_data,
    }


def compare_sqlite(
    output_file_name: str, gold_file_name: str | list[str], **options
) -> dict[str, Any]:
    """
    @args:
        output_file_name(str): the pred database
        gold_file_name(str|list[str]): the gold database or database files, maybe multiple
        option(dict): the configuration dictionary
            - specified_schema(dict): the specified schema for the tables
            - ignore_order(bool|list(bool)): whether to ignore the order of the rows
    @return:
        dict: {'score': float, 'errors': list[str], 'meaning': str}
    """
    if set(options.keys()) > {"specified_schema", "ignore_order"}:
        raise ValueError(f"Unexpected options: {options.keys()}")
    if isinstance(gold_file_name, list):
        specified_schema: list[dict] = options.get("specified_schema", [{}] * len(gold_file_name))
    elif isinstance(gold_file_name, str):
        specified_schema = [options.get("specified_schema", {})]
        gold_file_name = [gold_file_name]

    def convert_to_csvs(
        db_path: str, schema: dict, max_rows: int = 10000
    ) -> tuple[list[str], list[str]]:
        """Convert specified tables in a SQLite database to CSV files and return their paths."""
        csv_dir = os.path.dirname(db_path)
        csv_paths: list[str] = []
        table_names: list[str] = []
        conn = sqlite3.connect(db_path)
        try:
            for table_name, column_names in schema.items():
                if column_names is None:
                    query = f"SELECT * FROM {table_name} LIMIT {max_rows}"  # noqa: S608
                else:
                    cols_str = ", ".join([f'"{col}"' for col in column_names])
                    query = f"SELECT {cols_str} FROM {table_name} LIMIT {max_rows}"  # noqa: S608
                df = pd.read_sql_query(query, conn)
                csv_path = os.path.join(csv_dir, f"_{table_name}.csv")
                df.to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
                table_names.append(table_name)
        except Exception as e:
            logging.warning(f"Error converting table from {db_path}: {e}")
        finally:
            conn.close()
        return csv_paths, table_names

    def get_table_names(db_path: str) -> list[str]:
        if not os.path.exists(db_path):
            logging.warning(f"Database file does not exist: {db_path}")
            return []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]

    def get_table_schema(db_path: str, table_name: str) -> list[str]:
        """Get the schema (column names) of a table."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        conn.close()
        return [col[1] for col in columns]

    output_data = None
    gold_data = None

    if not os.path.exists(output_file_name):
        return {
            "score": 0,
            "errors": ["Output database file does not exist"],
            "meaning": (
                f"Failed to compare SQLite databases: output file "
                f"'{os.path.basename(output_file_name)}' does not exist"
            ),
            "output_data": None,
            "gold_data": None,
        }
    output_tables = get_table_names(output_file_name)
    if not output_tables:
        return {
            "score": 0,
            "errors": ["Output database is empty or invalid"],
            "meaning": (
                f"Failed to compare SQLite databases: output database "
                f"'{os.path.basename(output_file_name)}' is empty or invalid"
            ),
            "output_data": None,
            "gold_data": None,
        }

    output_scores: list[float] = []
    output_errors: list[list[str]] = []
    output_meanings: list[str] = []

    for i in range(len(gold_file_name)):  # type: ignore[arg-type]
        gold_db = gold_file_name[i]  # type: ignore[index]

        if not os.path.exists(gold_db):
            logging.warning(f"Gold database file does not exist: {gold_db}")
            output_scores.append(0)
            output_errors.append(
                [f"Gold database file does not exist: {os.path.basename(gold_db)}"]
            )
            output_meanings.append(f"Gold database '{os.path.basename(gold_db)}' does not exist")
            continue

        gold_tables = get_table_names(gold_db)
        if not gold_tables:
            logging.warning(f"Gold database is empty or invalid: {gold_db}")
            output_scores.append(0)
            output_errors.append(
                [f"Gold database is empty or invalid: {os.path.basename(gold_db)}"]
            )
            output_meanings.append(
                f"Gold database '{os.path.basename(gold_db)}' is empty or invalid"
            )
            continue

        if specified_schema[i]:
            tables_to_compare = list(specified_schema[i].keys())
            missing_gold_tables = [t for t in tables_to_compare if t not in gold_tables]
            if missing_gold_tables:
                logging.warning(
                    f"Specified tables not found in gold database: {missing_gold_tables}"
                )
                output_scores.append(0)
                output_errors.append(
                    [
                        f"Specified tables not found in gold database: "
                        f"{', '.join(missing_gold_tables)}"
                    ]
                )
                output_meanings.append(
                    f"Failed: specified tables {missing_gold_tables} not in gold database"
                )
                continue

            missing_output_tables = [t for t in tables_to_compare if t not in output_tables]
            if missing_output_tables:
                logging.warning(
                    f"Specified tables not found in output database: {missing_output_tables}"
                )
                output_scores.append(0)
                output_errors.append(
                    [
                        f"Specified tables not found in output database: "
                        f"{', '.join(missing_output_tables)}"
                    ]
                )
                output_meanings.append(
                    f"Failed: specified tables {missing_output_tables} not in output database"
                )
                continue
        else:
            tables_to_compare = sorted(set(output_tables) & set(gold_tables))
            missing_gold_tables = sorted(set(gold_tables) - set(output_tables))
            missing_output_tables = sorted(set(output_tables) - set(gold_tables))

            if missing_gold_tables:
                logging.warning(f"Tables missing in output database: {missing_gold_tables}")
            if missing_output_tables:
                logging.info(f"Extra tables in output database: {missing_output_tables}")

        if not tables_to_compare:
            output_scores.append(0)
            output_errors.append(["No common tables to compare"])
            output_meanings.append("Failed: no common tables between databases")
            continue

        schema_to_use: dict[str, list[str] | None] = {}
        for table_name in tables_to_compare:
            if specified_schema[i] and table_name in specified_schema[i]:
                schema_to_use[table_name] = specified_schema[i][table_name]
            else:
                schema_to_use[table_name] = None

        gold_csvs, _gold_table_names = convert_to_csvs(gold_db, schema_to_use)
        pred_csvs, _pred_table_names = convert_to_csvs(output_file_name, schema_to_use)

        if len(pred_csvs) != len(gold_csvs) or len(pred_csvs) == 0:
            output_scores.append(0)
            output_errors.append(
                [f"Failed to convert tables to CSV: pred={len(pred_csvs)}, gold={len(gold_csvs)}"]
            )
            output_meanings.append("Failed to convert tables to CSV for comparison")
            for p in gold_csvs + pred_csvs:
                if os.path.exists(p):
                    os.remove(p)
            continue

        table_scores: list[float] = []
        table_errors: list[str] = []

        for j, table_name in enumerate(tables_to_compare):
            if specified_schema[i] and table_name in specified_schema[i]:
                specified_cols = specified_schema[i][table_name]
                if specified_cols is None:
                    specified_cols = []
            else:
                specified_cols = []

            csv_result = compare_csv(
                pred_csvs[j],
                gold_csvs[j],
                specified_columns=specified_cols,
                ignore_order=False,
            )

            table_scores.append(csv_result["score"])
            if csv_result["errors"]:
                table_errors.extend(
                    [f"Table '{table_name}': {err}" for err in csv_result["errors"]]
                )

            if specified_cols == []:
                gold_schema = get_table_schema(gold_db, table_name)
                pred_schema = get_table_schema(output_file_name, table_name)

                missing_cols = set(gold_schema) - set(pred_schema)
                extra_cols = set(pred_schema) - set(gold_schema)

                if missing_cols:
                    table_errors.append(
                        f"Table '{table_name}': missing columns in output: "
                        f"{', '.join(missing_cols)}"
                    )
                    logging.warning(f"Table '{table_name}': missing columns {missing_cols}")

                if extra_cols:
                    logging.info(
                        f"Table '{table_name}': extra columns in output: {', '.join(extra_cols)}"
                    )

        if table_scores:
            overall_score = sum(table_scores) / len(table_scores)
            meaning = (
                f"Compared {len(table_scores)} tables with score rule 'divide' "
                f"(score = average of table scores)"
            )
        else:
            overall_score = 0.0
            meaning = "No tables compared"

        meaning += f". Tables: {', '.join(tables_to_compare)}"

        if specified_schema[i]:
            meaning += f" (with specified schema: {specified_schema[i]})"
        else:
            meaning += " (comparing all common tables)"

        try:
            gold_rows = sum(len(pd.read_csv(csv)) for csv in gold_csvs)
            pred_rows = sum(len(pd.read_csv(csv)) for csv in pred_csvs)
            meaning += f". Total rows - gold: {gold_rows}, output: {pred_rows}"
        except Exception:  # noqa: S110
            pass  # Row count info is informational only

        output_scores.append(overall_score)
        output_errors.append(table_errors)
        output_meanings.append(meaning)

        for p in gold_csvs + pred_csvs:
            if os.path.exists(p):
                os.remove(p)

    if not output_scores:
        return {
            "score": 0,
            "errors": ["No valid gold databases to compare"],
            "meaning": "Failed: no valid gold databases",
            "output_data": None,
            "gold_data": None,
        }

    max_idx = output_scores.index(max(output_scores))
    best_meaning = output_meanings[max_idx]

    return {
        "score": max(output_scores),
        "errors": output_errors[max_idx],
        "meaning": best_meaning,
        "output_data": output_data,
        "gold_data": gold_data,
    }
