"""
Sample MCP Server for ChatGPT Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's chat and deep research features.
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import ValidationError

from src.discovera_fastmcp.pydantic import (
    CsvAggregateInput,
    CsvFilterInput,
    CsvJoinInput,
    CsvReadInput,
    CsvRecordInput,
    CsvSelectInput,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discovera_fastmcp")


# =========================
# Local storage helpers
# =========================
STORAGE_INDEX_PATH = os.path.join("output", "storage_index.json")


# Global CPU-bound worker pool for true parallelism
CPU_POOL = ProcessPoolExecutor()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_storage_index() -> list:
    if not os.path.exists(STORAGE_INDEX_PATH):
        _ensure_dir(os.path.dirname(STORAGE_INDEX_PATH))
        with open(STORAGE_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(STORAGE_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []


def _save_storage_index(entries: list) -> None:
    _ensure_dir(os.path.dirname(STORAGE_INDEX_PATH))
    with open(STORAGE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _register_file(
    path: str,
    category: str,
) -> dict:
    abs_path = str(Path(path).resolve())
    p = Path(abs_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found to register: {abs_path}")

    # Build entry
    stat = p.stat()
    file_hash = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:8]
    # Timestamp omitted in minimal entry to save tokens
    entry_id = f"{int(stat.st_mtime)}_{file_hash}"

    entry = {
        "id": entry_id,
        "path": abs_path,
        "name": p.name,
        "category": category,
    }

    entries = _load_storage_index()
    entries = [
        {k: e.get(k) for k in ("id", "path", "name", "category")}
        for e in entries
        if e.get("path") != abs_path
    ]
    entries.append(entry)
    _save_storage_index(entries)
    return {k: entry.get(k) for k in ("id", "name", "path")}


def _resolve_path_from_id(file_id: str | None) -> str | None:
    """Resolve absolute file path from storage id or provided path."""
    if file_id:
        entries = _load_storage_index()
        for e in entries:
            if e.get("id") == file_id:
                p = e.get("path")
                return str(Path(p).resolve()) if p else None
        raise FileNotFoundError(f"No entry with id {file_id}")
    return None


# =========================
# CSV helpers
# =========================
def _save_dataframe_csv(
    df: pd.DataFrame,
    filename: str,
) -> dict:
    """Save DataFrame to output/user_csvs and register.

    Returns the storage entry dict.
    """
    try:
        base_dir = os.path.join("output", "user_csvs")
        _ensure_dir(base_dir)
        abs_path = os.path.join(base_dir, filename)
        df.to_csv(abs_path, index=False)
        entry = _register_file(
            abs_path,
            category="generated",
        )
        # Return public/minimal info
        return {k: entry.get(k) for k in ("id", "name", "path")}
    except Exception as e:
        raise e


def _preview_dataframe(df: pd.DataFrame, n_rows: int | None = None) -> dict:
    """Return a lightweight preview dict for a DataFrame."""
    try:
        n = int(n_rows or 20)
        head = df.head(n)
        return {
            "columns": list(head.columns),
            "rows": head.to_json(orient="records"),
            "total_rows": int(len(df)),
        }
    except Exception:
        return {"columns": [], "rows": [], "total_rows": 0}


# =========================
# Output limiting helpers
# =========================
MAX_ROWS_DEFAULT = int(os.environ.get("MCP_MAX_ROWS", "200"))
MAX_CELL_CHARS_DEFAULT = int(os.environ.get("MCP_MAX_CELL_CHARS", "240"))


def _truncate_string(value: Any, max_chars: int) -> Any:
    try:
        if isinstance(value, str) and len(value) > max_chars:
            return value[: max_chars - 1] + "\u2026"
        return value
    except Exception:
        return value


def _limit_dataframe_rows(
    df: pd.DataFrame, max_rows: int | None = None
) -> pd.DataFrame:
    try:
        n = int(max_rows or MAX_ROWS_DEFAULT)
        if len(df) <= n:
            return df
        return df.head(n)
    except Exception:
        return df


# =========================
# Keep-alive (SSE heartbeat) helpers
# =========================
async def _try_send_keepalive(
    ctx: Context, event_name: str = "ping", data: Optional[Dict[str, Any]] = None
) -> None:
    """Attempt to send a lightweight keep-alive over SSE if supported by FastMCP.

    This function checks for common streaming hooks. If none are available,
    it safely no-ops to avoid breaking the server.
    """
    try:
        if hasattr(ctx, "report_progress"):
            print(f"Sending keepalive to {event_name}")
            await ctx.report_progress(50, 100, "Working on the task...")
            return
        # Prefer a generic event emitter if available
        if hasattr(ctx, "emit") and callable(getattr(ctx, "emit")):
            print(f"Sending keepalive to {event_name}")
            payload = data or {"ts": datetime.now().isoformat()}
            await ctx.emit(event_name, payload)
            return
        # Fallback: explicit keepalive if exposed
        if hasattr(ctx, "send_keepalive") and callable(getattr(ctx, "send_keepalive")):
            print(f"Sending keepalive to {event_name}")
            await ctx.send_keepalive()
            return
        # If we got here, no supported keepalive hook exists on this FastMCP instance
        print(f"No keepalive hook available; event={event_name} no-op")
    except Exception:
        print(f"Error sending keepalive to {event_name}")
        # Best-effort: swallow errors so keepalive never crashes user tasks
        pass


def start_keepalive(
    ctx: Context, interval_seconds: float = 10.0, event_name: str = "ping"
) -> asyncio.Task:
    """Start a background heartbeat that periodically sends SSE keep-alive.

    Returns an asyncio.Task; cancel it when the long-running job completes.
    Usage:
        task = start_keepalive(mcp, 5.0)
        try:
            ... long running work ...
        finally:
            task.cancel()
    """

    async def _heartbeat_loop() -> None:
        logger.info(
            "Keepalive started: event=%s interval=%.2fs",
            event_name,
            float(interval_seconds),
        )
        try:
            while True:
                await _try_send_keepalive(ctx, event_name)
                await asyncio.sleep(max(0.5, float(interval_seconds)))
        except asyncio.CancelledError:
            logger.info("Keepalive cancelled: event=%s", event_name)
            raise

    return asyncio.create_task(_heartbeat_loop(), name=f"keepalive-{event_name}")


def _truncate_dataframe_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_chars: int | None = None,
) -> pd.DataFrame:
    try:
        limit = int(max_chars or MAX_CELL_CHARS_DEFAULT)
        target_cols = columns or [c for c in df.columns if df[c].dtype == object]
        for c in target_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda v: _truncate_string(v, limit))
        return df
    except Exception:
        return df


def create_server():
    mcp = FastMCP(
        name="discovera_fastmcp",
        # instructions=server_instructions,
        stateless_http=False,
    )

    def _coalesce_none_defaults(model_obj: Any) -> Any:
        """Replace None values with field defaults for pydantic v2 models."""
        try:
            fields = getattr(model_obj.__class__, "model_fields", {})
            for field_name, field in fields.items():
                current_value = getattr(model_obj, field_name, None)
                if current_value is None:
                    if getattr(field, "default", None) is not None:
                        setattr(model_obj, field_name, field.default)
                    else:
                        default_factory = getattr(field, "default_factory", None)
                        if callable(default_factory):
                            setattr(model_obj, field_name, default_factory())
        except Exception:
            pass
        return model_obj

    def _validate_params(params: Any, model_cls, tool_name: str):
        """Coerce incoming params (dict or model) into model_cls or raise a clear error."""
        if isinstance(params, model_cls):
            return _coalesce_none_defaults(params)
        if isinstance(params, dict):
            try:
                # Drop None values so model defaults are applied
                sanitized = {k: v for k, v in params.items() if v is not None}
                model_obj = model_cls(**sanitized)
                return _coalesce_none_defaults(model_obj)
            except ValidationError as e:
                # Keep message concise but actionable
                raise ValueError(
                    f"[{tool_name}] Invalid request parameters for {model_cls.__name__}: {e.errors()}"
                )
        raise TypeError(
            f"[{tool_name}] Invalid parameter type: {type(params).__name__}. Expected dict or {model_cls.__name__}."
        )

    def _exception_payload(
        tool_name: str, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return a structured error payload for tool outputs."""
        return {
            "error": {
                "tool": tool_name,
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "context": context or {},
        }

    def _apply_sort(df: pd.DataFrame, sort_by: list[str] | None) -> pd.DataFrame:
        try:
            if not sort_by:
                return df
            by_cols: list[str] = []
            ascending: list[bool] = []
            for key in sort_by:
                if key.startswith("-"):
                    by_cols.append(key[1:])
                    ascending.append(False)
                else:
                    by_cols.append(key)
                    ascending.append(True)
            # Keep only existing columns
            filtered_cols = [c for c in by_cols if c in df.columns]
            if not filtered_cols:
                return df
            asc_filtered = [
                ascending[i] for i, c in enumerate(by_cols) if c in filtered_cols
            ]
            return df.sort_values(by=filtered_cols, ascending=asc_filtered)
        except Exception:
            return df

    @mcp.tool()
    async def csv_record(
        name: Optional[str] = None,
        csv_text: Optional[str] = None,
        csv_base64: Optional[str] = None,
        csv_path: Optional[str] = None,
        csv_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a CSV to output/user_csvs and register it in storage.

        Exactly one of csv_text, csv_base64, csv_path, csv_url must be provided.

        Args:
            name (str): Logical name for the CSV; used to derive filename (required).
            csv_text (Optional[str]): CSV content as text.
            csv_base64 (Optional[str]): CSV content as base64 (UTF-8 text expected).
            csv_path (Optional[str]): Absolute path to a local CSV file.
            csv_url (Optional[str]): URL to a CSV file.
            tags (Optional[List[str]]): Tags to associate (e.g., user_input, dataset).
            metadata (Optional[Dict[str, Any]]): Arbitrary metadata (e.g., source, description).

        Returns:
            Dict[str, Any]: Registered storage entry for the saved file.
        """
        try:
            # Validate/construct params
            params = _validate_params(
                {
                    "name": name,
                    "csv_text": csv_text,
                    "csv_base64": csv_base64,
                    "csv_path": csv_path,
                    "csv_url": csv_url,
                    "tags": tags,
                    "metadata": metadata,
                },
                CsvRecordInput,
                "csv_record",
            )

            # Determine exactly one source of CSV content
            provided_sources = {
                "csv_text": params.csv_text,
                "csv_base64": params.csv_base64,
                "csv_path": params.csv_path,
                "csv_url": getattr(params, "csv_url", None),
            }
            provided = [k for k, v in provided_sources.items() if v]
            if len(provided) == 0:
                raise ValueError(
                    "Provide exactly one of csv_text, csv_base64, csv_path, or csv_url"
                )
            if len(provided) > 1:
                raise ValueError(
                    f"Multiple inputs provided: {provided}. Provide only one of csv_text, csv_base64, csv_path, csv_url"
                )

            source = provided[0]
            normalized_csv_text = None

            if source == "csv_text":
                text = str(params.csv_text)
                if not text.strip():
                    raise ValueError("csv_text is empty")
                # Remove UTF-8 BOM if present
                text = text.lstrip("\ufeff")
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception as e:
                    raise ValueError(f"csv_text is not a valid CSV: {e}")
                normalized_csv_text = df.to_csv(index=False)

            elif source == "csv_base64":
                try:
                    raw = base64.b64decode(params.csv_base64)
                    # Decode using UTF-8-SIG to drop BOM if present
                    text = raw.decode("utf-8-sig")
                except Exception as e:
                    raise ValueError(f"Invalid base64 CSV (must be UTF-8 text): {e}")
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception as e:
                    raise ValueError(f"Decoded base64 is not a valid CSV: {e}")
                normalized_csv_text = df.to_csv(index=False)

            elif source == "csv_path":
                abs_in_path = str(Path(params.csv_path).resolve())
                if not os.path.exists(abs_in_path):
                    raise FileNotFoundError(f"File not found: {abs_in_path}")
                try:
                    # Read with UTF-8-SIG to remove BOM if present
                    df = pd.read_csv(abs_in_path, encoding="utf-8-sig")
                except Exception as e:
                    raise ValueError(f"Failed to read CSV from path: {e}")
                normalized_csv_text = df.to_csv(index=False)

            elif source == "csv_url":
                try:
                    response = requests.get(params.csv_url, timeout=20)
                    response.raise_for_status()
                    # Parse from bytes and let pandas handle UTF-8-SIG to drop BOM
                    df = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
                except Exception as e:
                    raise ValueError(f"Failed to read CSV from URL: {e}")
                normalized_csv_text = df.to_csv(index=False)

            # Ensure output directory exists
            base_dir = os.path.join("output", "user_csvs")
            _ensure_dir(base_dir)

            # Build unique filename
            filename = f"{params.name}.csv"
            abs_path = os.path.join(base_dir, filename)

            # Persist normalized CSV text
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(normalized_csv_text)

            # Register entry
            tags = ["csv", "user_input"] + (params.tags or [])
            metadata = dict(params.metadata or {})
            metadata.update({"source": source})
            if source == "csv_url":
                metadata["csv_url"] = params.csv_url
            if source == "csv_path":
                metadata["csv_path"] = str(Path(params.csv_path).resolve())

            entry = _register_file(
                abs_path,
                category="user_input",
            )
            logger.info("🛠️[csv_record] CSV file registered successfully: %s", abs_path)
            return entry
        except Exception as e:
            return _exception_payload("csv_record", e, {"name": params.name})

    @mcp.tool()
    async def csv_read(
        id: str,
        n_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read a stored CSV by id on MCP server and return a preview of rows.

        Args:
            id (str): Storage id of the CSV (required).
            n_rows (Optional[int]): Number of rows to preview. Defaults to 20.

        Returns:
            Dict[str, Any]: {"path": str, "columns": list, "rows": list, "total_rows": int}.
        """
        try:
            # Validate/construct params
            params = _validate_params(
                {"id": id, "n_rows": n_rows}, CsvReadInput, "csv_read"
            )

            target_path = None
            if params.id:
                entries = _load_storage_index()
                for e in entries:
                    if e.get("id") == params.id:
                        target_path = e.get("path")
                        break
                if not target_path:
                    raise FileNotFoundError(f"No entry with id {params.id}")
            else:
                raise ValueError("Provide id")

            abs_path = str(Path(target_path).resolve())
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"File not found: {abs_path}")

            df = pd.read_csv(abs_path)
            n = int(params.n_rows or 20)
            head = df.head(n)
            logger.info("🛠️[csv_read] CSV file read successfully: %s", abs_path)
            return {
                "path": abs_path,
                "columns": list(head.columns),
                "rows": head.to_dict(orient="records"),
                "total_rows": int(len(df)),
            }
        except Exception as e:
            return _exception_payload(
                "csv_read", e, {"id": params.id, "path": params.path}
            )

    @mcp.tool()
    async def csv_filter(
        csv_id: str,
        conditions: List[Dict[str, Any]],
        keep_columns: Optional[List[str]] = None,
        sort_by: Optional[List[str]] = None,
        distinct: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Filter a stored CSV by column conditions with AND logic.

        Args:
            csv_id (str): Storage id of the input CSV to filter.
            conditions (List[Dict[str, Any]]): List of filter conditions to apply.
                Each item accepts keys: {"column", "op", "value"} where:
                - column (str): Column to filter on.
                - op (str): One of "==", "!=", ">", ">=", "<", "<=", "in", "not_in",
                  "contains", "not_contains", "startswith", "endswith", "isnull", "notnull".
                - value (Any): Comparison value. For "in"/"not_in", provide a list of values.
            keep_columns (Optional[List[str]]): Optional subset of columns to keep in the output.
            sort_by (Optional[List[str]]): Optional sort keys. Prefix with '-' for descending
                (e.g., ["-padj", "log2fc"]). Non-existing columns are ignored.
            distinct (Optional[bool]): If true, drops duplicate rows after filtering.
            name (Optional[str]): Logical name for the output CSV; if not provided an auto name is used.

        Returns:
            Dict[str, Any]: Preview with keys {"columns", "rows", "total_rows", "storage_entry"}.
        """
        try:
            params = _validate_params(
                {
                    "csv_id": csv_id,
                    "conditions": conditions,
                    "keep_columns": keep_columns,
                    "sort_by": sort_by,
                    "distinct": distinct,
                    "name": name,
                },
                CsvFilterInput,
                "csv_filter",
            )

            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide csv_id")
            df = pd.read_csv(csv_abs_path)

            mask = pd.Series([True] * len(df))
            for cond in params.conditions:
                col = cond.column
                op = (cond.op or "").lower()
                val = cond.value
                if col not in df.columns:
                    # Non-existent column -> always False for this condition
                    mask &= False
                    continue
                series = df[col]
                if op == "==":
                    mask &= series == val
                elif op == "!=":
                    mask &= series != val
                elif op == ">":
                    mask &= series > val
                elif op == ">=":
                    mask &= series >= val
                elif op == "<":
                    mask &= series < val
                elif op == "<=":
                    mask &= series <= val
                elif op == "in":
                    vals = val if isinstance(val, list) else [val]
                    mask &= series.isin(vals)
                elif op == "not_in":
                    vals = val if isinstance(val, list) else [val]
                    mask &= ~series.isin(vals)
                elif op == "contains":
                    mask &= series.astype(str).str.contains(
                        str(val), na=False, case=False
                    )
                elif op == "not_contains":
                    mask &= ~series.astype(str).str.contains(
                        str(val), na=False, case=False
                    )
                elif op == "startswith":
                    mask &= series.astype(str).str.startswith(str(val), na=False)
                elif op == "endswith":
                    mask &= series.astype(str).str.endswith(str(val), na=False)
                elif op == "isnull":
                    mask &= series.isna()
                elif op == "notnull":
                    mask &= series.notna()
                else:
                    raise ValueError(f"Unsupported operator: {cond.op}")

            filtered = df[mask]
            if params.keep_columns:
                existing = [c for c in params.keep_columns if c in filtered.columns]
                if existing:
                    filtered = filtered[existing]
            if params.distinct:
                filtered = filtered.drop_duplicates()
            if params.sort_by:
                filtered = _apply_sort(filtered, params.sort_by)

            entry = _save_dataframe_csv(
                filtered, filename=f"csv_filter_{params.csv_id}.csv"
            )

            preview = _preview_dataframe(filtered)
            preview.update({"storage_entry": entry})
            return preview
        except Exception as e:
            return _exception_payload(
                "csv_filter",
                e,
                {
                    "csv_id": csv_id,
                },
            )

    @mcp.tool()
    async def csv_select(
        csv_id: str,
        columns: Optional[List[str]] = None,
        rename: Optional[Dict[str, str]] = None,
        distinct: Optional[bool] = None,
        sort_by: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Project a stored CSV to a subset of columns, optionally rename, distinct, and sort.

        Args:
            csv_id (str): Storage id of the input CSV to project.
            columns (Optional[List[str]]): Ordered list of columns to keep. If omitted, keep all.
            rename (Optional[Dict[str, str]]): Mapping of old -> new column names to apply
                (**avoid renaming gene column to "symbol"**).
            distinct (Optional[bool]): If true, drops duplicate rows after selection.
            sort_by (Optional[List[str]]): Optional sort keys. Prefix with '-' for descending.
            name (Optional[str]): Logical name for the output CSV; if not provided an auto name is used.

        Returns:
            Dict[str, Any]: Preview with keys {"columns", "rows", "total_rows", "storage_entry"}.
        """
        try:
            params = _validate_params(
                {
                    "csv_id": csv_id,
                    "columns": columns,
                    "rename": rename,
                    "distinct": distinct,
                    "sort_by": sort_by,
                    "name": name,
                },
                CsvSelectInput,
                "csv_select",
            )

            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide csv_id")
            df = pd.read_csv(csv_abs_path)

            out = df
            if params.columns:
                existing = [c for c in params.columns if c in out.columns]
                out = out[existing]
            if params.rename:
                mapping = {
                    k: v for k, v in (params.rename or {}).items() if k in out.columns
                }
                out = out.rename(columns=mapping)
            if params.distinct:
                out = out.drop_duplicates()
            if params.sort_by:
                out = _apply_sort(out, params.sort_by)

            entry = _save_dataframe_csv(out, filename=f"csv_select_{params.csv_id}.csv")

            preview = _preview_dataframe(out)
            preview.update({"storage_entry": entry})
            return preview
        except Exception as e:
            return _exception_payload(
                "csv_select",
                e,
                {
                    "csv_id": csv_id,
                },
            )

    @mcp.tool()
    async def csv_join(
        left_csv_id: str,
        right_csv_id: str,
        on: str,
        how: Optional[str] = None,
        select: Optional[Dict[str, List[str]]] = None,
        suffixes: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Join two CSVs on a key with flexible select/suffix options, returning the merged table.

        Args:
            left_csv_id (str): Storage id of the left CSV (required).
            right_csv_id (str): Storage id of the right CSV (required).
            on (str): Join key column present in both CSVs (e.g., "gene").
            how (Optional[str]): Join type. One of "inner" | "left" | "right" | "outer".
                Defaults to "inner".
            select (Optional[Dict[str, List[str]]]): Optional selection mapping to reduce columns.
                Format: {"left": [cols...], "right": [cols...]}. If omitted, all columns are used.
                The join key is always included.
            suffixes (Optional[List[str]]): Suffixes for overlapping column names (length 2), e.g.,
                ["_left", "_right"]. Defaults to ["_left", "_right"].
            name (Optional[str]): Logical name for the output CSV; if omitted an auto name is used.

        Returns:
            Dict[str, Any]: Preview with keys {"columns", "rows", "total_rows", "storage_entry"}.
        """
        try:
            params = _validate_params(
                {
                    "left_csv_id": left_csv_id,
                    "right_csv_id": right_csv_id,
                    "on": on,
                    "how": how,
                    "select": select,
                    "suffixes": suffixes,
                    "name": name,
                },
                CsvJoinInput,
                "csv_join",
            )

            left_path = _resolve_path_from_id(params.left_csv_id)
            right_path = _resolve_path_from_id(params.right_csv_id)
            if not left_path or not right_path:
                raise ValueError("Provide left_csv_id and right_csv_id")
            left_df = pd.read_csv(left_path)
            right_df = pd.read_csv(right_path)

            if params.on not in left_df.columns or params.on not in right_df.columns:
                raise ValueError(f"Join key '{params.on}' not found in both CSVs")

            # Column selection
            if params.select and isinstance(params.select, dict):
                lcols = params.select.get("left") or list(left_df.columns)
                rcols = params.select.get("right") or list(right_df.columns)
            else:
                lcols = list(left_df.columns)
                rcols = list(right_df.columns)
            lcols = [c for c in lcols if c in left_df.columns]
            rcols = [c for c in rcols if c in right_df.columns and c != params.on]

            left_sel = left_df[
                [c for c in set([params.on] + lcols) if c in left_df.columns]
            ]
            right_sel = right_df[
                [c for c in set([params.on] + rcols) if c in right_df.columns]
            ]

            how = (params.how or "inner").lower()
            if how not in {"inner", "left", "right", "outer"}:
                raise ValueError("how must be one of 'inner'|'left'|'right'|'outer'")

            suffixes = (
                params.suffixes
                if (params.suffixes and len(params.suffixes) == 2)
                else ["_left", "_right"]
            )
            merged = left_sel.merge(
                right_sel, how=how, on=params.on, suffixes=tuple(suffixes)
            )

            entry = _save_dataframe_csv(
                merged,
                filename=f"csv_join_{params.left_csv_id}_{params.right_csv_id}_{params.on}.csv",
            )

            preview = _preview_dataframe(merged)
            preview.update({"storage_entry": entry})
            return preview
        except Exception as e:
            return _exception_payload(
                "csv_join",
                e,
                {
                    "left_csv_id": left_csv_id,
                    "right_csv_id": right_csv_id,
                    "on": on,
                },
            )

    @mcp.tool()
    async def csv_aggregate(
        csv_id: str,
        aggregations: Dict[str, Dict[str, Union[str, List[str]]]],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate columns within a CSV to produce new columns (e.g., mean of replicate log2FC/padj).

        Args:
            csv_id (str): Storage id of the input CSV (required).
            aggregations (dict): Mapping of output_col -> {"func": "mean|sum|min|max|median|first|last",
                "cols": [colA, colB, ...]} specifying how to derive each output column.
                - mean: row-wise numeric mean across columns (NaNs ignored)
                - sum: row-wise numeric sum
                - min|max|median: row-wise reducer across columns
                - first|last: take first/last column values
            name (Optional[str]): Logical name for the output CSV; auto if omitted.

        Returns:
            Dict[str, Any]: Preview with keys {"columns", "rows", "total_rows", "storage_entry"}.
        """
        try:
            params = _validate_params(
                {
                    "csv_id": csv_id,
                    "aggregations": aggregations,
                    "name": name,
                },
                CsvAggregateInput,
                "csv_aggregate",
            )

            csv_path = _resolve_path_from_id(params.csv_id)
            if not csv_path:
                raise ValueError("Provide csv_id")
            df = pd.read_csv(csv_path)

            for out_col, spec in (params.aggregations or {}).items():
                func = (spec or {}).get("func", "mean").lower()
                cols = (spec or {}).get("cols") or []
                if not cols:
                    raise ValueError(
                        f"No columns specified to aggregate for '{out_col}'"
                    )
                for c in cols:
                    if c not in df.columns:
                        raise ValueError(f"Aggregation source column '{c}' not found")
                values = pd.concat(
                    [pd.to_numeric(df[c], errors="coerce") for c in cols], axis=1
                )
                if func == "mean":
                    df[out_col] = values.mean(axis=1, skipna=True)
                elif func == "sum":
                    df[out_col] = values.sum(axis=1, skipna=True)
                elif func == "min":
                    df[out_col] = values.min(axis=1, skipna=True)
                elif func == "max":
                    df[out_col] = values.max(axis=1, skipna=True)
                elif func == "median":
                    df[out_col] = values.median(axis=1, skipna=True)
                elif func == "first":
                    df[out_col] = values.iloc[:, 0]
                elif func == "last":
                    df[out_col] = values.iloc[:, -1]
                else:
                    raise ValueError(
                        f"Unsupported aggregation func '{func}' for '{out_col}'"
                    )

            entry = _save_dataframe_csv(
                df,
                filename=f"csv_aggregate_{params.csv_id}_{'_'.join(params.aggregations.keys())}.csv",
            )

            preview = _preview_dataframe(df)
            preview.update({"storage_entry": entry})
            return preview
        except Exception as e:
            return _exception_payload(
                "csv_aggregate",
                e,
                {
                    "csv_id": csv_id,
                },
            )

    return mcp


def main():
    """Main function to start the MCP server."""
    server = create_server()

    try:
        server.run(transport="sse", host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
