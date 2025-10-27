"""
Sample MCP Server for ChatGPT Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's chat and deep research features.
"""

import asyncio
import base64
import functools
import hashlib
import io
import json
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd
import requests
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import ValidationError

from src.discovera.bkd.gsea import nrank_ora, rank_gsea, run_deseq2
from src.discovera.bkd.mygeneinfo import fetch_gene_annota
from src.discovera.bkd.pubmed import literature_timeline
from src.discovera.bkd.pubmed import prioritize_genes as prioritize_genes_fn
from src.discovera.bkd.query_indra import nodes_batch, normalize_nodes
from src.discovera.bkd.rummagene import (
    enrich_query,
    fetch_pmc_info,
    gene_sets_paper_query,
    genes_query,
)
from src.discovera.bkd.rummagene import search_pubmed as rumm_search_pubmed
from src.discovera.bkd.rummagene import table_search_query
from src.discovera_fastmcp.pydantic import (
    CountEdgesInput,
    CsvAggregateInput,
    CsvFilterInput,
    CsvJoinInput,
    CsvReadInput,
    CsvRecordInput,
    CsvSelectInput,
    EnrichRummaInput,
    GeneInfoInput,
    GseaPipeInput,
    LiteratureTrendsInput,
    OraPipeInput,
    PrioritizeGenesInput,
    QueryGenesInput,
    QueryStringRummaInput,
    QueryTableRummaInput,
    RunDeseq2GseaInput,
    SetsInfoRummInput,
    StorageGetInput,
    StorageListInput,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discovera_fastmcp")


# =========================
# Local storage helpers
# =========================
STORAGE_INDEX_PATH = os.path.join("output", "storage_index.json")

# S3 configuration (optional)
S3_BUCKET = os.getenv("OUTPUTS_S3_BUCKET")
S3_PREFIX = os.getenv("OUTPUTS_S3_PREFIX", "outputs")
S3_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_PRESIGN_TTL = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "604800"))  # 7 days
_S3_CLIENT = None


def _get_s3():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    if not S3_BUCKET:
        return None
    try:
        _S3_CLIENT = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=BotoConfig(
                signature_version="s3v4", s3={"addressing_style": "virtual"}
            ),
        )
        return _S3_CLIENT
    except Exception:
        return None


def _s3_upload_and_presign(abs_path: str) -> dict | None:
    try:
        s3 = _get_s3()
        if not s3 or not S3_BUCKET:
            return None
        if not os.path.exists(abs_path):
            return None
        # Detect content type from extension
        ext = (os.path.splitext(abs_path)[1] or "").lower()
        if ext == ".csv":
            content_type = "text/csv"
        elif ext == ".txt":
            content_type = "text/plain"
        elif ext == ".json":
            content_type = "application/json"
        elif ext == ".png":
            content_type = "image/png"
        elif ext == ".svg":
            content_type = "image/svg+xml"
        else:
            content_type = "application/octet-stream"

        # Build key with date + filename; prefix includes a date partition
        today = datetime.now().strftime("%Y-%m-%d")
        filename = os.path.basename(abs_path)
        s3_key = f"{S3_PREFIX}/{today}/{filename}"
        with open(abs_path, "rb") as f:
            body = f.read()
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=body, ContentType=content_type)
        url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": s3_key,
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=S3_PRESIGN_TTL,
        )
        return {"s3_key": s3_key, "s3_presigned_url": url}
    except (BotoCoreError, NoCredentialsError, ClientError, Exception):
        return None


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


def _slugify(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in value.strip())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "file"


def _register_file(
    path: str,
    category: str,
    to_s3: bool = False,
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

    # Optionally upload to S3 and attach presigned URL
    if to_s3:
        try:
            s3_info = _s3_upload_and_presign(abs_path)
            if s3_info:
                entry["s3_presigned_url"] = s3_info.get("s3_presigned_url")
        except Exception:
            pass

    entries = _load_storage_index()
    # Preserve existing entries keyed by id to avoid invalidating previously returned ids,
    # even when the same path is re-registered (e.g., repeated writes to a deterministic filename).
    entries_by_id: dict[str, dict] = {}
    for e in entries:
        eid = e.get("id")
        if not eid:
            continue
        entries_by_id[eid] = {
            k: e.get(k) for k in ("id", "path", "name", "category", "s3_presigned_url")
        }
    # Add/overwrite current entry by its unique id
    entries_by_id[entry_id] = entry
    _save_storage_index(list(entries_by_id.values()))
    if to_s3:
        return {k: entry.get(k) for k in ("id", "name", "path", "s3_presigned_url")}
    else:
        return {k: entry.get(k) for k in ("id", "name", "path")}


def _filter_entries(
    entries: list,
    origin_tool: str | None = None,
    tag: str | None = None,
    ext: str | None = None,
    category: str | None = None,
    name_contains: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> list:
    """Filter minimal storage entries using only supported fields.

    Legacy arguments (origin_tool, tag, ext, since, until) are accepted for
    compatibility but ignored because entries no longer store those attributes.
    """

    def _match(e: dict) -> bool:
        if category and e.get("category") != category:
            return False
        if name_contains and name_contains.lower() not in (e.get("name") or "").lower():
            return False
        return True

    return [e for e in entries if _match(e)]


def _read_file_content(entry: dict, with_content: bool, max_bytes: int) -> dict:
    # Only return minimal metadata + optional content
    result = {k: entry.get(k) for k in ("id", "name", "path", "s3_presigned_url")}
    if not with_content:
        return result
    try:
        path = entry.get("path")
        if not path or not os.path.exists(path):
            result["content_text"] = None
            result["truncated"] = False
            return result
        size = os.path.getsize(path)
        truncated = size > max_bytes
        # Text preview only (saves tokens vs base64)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_bytes)
        result["content_text"] = data
        result["truncated"] = truncated
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def _resolve_path_from_id(file_id: str | None) -> str | None:
    """Resolve absolute file path from storage id or provided path."""
    if file_id:
        entries = _load_storage_index()
        for e in entries:
            if e.get("id") == file_id:
                p = e.get("path")
                return str(Path(p).resolve()) if p else None
        # Fallback: if exact id not found, attempt to match by stable hash suffix
        # This helps when a file is re-registered at the same path, generating a new timestamp-based id
        # but preserving the path-hash suffix (last 8 hex chars after the underscore).
        try:
            suffix = str(file_id).split("_")[-1]
            candidates = [
                e for e in entries if str(e.get("id", "")).split("_")[-1] == suffix
            ]
            if len(candidates) == 1:
                p = candidates[0].get("path")
                return str(Path(p).resolve()) if p else None
        except Exception:
            pass
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
            to_s3=False,
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
MAX_IDS_PER_YEAR = int(os.environ.get("MCP_MAX_IDS_PER_YEAR", "5"))
STORAGE_MAX_BYTES_DEFAULT = int(os.environ.get("MCP_STORAGE_MAX_BYTES", "65536"))


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


def _run_nodes_batches_sync(nodes_lists: list[tuple[str, ...]]):
    """Run nodes_batch over combos and collect DataFrames (sync function for CPU pool)."""
    results: list[pd.DataFrame] = []
    try:
        for combo in nodes_lists:
            try:
                df = nodes_batch(combo)
                if df is not None:
                    results.append(df)
            except Exception:
                # best-effort: skip failing combo
                continue
    except Exception:
        pass
    return results


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

    @mcp.tool
    async def run_deseq2_gsea_pipe(
        ctx: Context,
        raw_counts_csv_id: str,
        sample_groups: Optional[Dict[str, List[str]]] = None,
        gene_column: Optional[str] = None,
        gene_sets: Optional[List[str]] = None,
        p_value_threshold: Optional[float] = None,
        fdr_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Performs DESeq2 differential expression (DE) from raw RNA‑seq counts, then runs
        Gene Set Enrichment Analysis (GSEA, preranked) on the DE results to identify enriched pathways.

        High-level behavior
        - Input: a raw counts matrix (genes × samples) and a mapping of samples to groups.
        - Step 1 (DE): Runs DESeq2 to compute per-gene log2FoldChange and adjusted p-values (padj).
        - Step 2 (GSEA): Ranks genes by log2FoldChange and runs preranked GSEA across the selected libraries.
        - Output: Top GSEA results in the immediate response + storage IDs for full
          DESeq2 and GSEA result tables (use CSV tools to fetch all rows).

        Args:
            raw_counts_csv_id (str):
                Storage ID for the raw counts CSV (genes × samples). One column must hold gene identifiers
                (see `gene_column`). Prefer raw counts over normalized matrices.
            sample_groups (Optional[Dict[str, List[str]]]):
                Mapping of {group_name: [sample_id, ...]} used to build DE design (default ~ group)..
                Numeric sample patterns like "3_1", "3_2", "7_1" are supported.
                e.g.
                sample_groups = {
                    "CBD":  ["3_1", "3_2", "3_3"],
                    "DMSO": ["7_1", "7_2", "7_3"],
                }
            gene_column (str, optional):
                Column name in the raw counts CSV that contains gene identifiers.
                Defaults to the first non-sample column.
            gene_sets (Optional[List[str]]):
                Gene set libraries for GSEA. Defaults to:
                ["KEGG_2016", "GO_Biological_Process_2023", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"].
            p_value_threshold (float, optional): Filter GSEA results by p-value. Defaults to 0.05.
                If there are not enough results, rerun with a higher threshold.
            fdr_threshold (float, optional): Filter GSEA results by FDR. Defaults to 0.05.
                If there are not enough results, rerun with a higher threshold.

        Returns:
            Dict[str, Any] with keys including:
                - top_high_nes: Top 5 positive NES results per library (preview only).
                - top_low_nes: Top 5 negative NES results per library (preview only).
                - deseq2_preview: Preview of DESeq2 results (columns, rows, total_rows).
                - deseq2_result_metadata: Storage entry for the full DESeq2 results CSV.
                - gsea: {
                    "gsea_result_metadata": Storage entry for the full GSEA results CSV,
                    "counts_by_database": Up/Down counts per library (for ratio summaries),
                }
        """
        # Validate/construct params
        params = _validate_params(
            {
                "raw_counts_csv_id": raw_counts_csv_id,
                "sample_groups": sample_groups,
                "gene_column": gene_column,
                "gene_sets": gene_sets,
                "p_value_threshold": p_value_threshold,
                "fdr_threshold": fdr_threshold,
            },
            RunDeseq2GseaInput,
            "run_deseq2_gsea_pipe",
        )

        try:
            # Resolve paths
            raw_counts_path = _resolve_path_from_id(params.raw_counts_csv_id)
            if not raw_counts_path:
                raise ValueError("Provide raw_counts_csv_id")

            raw_counts_df = pd.read_csv(raw_counts_path)

            # Build sample metadata DataFrame from mapping or infer heuristically
            sample_col_name = "SampleID"
            group_col_name = "Group"
            gene_col = params.gene_column or "Unnamed: 0"
            sample_cols = [c for c in raw_counts_df.columns if c != gene_col]

            if params.sample_groups:
                records: list[dict[str, str]] = []
                for group_name, sample_ids in (params.sample_groups or {}).items():
                    for sid in sample_ids or []:
                        if sid in sample_cols:
                            records.append(
                                {
                                    sample_col_name: str(sid),
                                    group_col_name: str(group_name),
                                }
                            )
                if not records:
                    raise ValueError(
                        "sample_groups provided but no matching sample IDs found in raw counts columns"
                    )
                sample_meta_df = pd.DataFrame.from_records(records)
                # Ensure raw counts are restricted and ordered to the provided samples
                sample_order = [r[sample_col_name] for r in records]
                # Deduplicate while preserving order
                seen: set[str] = set()
                sample_order = [
                    s for s in sample_order if not (s in seen or seen.add(s))
                ]
                # Restrict columns to gene + selected samples (in the given order)
                kept_cols = [gene_col] + [
                    s for s in sample_order if s in raw_counts_df.columns
                ]
                raw_counts_df = raw_counts_df[kept_cols]
            else:
                # Naive inference: take token before '_' as group label
                records = []
                for c in sample_cols:
                    group_guess = str(c).split("_")[0]
                    records.append(
                        {sample_col_name: str(c), group_col_name: group_guess}
                    )
                sample_meta_df = pd.DataFrame.from_records(records)
                # Keep all sample columns and preserve their original order
                kept_cols = [gene_col] + [c for c in sample_cols]
                raw_counts_df = raw_counts_df[kept_cols]

            # start sending heartbeat
            heartbeat_task = start_keepalive(ctx, 5.0)

            # Run DESeq2 in a process to achieve true parallelism (CPU-bound)
            de_results = await asyncio.get_running_loop().run_in_executor(
                CPU_POOL,
                functools.partial(
                    run_deseq2,
                    raw_counts_df,
                    sample_meta_df,
                    params.gene_column,
                    sample_col_name,
                    group_col_name,
                ),
            )

            # cancel heartbeat
            heartbeat_task.cancel()

            # Save DESeq2 results and register
            _save_dataframe_csv(
                de_results, filename=f"deseq2_results_{params.raw_counts_csv_id}.csv"
            )
            # Run GSEA directly using helper to avoid calling MCP tool from inside server
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # start sending heartbeat
            heartbeat_task = start_keepalive(ctx, 5.0)

            sorted_results = await asyncio.get_running_loop().run_in_executor(
                CPU_POOL,
                functools.partial(
                    rank_gsea,
                    dataset=de_results,
                    gene_sets=params.gene_sets,
                    hit_col="GeneID",
                    corr_col="log2FoldChange",
                    min_size=5,
                    max_size=200,
                    p_value_threshold=params.p_value_threshold,
                    fdr_threshold=params.fdr_threshold,
                    timestamp=timestamp,
                ),
            )

            # cancel heartbeat
            heartbeat_task.cancel()

            if isinstance(sorted_results, dict):
                # Safety: in case rank_gsea returns a dict unexpectedly
                sorted_results = pd.DataFrame(sorted_results)

            if sorted_results is None or len(sorted_results) == 0:
                logger.info("[gsea_pipe] No GSEA results after filtering")
                return {
                    "top_high_nes": {},
                    "top_low_nes": {},
                    "gsea_result_metadata": {},
                }

            # rank_gsea returns results sorted by NES ascending
            counts_by_database = {}
            top_high_nes = {}
            top_low_nes = {}
            if "Data Base" in sorted_results.columns:
                databases = sorted_results["Data Base"].unique()
                for database in databases:
                    sorted_results_db = sorted_results[
                        sorted_results["Data Base"] == database
                    ]
                    total_ups = int((sorted_results_db["NES"] > 0).sum())
                    total_downs = int((sorted_results_db["NES"] < 0).sum())
                    # Clip the leading genes to at most 5
                    up_genes = sorted_results_db.head(5)
                    up_genes["Lead_genes"] = up_genes["Lead_genes"].apply(
                        lambda x: x.split(",")
                    )
                    down_genes = sorted_results_db.tail(5)
                    down_genes["Lead_genes"] = down_genes["Lead_genes"].apply(
                        lambda x: x.split(",")
                    )
                    top_low_nes[database] = down_genes.to_dict(orient="records")
                    top_high_nes[database] = up_genes.to_dict(orient="records")
                    counts_by_database[database] = {
                        "up": total_ups,
                        "down": total_downs,
                        "total": total_ups + total_downs,
                    }
            else:
                total_ups = int((sorted_results["NES"] > 0).sum())
                total_downs = int((sorted_results["NES"] < 0).sum())
                up_genes = sorted_results.head(10)
                up_genes["Lead_genes"] = up_genes["Lead_genes"].apply(
                    lambda x: x.split(",")
                )
                down_genes = sorted_results.tail(10)
                down_genes["Lead_genes"] = down_genes["Lead_genes"].apply(
                    lambda x: x.split(",")
                )
                top_low_nes = down_genes.to_dict(orient="records")
                top_high_nes = up_genes.to_dict(orient="records")
                counts_by_database = {
                    "up": total_ups,
                    "down": total_downs,
                    "total": total_ups + total_downs,
                }

            # Register generated CSV and plot paths produced by rank_gsea
            _ensure_dir("output")
            csv_path = os.path.join("output", f"gsea_results_{timestamp}.csv")
            plot_path = os.path.join("output", f"gsea_results_{timestamp}.png")

            gsea_csv_metadata = None
            gsea_plot_metadata = None

            try:
                if os.path.exists(csv_path):
                    gsea_csv_metadata = _register_file(
                        csv_path, category="generated", to_s3=True
                    )
            except Exception:
                pass

            try:
                if os.path.exists(plot_path):
                    gsea_plot_metadata = _register_file(
                        plot_path, category="generated", to_s3=True
                    )
            except Exception:
                pass

            logger.info(
                (
                    f"[gsea_pipe] Results filtered by p<{params.p_value_threshold}, "
                    f"FDR<{params.fdr_threshold}: {len(sorted_results)}"
                )
            )

            return {
                "counts_by_database": counts_by_database,
                "top_high_nes": top_high_nes,
                "top_low_nes": top_low_nes,
                "gsea_result_metadata": gsea_csv_metadata,
                "gsea_plot_metadata": gsea_plot_metadata,
            }
        except Exception as e:
            context = {
                "dataset": params.csv_id,
                "hit_col": params.hit_col,
                "corr_col": params.corr_col,
                "gene_sets": params.gene_sets,
                "min_size": params.min_size,
                "max_size": params.max_size,
                "p_value_threshold": params.p_value_threshold,
                "fdr_threshold": params.fdr_threshold,
            }
            return _exception_payload("gsea_pipe", e, context)

    @mcp.tool
    async def query_genes(
        ctx: Context,
        genes: List[str],
        size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query the INDRA database for relationships among a list of genes.

        Args:
            genes (List[str]): Gene symbols to query. For performance, keep ≤ 10 genes.
            size (Optional[int]): Combination size (k) used when querying k-combinations of genes.
                Defaults to 2.

        Returns:
            Dict[str, Any]: Tabular results (as a dict) with columns such as:
                - nodes, type, subj.name, obj.name, belief, text, text_refs.PMID,
                  text_refs.DOI, text_refs.PMCID, text_refs.SOURCE, text_refs.READER, url
        """
        # Validate/construct params
        params = _validate_params(
            {"genes": genes, "size": size}, QueryGenesInput, "query_genes"
        )

        nodes = params.genes
        results = []

        nodes = normalize_nodes(nodes)

        nodes_lists = list(combinations(nodes, r=int(params.size)))

        # start sending heartbeat
        heartbeat_task = start_keepalive(ctx, 5.0)

        try:
            # Run nodes_batch over all combinations in a separate process for CPU-bound work
            results = await asyncio.get_running_loop().run_in_executor(
                CPU_POOL,
                functools.partial(
                    _run_nodes_batches_sync,
                    nodes_lists,
                ),
            )
        finally:
            # cancel heartbeat regardless of success/failure
            heartbeat_task.cancel()

        if results:
            combined_df = pd.concat(results, ignore_index=True)
            # Select only the desired columns from the combined DataFrame
            selected_columns = [
                "nodes",
                "type",
                "subj.name",
                "obj.name",
                "belief",
                "text",
                "text_refs.PMID",
                "text_refs.DOI",
                "text_refs.PMCID",
                "text_refs.SOURCE",
                "text_refs.READER",
            ]
            # Ensure that the columns exist before selecting to avoid KeyErrors
            existing_columns = [
                col for col in selected_columns if col in combined_df.columns
            ]
            combined_df = combined_df[existing_columns]
            # Safe URL construction if DOI present
            if "text_refs.DOI" in combined_df.columns:
                combined_df["url"] = "https://doi.org/" + combined_df[
                    "text_refs.DOI"
                ].astype(str)
            else:
                combined_df["url"] = ""

            # Ensure belief column exists for sorting
            if "belief" not in combined_df.columns:
                combined_df["belief"] = 0.0

            # TODO: check if this is the best way to handle this
            # Step 1: Pick one row with highest belief per unique 'type'
            # Sort by belief if available, then pick one per type when possible
            type_sorted = combined_df.sort_values(
                by=["belief"],
                ascending=[False] if "belief" in combined_df.columns else [True],
            )
            if "type" in type_sorted.columns:
                type_representatives = type_sorted.drop_duplicates(
                    subset="type", keep="first"
                )
            else:
                # Fallback: take top rows as representatives
                type_representatives = type_sorted.head(min(5, len(type_sorted)))

            # Step 2: Exclude already selected rows
            remaining_df = combined_df.drop(type_representatives.index)

            # Step 3: Track used subj and obj names
            subj_col = (
                "subj.name" if "subj.name" in type_representatives.columns else None
            )
            obj_col = "obj.name" if "obj.name" in type_representatives.columns else None
            used_subj = set(type_representatives[subj_col]) if subj_col else set()
            used_obj = set(type_representatives[obj_col]) if obj_col else set()

            # Step 4: Define mask to prioritize diverse subj/obj
            if subj_col and subj_col in remaining_df.columns:
                remaining_df = remaining_df.assign(
                    is_new_subj=~remaining_df[subj_col].isin(used_subj)
                )
            else:
                remaining_df = remaining_df.assign(is_new_subj=True)
            if obj_col and obj_col in remaining_df.columns:
                remaining_df = remaining_df.assign(
                    is_new_obj=~remaining_df[obj_col].isin(used_obj)
                )
            else:
                remaining_df = remaining_df.assign(is_new_obj=True)

            # Step 5: Sort by new subj/obj and belief
            sort_keys = [
                key
                for key in ["is_new_subj", "is_new_obj", "belief"]
                if key in remaining_df.columns
            ]
            if sort_keys:
                remaining_df = remaining_df.sort_values(
                    by=sort_keys,
                    ascending=[False if k != "belief" else False for k in sort_keys],
                )

            # Step 6: Select additional rows to make total 20
            additional_needed = 20 - len(type_representatives)
            additional_rows = remaining_df.head(additional_needed)

            # Combine and reset index
            final_df = pd.concat([type_representatives, additional_rows]).reset_index(
                drop=True
            )

            # Truncate verbose text fields and cap rows defensively
            final_df = _truncate_dataframe_columns(
                final_df, ["text"], MAX_CELL_CHARS_DEFAULT
            )
            final_df = _limit_dataframe_rows(final_df, MAX_ROWS_DEFAULT)
            output_dict = final_df.to_dict(orient="records")
            if output_dict and len(output_dict) > 0:
                return {"results": output_dict}
            else:
                return {"results": []}
        else:
            return {"results": []}

    @mcp.tool
    async def count_edges(
        edges: List[Dict[str, Any]],
        grouping: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Group and count interactions (edges) by a chosen grouping level.

        Args:
            edges (List[Dict[str, Any]]): REQUIRED Edge records to aggregate (tabular dict format).
            grouping (Optional[str]): One of "summary" | "detailed" | "view". Defaults to "detailed".
                - summary: group by nodes
                - detailed: group by nodes, type, subj.name, obj.name
                - view: group by nodes, type

        Returns:
            Dict[str, Any]: Grouped table with an added "count" column.
        """
        # Validate/construct params
        params = _validate_params(
            {"edges": edges, "grouping": grouping}, CountEdgesInput, "count_edges"
        )

        # Define grouping options
        group_options = {
            "summary": ["nodes"],
            "detailed": ["nodes", "type", "subj.name", "obj.name"],
            "view": ["nodes", "type"],
        }

        # Convert JSON-like input to DataFrame
        edges_list = params.edges or []
        edges_df = pd.DataFrame(edges_list)

        # Validate group_type
        if params.grouping not in group_options:
            raise ValueError(
                f"Invalid group_type '{params.grouping}'. Choose from {list(group_options.keys())}."
            )

        group_columns = group_options[params.grouping]

        # Check if the provided group columns exist in the DataFrame
        if not set(group_columns).issubset(edges_df.columns):
            raise ValueError(
                f"The DataFrame does not contain the required columns for grouping: {group_columns}"
            )

        # Group by the specified columns and count the occurrences
        grouped_df = edges_df.groupby(group_columns).size().reset_index(name="count")
        grouped_df = _limit_dataframe_rows(grouped_df, MAX_ROWS_DEFAULT)
        output_dict = grouped_df.to_dict(orient="records")
        if output_dict and len(output_dict) > 0:
            return {"results": output_dict}
        else:
            return {"results": []}

    @mcp.tool
    async def gsea_pipe(
        ctx: Context,
        csv_id: str,
        hit_col: str = "hit",
        corr_col: str = "corr",
        gene_sets: Optional[List[str]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        p_value_threshold: Optional[float] = None,
        fdr_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Performs Gene Set Enrichment Analysis (GSEA), a computational method
        used to determine whether a set of genes related to a biological function or pathway
        shows a consistent pattern of upregulation or downregulation between two conditions
        (e.g., healthy vs. diseased, treated vs. untreated). GSEA helps identify pathways
        that are significantly enriched in the data. p-value and FDR thresholds are used to filter the results.

        Note for LLM agents:
            - This tool returns only the top results (high/low NES) for brevity.
            - To explore more rows, use `csv_filter`, `csv_select`, `csv_join` or
              `csv_aggregate` on returned gsea_result_metadata.

        Args:
            csv_id (str): ID of a stored CSV entry (required).

            hit_col (str): The column name in the dataset that contains gene symbols
                (e.g., "VWA2", "TSC22D4", etc.). These will be used as identifiers
                to match against gene sets during enrichment (required).
                - Usually the first column in the dataset.

            corr_col (str): The column in the ranked gene list containing correlation or scoring values,
                which are used to rank genes by association with a condition (required).

            gene_sets (Optional[List[str]]): A list of predefined gene set collections used for enrichment analysis.
                Defaults to ["KEGG_2016", "GO_Biological_Process_2023",
                "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"].
                - KEGG_2016: Kyoto Encyclopedia of Genes and Genomes (metabolic and signaling pathways).
                - GO_Biological_Process_2023: Gene Ontology (GO) focusing on biological processes
                  (e.g., cell division, immune response).
                - Reactome_Pathways_2024: Reactome database of curated molecular pathways.
                - MSigDB_Hallmark_2020: Hallmark gene sets representing broad biological themes
                  (e.g., inflammation, metabolism).
                - GO_Molecular_Function_2015: GO category focusing on molecular functions
                  (e.g., enzyme activity, receptor binding).
                - GO_Cellular_Component_2015: GO category focusing on cellular locations
                  (e.g., nucleus, membrane, mitochondria).

            min_size (int, optional): The minimum number of genes required for a gene set to be included
                in the analysis.
                - Default is `5` (gene sets with fewer than 5 genes are excluded).

            max_size (int, optional): The maximum number of genes allowed in a gene set for it to be tested.
                - Default is `200` (gene sets with more than 200 genes are excluded to maintain specificity).

            p_value_threshold (float, optional): Defaults to 0.05.
                If there are not enough results, rerun with a higher threshold.
            fdr_threshold (float, optional): Defaults to 0.05.
                If there are not enough results, rerun with a higher threshold.

        Returns:
            - top_high_nes: Top 5 high NES results for each database.
            - top_low_nes: Top 5 low NES results for each database.
            - gsea_result_metadata: Storage entry for the saved GSEA results CSV.
            - gsea_plot_metadata: Storage entry for the saved GSEA plot.
        Notes:
            - The `hit_col` must contain actual gene symbols (not booleans).
            - Smaller `min_size` includes more pathways but can reduce reliability.
            - Larger `max_size` can dilute pathway specificity.
            - If `hit_col` contains numeric or non-symbol gene identifiers (e.g., Entrez or Ensembl),
            they will be automatically mapped to gene symbols before running GSEA.
        """

        # Validate/construct params
        params = _validate_params(
            {
                "csv_id": csv_id,
                "gene_sets": gene_sets,
                "hit_col": hit_col,
                "corr_col": corr_col,
                "min_size": min_size,
                "max_size": max_size,
                "p_value_threshold": p_value_threshold,
                "fdr_threshold": fdr_threshold,
            },
            GseaPipeInput,
            "gsea_pipe",
        )

        # Convert Pydantic models to DataFrame or load from CSV
        try:
            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide dataset or csv_id/csv_path")

            dataset_df = pd.read_csv(csv_abs_path)

            # Use the newer rank_gsea which auto-maps IDs and filters by p/q thresholds
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # start sending heartbeat
            print("FastMCP has emit?", hasattr(ctx, "emit"))
            heartbeat_task = start_keepalive(ctx, 5.0)

            sorted_results = await asyncio.get_running_loop().run_in_executor(
                CPU_POOL,
                functools.partial(
                    rank_gsea,
                    dataset=dataset_df,
                    gene_sets=params.gene_sets,
                    hit_col=params.hit_col,
                    corr_col=params.corr_col,
                    min_size=params.min_size,
                    max_size=params.max_size,
                    p_value_threshold=params.p_value_threshold,
                    fdr_threshold=params.fdr_threshold,
                    timestamp=timestamp,
                ),
            )

            # cancel heartbeat
            heartbeat_task.cancel()

            if isinstance(sorted_results, dict):
                # Safety: in case rank_gsea returns a dict unexpectedly
                sorted_results = pd.DataFrame(sorted_results)

            if sorted_results is None or len(sorted_results) == 0:
                logger.info("[gsea_pipe] No GSEA results after filtering")
                return {
                    "top_high_nes": {},
                    "top_low_nes": {},
                    "gsea_result_metadata": {},
                }

            # rank_gsea returns results sorted by NES ascending
            counts_by_database = {}
            top_high_nes = {}
            top_low_nes = {}
            if "Data Base" in sorted_results.columns:
                databases = sorted_results["Data Base"].unique()
                for database in databases:
                    sorted_results_db = sorted_results[
                        sorted_results["Data Base"] == database
                    ]
                    total_ups = int((sorted_results_db["NES"] > 0).sum())
                    total_downs = int((sorted_results_db["NES"] < 0).sum())
                    # Clip the leading genes to at most 5
                    up_genes = sorted_results_db.head(5)
                    up_genes["Lead_genes"] = up_genes["Lead_genes"].apply(
                        lambda x: x.split(",")
                    )
                    down_genes = sorted_results_db.tail(5)
                    down_genes["Lead_genes"] = down_genes["Lead_genes"].apply(
                        lambda x: x.split(",")
                    )
                    top_low_nes[database] = down_genes.to_dict(orient="records")
                    top_high_nes[database] = up_genes.to_dict(orient="records")
                    counts_by_database[database] = {
                        "up": total_ups,
                        "down": total_downs,
                        "total": total_ups + total_downs,
                    }
            else:
                total_ups = int((sorted_results["NES"] > 0).sum())
                total_downs = int((sorted_results["NES"] < 0).sum())
                up_genes = sorted_results.head(10)
                up_genes["Lead_genes"] = up_genes["Lead_genes"].apply(
                    lambda x: x.split(",")
                )
                down_genes = sorted_results.tail(10)
                down_genes["Lead_genes"] = down_genes["Lead_genes"].apply(
                    lambda x: x.split(",")
                )
                top_low_nes = down_genes.to_dict(orient="records")
                top_high_nes = up_genes.to_dict(orient="records")
                counts_by_database = {
                    "up": total_ups,
                    "down": total_downs,
                    "total": total_ups + total_downs,
                }

            # Register generated CSV and plot paths produced by rank_gsea
            _ensure_dir("output")
            csv_path = os.path.join("output", f"gsea_results_{timestamp}.csv")
            plot_path = os.path.join("output", f"gsea_results_{timestamp}.png")

            gsea_csv_metadata = None
            gsea_plot_metadata = None

            try:
                if os.path.exists(csv_path):
                    gsea_csv_metadata = _register_file(
                        csv_path, category="generated", to_s3=True
                    )
            except Exception:
                pass

            try:
                if os.path.exists(plot_path):
                    gsea_plot_metadata = _register_file(
                        plot_path, category="generated", to_s3=True
                    )
            except Exception:
                pass

            logger.info(
                (
                    f"[gsea_pipe] Results filtered by p<{params.p_value_threshold}, "
                    f"FDR<{params.fdr_threshold}: {len(sorted_results)}"
                )
            )

            return {
                "counts_by_database": counts_by_database,
                "top_high_nes": top_high_nes,
                "top_low_nes": top_low_nes,
                "gsea_result_metadata": gsea_csv_metadata,
                "gsea_plot_metadata": gsea_plot_metadata,
            }
        except Exception as e:
            context = {
                "dataset": params.csv_id,
                "hit_col": params.hit_col,
                "corr_col": params.corr_col,
                "gene_sets": params.gene_sets,
                "min_size": params.min_size,
                "max_size": params.max_size,
                "p_value_threshold": params.p_value_threshold,
                "fdr_threshold": params.fdr_threshold,
            }
            return _exception_payload("gsea_pipe", e, context)

    @mcp.tool
    async def ora_pipe(
        ctx: Context,
        csv_id: str,
        gene_col: str = "gene",
        gene_sets: Optional[List[str]] = None,
        p_value_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Performs Over Representation Analysis (ORA), a computational method
        used to determine whether a set of genes related to a biological function or pathway
        shows a consistent pattern of upregulation or downregulation between two conditions
        (e.g., healthy vs. diseased, treated vs. untreated). ORA helps identify pathways
        that are significantly enriched in your data.

        Note for LLM agents:
            - This tool returns only the top results to keep responses concise.
            - Use `csv_read`/`csv_filter`/`csv_select` to view additional rows on returned ora_result_metadata.

        Args:
            csv_id (str): ID of a stored CSV entry (required).
            gene_col (str): Column name in the CSV containing gene identifiers (required).
                Supports symbols and common identifier types (Ensembl/Entrez/RefSeq/UniProt)
                which will be auto-mapped to symbols. Defaults to "gene".
            gene_sets (Optional[List[str]]): Predefined gene set collections used for enrichment.
                Defaults to ["MSigDB_Hallmark_2020", "KEGG_2021_Human"].
            p_value_threshold (Optional[float]): Defaults to 0.2.
                Rerun with a higher threshold if there are not enough results.
        Returns:
            pd.DataFrame: A DataFrame containing the ORA results, including:
                - Term: The biological pathway or process being tested for enrichment.
                - Overlap: The number of genes in the gene set that are also in the dataset.
                - P-value: The statistical significance of the enrichment.
                - Adjusted P-value: Multiple-testing corrected p-value (FDR).
                - Odds Ratio: The ratio of the odds of the gene set being enriched
                  in the dataset to the odds of it not being enriched.
                - Combined Score: Combined strength metric of enrichment.
                - Genes: The overlapping genes contributing to the enrichment.
        """
        # Validate/construct params
        params = _validate_params(
            {
                "csv_id": csv_id,
                "gene_col": gene_col,
                "gene_sets": gene_sets,
                "p_value_threshold": p_value_threshold,
            },
            OraPipeInput,
            "ora_pipe",
        )
        try:
            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide csv_id")

            df_genes = pd.read_csv(csv_abs_path)
            gene_col = params.gene_col or "gene"
            if gene_col not in df_genes.columns:
                raise ValueError(f"Column '{gene_col}' not found in CSV")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # start sending heartbeat
            heartbeat_task = start_keepalive(ctx, 5.0)

            ora_results = await asyncio.get_running_loop().run_in_executor(
                CPU_POOL,
                functools.partial(
                    nrank_ora,
                    dataset=df_genes,
                    gene_sets=params.gene_sets,
                    gene_col=gene_col,
                    p_value_threshold=params.p_value_threshold,
                    timestamp=timestamp,
                ),
            )

            # cancel heartbeat
            heartbeat_task.cancel()

            if isinstance(ora_results, dict):
                ora_results = pd.DataFrame(ora_results)

            if ora_results is None or len(ora_results) == 0:
                logger.info("[ora_pipe] No ORA results after filtering")
                return {"ora_results": {}, "ora_result_metadata": {}}

            _ensure_dir("output")
            csv_path = os.path.join("output", f"ora_results_{timestamp}.csv")

            ora_result_metadata = None
            try:
                if os.path.exists(csv_path):
                    ora_result_metadata = _register_file(
                        csv_path, category="generated", to_s3=True
                    )
            except Exception:
                pass

            logger.info(f"[ora_pipe] Number of matching results: {len(ora_results)}")
            return {
                "ora_results": ora_results.to_dict(orient="records"),
                "ora_result_metadata": ora_result_metadata,
            }
        except Exception as e:
            context = {
                "gene_col": gene_col if "gene_col" in locals() else None,
                "gene_sets": params.gene_sets,
            }
            return _exception_payload("ora_pipe", e, context)

    @mcp.tool()
    async def enrich_rumma(
        genes: List[str],
        first: Optional[int] = None,
        max_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run enrichment using the Rummagene GraphQL API for a list of genes.

        Args:
            genes (List[str]): Gene symbols.
            first (Optional[int]): Page size per request. Defaults to 30.
            max_records (Optional[int]): Maximum total records to return. Defaults to 100.

        Returns:
            Dict[str, Any]: Tabular enrichment results.
        """
        # Validate/construct params
        params = _validate_params(
            {"genes": genes, "first": first, "max_records": max_records},
            EnrichRummaInput,
            "enrich_rumma",
        )

        df = enrich_query(
            params.genes, first=params.first, max_records=params.max_records
        )
        # Limit rows and truncate verbose columns
        df = _truncate_dataframe_columns(
            df, ["genes", "term", "source"], MAX_CELL_CHARS_DEFAULT
        )
        df = _limit_dataframe_rows(df, MAX_ROWS_DEFAULT)
        logger.info("🛠️[enrich_rumma] Enrichment fetched successfully")
        records = df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else []
        return {"results": records}

    @mcp.tool()
    async def query_string_rumma(
        term: str,
        retmax: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search PubMed by a free-text term and map articles to gene sets via Rummagene.

        Args:
            term (str): Search keyword/phrase.
            retmax (Optional[int]): Maximum PubMed Central IDs to retrieve. Defaults to 5000.

        Returns:
            Dict[str, Any]: Articles joined with mapped gene sets when available.
        """
        # Validate/construct params
        params = _validate_params(
            {"term": term, "retmax": retmax},
            QueryStringRummaInput,
            "query_string_rumma",
        )

        pmc_ids = rumm_search_pubmed(params.term, retmax=params.retmax)
        pmcs_with_prefix = ["PMC" + pmc for pmc in pmc_ids]
        articles = fetch_pmc_info(pmcs_with_prefix)
        if articles.empty or "pmcid" not in articles.columns:
            return {"results": []}
        sets_art = gene_sets_paper_query(articles["pmcid"].tolist())
        if not isinstance(sets_art, pd.DataFrame) or sets_art.empty:
            return {"results": articles.to_dict(orient="records")}
        sets_art = sets_art.rename(columns={"pmc": "pmcid"})
        merged = articles.merge(sets_art, how="left", on="pmcid")
        # Preserve original order
        merged = merged.set_index("pmcid").loc[articles["pmcid"]].reset_index()
        # Truncate long text fields and cap rows
        merged = _truncate_dataframe_columns(
            merged,
            columns=["title", "journal", "authors", "abstract", "gene_sets"],
            max_chars=MAX_CELL_CHARS_DEFAULT,
        )
        merged = _limit_dataframe_rows(merged, MAX_ROWS_DEFAULT)
        logger.info("🛠️[query_string_rumm] Query string fetched successfully")
        return {"results": merged.to_dict(orient="records")}

    @mcp.tool()
    async def query_table_rumma(term: str) -> Dict[str, Any]:
        """
        Search Rummagene's gene set tables by a term and extract PMCIDs.

        Args:
            term (str): Keyword for searching curated gene set tables.

        Returns:
            Dict[str, Any]: Matching rows with extracted "pmcid" where present.
        """
        # Validate/construct params
        params = _validate_params(
            {"term": term}, QueryTableRummaInput, "query_table_rumma"
        )

        df = table_search_query(params.term)
        if df.empty or "term" not in df.columns:
            return {"results": []}
        df["pmcid"] = df["term"].str.extract(r"(PMC\d+)")
        df["term"] = df["term"].str.replace(r"PMC\d+-?", "", regex=True)
        df = _truncate_dataframe_columns(df, ["term", "pmcid"], MAX_CELL_CHARS_DEFAULT)
        df = _limit_dataframe_rows(df, MAX_ROWS_DEFAULT)
        logger.info("🛠️[query_table_rumm] Query table fetched successfully")
        return {"results": df.to_dict(orient="records")}

    @mcp.tool()
    async def sets_info_rumm(gene_set_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed gene membership and annotations for a specific gene set.

        Args:
            gene_set_id (str): UUID of the gene set.

        Returns:
            Dict[str, Any]: Gene membership and annotations.
        """
        # Validate/construct params
        params = _validate_params(
            {"gene_set_id": gene_set_id}, SetsInfoRummInput, "sets_info_rumm"
        )

        df = genes_query(params.gene_set_id)
        df = _truncate_dataframe_columns(
            df, ["symbol", "name", "summary"], MAX_CELL_CHARS_DEFAULT
        )
        df = _limit_dataframe_rows(df, MAX_ROWS_DEFAULT)
        logger.info("🛠️[sets_info_rumm] Sets info fetched successfully")
        return {"results": df.to_dict(orient="records")}

    @mcp.tool()
    async def literature_trends(
        term: str,
        email: str,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Plot a PubMed publication timeline and return a year-to-IDs map and image path.

        Args:
            term (str): Keyword or phrase for PubMed timeline analysis.
            email (str): Contact email required by NCBI Entrez API.
            batch_size (Optional[int]): Entrez batch size. Defaults to 500.

        Returns:
            Dict[str, Any]: {"year_to_ids": dict, "image_path": str or None}.
        """
        # Validate/construct params
        params = _validate_params(
            {"term": term, "email": email, "batch_size": batch_size},
            LiteratureTrendsInput,
            "literature_trends",
        )

        year_to_ids, save_path = literature_timeline(
            term=params.term,
            email=params.email,
            batch_size=params.batch_size,
        )
        # Compress the year_to_ids to reduce token size: keep counts and sample up to N IDs per year
        compressed: Dict[str, Any] = {}
        try:
            for year, ids in (year_to_ids or {}).items():
                try:
                    ids_list = list(ids) if not isinstance(ids, list) else ids
                except Exception:
                    ids_list = []
                compressed[str(year)] = {
                    "count": len(ids_list),
                    "sample_ids": ids_list[:MAX_IDS_PER_YEAR],
                }
        except Exception:
            compressed = {}

        # Register the saved plot if present
        try:
            if save_path and os.path.exists(save_path):
                literature_plot_metadata = _register_file(
                    save_path, category="generated", to_s3=True
                )
        except Exception:
            pass
        logger.info("🛠️[literature_trends] Literature trends fetched successfully")
        return {
            "year_to_ids": compressed,
            "literature_plot_metadata": literature_plot_metadata,
        }

    @mcp.tool()
    async def prioritize_genes(
        gene_list: List[str],
        context_term: str,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prioritize genes in the context of a disease or phenotype.

        Args:
            gene_list (List[str]): Gene symbols to prioritize.
            context_term (str): Disease/phenotype context.
            email (Optional[str]): Contact email for PubMed queries. Optional.

        Returns:
            Dict[str, Any]: Ranked/prioritized genes as a table.
        """
        # Validate/construct params
        params = _validate_params(
            {"gene_list": gene_list, "context_term": context_term, "email": email},
            PrioritizeGenesInput,
            "prioritize_genes",
        )

        email = params.email or "test@example.com"
        gene_list_str = ", ".join(params.gene_list)
        df = prioritize_genes_fn(gene_list_str, params.context_term, email=email)
        df = _truncate_dataframe_columns(
            df, ["gene", "evidence", "pmids"], MAX_CELL_CHARS_DEFAULT
        )
        df = _limit_dataframe_rows(df, MAX_ROWS_DEFAULT)
        logger.info("🛠️[prioritize_genes] Prioritized genes fetched successfully")
        output_dict = df.to_dict(orient="records")
        if output_dict and len(output_dict) > 0:
            return {"results": output_dict}
        else:
            return {"results": []}

    @mcp.tool()
    async def gene_info(gene_list: List[str]) -> Dict[str, Any]:
        """
        Fetch gene annotations from MyGene.info for given gene symbols.

        Args:
            gene_list (List[str]): Gene symbols for annotation.

        Returns:
            Dict[str, Any]: Table of gene symbol, name, summary, aliases, etc.
        """
        # Validate/construct params
        params = _validate_params({"gene_list": gene_list}, GeneInfoInput, "gene_info")

        parsed = [g.strip() for g in params.gene_list if g.strip()]
        df = fetch_gene_annota(parsed)
        df = _truncate_dataframe_columns(
            df, ["symbol", "name", "summary", "aliases"], MAX_CELL_CHARS_DEFAULT
        )
        df = _limit_dataframe_rows(df, MAX_ROWS_DEFAULT)

        logger.info("🛠️[gene_info] Gene info fetched successfully")
        output_dict = df.to_dict(orient="records")

        if output_dict and len(output_dict) > 0:
            return {"results": output_dict}
        else:
            return {"results": []}

    # =========================
    # Local storage tools
    # =========================

    @mcp.tool()
    async def storage_list(
        category: Optional[str] = None,
        name_contains: Optional[str] = None,
        with_content: Optional[bool] = None,
        max_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List stored files on MCP server with optional filters and optional content preview.

        Args:
            category (Optional[str]): Logical category (e.g., generated, user_input).
            name_contains (Optional[str]): Substring match on filename.
            with_content (Optional[bool]): If true, include content up to max_bytes.
            max_bytes (Optional[int]): Max bytes to read when including content (default 1 MB).

        Returns:
            Dict[str, Any]: {"items": list} where each item is a stored file entry; content included if requested.
        """
        try:
            # Validate/construct params
            params = _validate_params(
                {
                    "category": category,
                    "name_contains": name_contains,
                    "with_content": with_content,
                    "max_bytes": max_bytes,
                },
                StorageListInput,
                "storage_list",
            )

            entries = _load_storage_index()
            # Keep entries as-is; minimal index no longer prunes by age
            filtered = _filter_entries(
                entries,
                category=params.category,
                name_contains=params.name_contains,
            )
            if params.with_content:
                enriched = [
                    _read_file_content(
                        e,
                        with_content=True,
                        max_bytes=int(
                            params.max_bytes
                            if params.max_bytes is not None
                            else STORAGE_MAX_BYTES_DEFAULT
                        ),
                    )
                    for e in filtered
                ]
                logger.info("🛠️[storage_list] %d enriched files found", len(enriched))
                return {"items": enriched}
            # Minimal fields only
            minimal = [
                {k: e.get(k) for k in ("id", "name", "path", "s3_presigned_url")}
                for e in filtered
            ]
            logger.info("🛠️[storage_list] %d files found", len(minimal))
            return {"items": minimal}
        except Exception as e:
            return _exception_payload("storage_list", e, {})

    @mcp.tool()
    async def storage_get(
        id: str,
        with_content: Optional[bool] = None,
        max_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve a stored file entry by id on MCP server; optionally include its content.

        Args:
            id (str): Storage entry id (required).
            with_content (Optional[bool]): If true, include content up to max_bytes.
            max_bytes (Optional[int]): Max bytes to read when including content (default 1 MB).

        Returns:
            Dict[str, Any]: Entry metadata and optional content.
        """
        try:
            # Validate/construct params
            params = _validate_params(
                {"id": id, "with_content": with_content, "max_bytes": max_bytes},
                StorageGetInput,
                "storage_get",
            )

            entries = _load_storage_index()
            entry = None
            if params.id:
                for e in entries:
                    if e.get("id") == params.id:
                        entry = e
                        break
                if not entry:
                    raise FileNotFoundError(f"No entry with id {params.id}")
            else:
                raise ValueError("Provide id")

            if params.with_content:
                logger.info("🛠️[storage_get] Reading content for %s", entry["path"])
                content = _read_file_content(
                    entry,
                    with_content=True,
                    max_bytes=int(
                        params.max_bytes
                        if params.max_bytes is not None
                        else STORAGE_MAX_BYTES_DEFAULT
                    ),
                )
                logger.info("🛠️[storage_get] Content read successfully")
                return content
            logger.info("🛠️[storage_get] Returning minimal entry for %s", entry["path"])
            return {k: entry.get(k) for k in ("id", "name", "path", "s3_presigned_url")}
        except Exception as e:
            return _exception_payload("storage_get", e, {"id": params.id})

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

            entry = _register_file(abs_path, category="user_input", to_s3=False)
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

            # Deterministic hash of filter parameters so multiple distinct filters on the
            # same source CSV produce distinct, discoverable filenames without metadata.
            cond_payload = [
                {
                    "column": c.column,
                    "op": c.op,
                    "value": c.value,
                }
                for c in (params.conditions or [])
            ]
            flavor_str = json.dumps(
                {
                    "conditions": cond_payload,
                    "keep": params.keep_columns,
                    "sort": params.sort_by,
                    "distinct": bool(params.distinct),
                },
                sort_keys=True,
            )
            flavor_hash = hashlib.md5(flavor_str.encode("utf-8")).hexdigest()[:10]
            out_name = (
                f"csv_filter_{params.csv_id}_{flavor_hash}.csv"
                if not params.name
                else f"{_slugify(params.name)}_{flavor_hash}.csv"
            )

            entry = _save_dataframe_csv(filtered, filename=out_name)

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

            # Deterministic hash of selection parameters (columns/rename/distinct/sort)
            select_str = json.dumps(
                {
                    "columns": params.columns,
                    "rename": params.rename,
                    "distinct": bool(params.distinct),
                    "sort": params.sort_by,
                },
                sort_keys=True,
            )
            select_hash = hashlib.md5(select_str.encode("utf-8")).hexdigest()[:10]
            out_name = (
                f"csv_select_{params.csv_id}_{select_hash}.csv"
                if not params.name
                else f"{_slugify(params.name)}_{select_hash}.csv"
            )

            entry = _save_dataframe_csv(out, filename=out_name)

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

            # Deterministic hash of join parameters (on/how/select/suffixes)
            join_str = json.dumps(
                {
                    "on": params.on,
                    "how": how,
                    "select": params.select,
                    "suffixes": suffixes,
                },
                sort_keys=True,
            )
            join_hash = hashlib.md5(join_str.encode("utf-8")).hexdigest()[:10]
            out_name = (
                f"csv_join_{params.left_csv_id}_{params.right_csv_id}_{join_hash}.csv"
                if not params.name
                else f"{_slugify(params.name)}_{join_hash}.csv"
            )

            entry = _save_dataframe_csv(merged, filename=out_name)

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

            # Deterministic hash of aggregation specs so different specs produce distinct files
            agg_str = json.dumps(
                {k: v for k, v in (params.aggregations or {}).items()}, sort_keys=True
            )
            agg_hash = hashlib.md5(agg_str.encode("utf-8")).hexdigest()[:10]
            out_name = (
                f"csv_aggregate_{params.csv_id}_{agg_hash}.csv"
                if not params.name
                else f"{_slugify(params.name)}_{agg_hash}.csv"
            )

            entry = _save_dataframe_csv(df, filename=out_name)

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
