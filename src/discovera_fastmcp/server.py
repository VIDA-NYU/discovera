"""
Sample MCP Server for ChatGPT Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's chat and deep research features.
"""

import logging
import os
import json
import base64
import hashlib
import requests
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List

import gseapy as gp
import pandas as pd
import traceback
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
from tqdm import tqdm
from pydantic import ValidationError

from src.discovera_fastmcp.pydantic import (
    GseaPipeInput,
    QueryGenesInput,
    OraPipeInput,
    EnrichRummaInput,
    QueryStringRummaInput,
    QueryTableRummaInput,
    SetsInfoRummInput,
    LiteratureTrendsInput,
    PrioritizeGenesInput,
    GeneInfoInput,
    StorageListInput,
    StorageGetInput,
    CsvRecordInput,
    CsvReadInput,
)
from src.discovera.bkd.gsea import plot_gsea_results
from src.discovera.bkd.query_indra import nodes_batch, normalize_nodes
from src.discovera.bkd.rummagene import (
    enrich_query,
    search_pubmed as rumm_search_pubmed,
    fetch_pmc_info,
    gene_sets_paper_query,
    table_search_query,
    genes_query,
)
from src.discovera.bkd.pubmed import literature_timeline
from src.discovera.bkd.pubmed import prioritize_genes as prioritize_genes_fn
from src.discovera.bkd.mygeneinfo import fetch_gene_annota

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_ID = os.environ.get(
    "VECTOR_STORE_ID", "vs_68a75bf9ade88191b6f79599ee23b7f2"
)

# Initialize OpenAI client
openai_client = OpenAI()

server_instructions = """
You are an assistant MCP specializing in biomedical research, focusing on gene-disease relationships,
molecular mechanisms, and mutation-specific effects. Always consider tissue specificity and biological
context when interpreting data.


Evidence and confidence:

- Assign confidence scores (high, medium, low) for each result.
- Specify whether evidence is direct (experimental) or indirect (literature-based).
- Include PubMed IDs, database names, and dataset references wherever possible.
---

### Pipeline (follow strictly):

0. If user imput a csv file, use csv_record to register the file.
1. Summarize the user's research problem and ask clarifying questions if needed.
2. Request and load the dataset using csv_read.
3. Run analysis tools in order:
    a. gsea_pipe or ora_pipe (based on dataset type)
    b. enrich_rumma for top 30 genes by ranking metric and mention leading genes from the top pathway
    c. gene_info for **leading genes from the top pathway**
    d. query_genes for **leading genes from the top pathway**
    e. **Literature search**: query_string_rumma, query_table_rumma, literature_trends
4. Summarize results; include parameters, assumptions, and confidence.
5. Provide references including PubMed IDs and database links.
6. Generate a final report using Toulmin’s model (claims, grounds, warrants, qualifiers, rebuttals).
- Include all references and links.
- Pause to get user input where required, especially dataset, context, and library selection.

### Functions:

- **csv_record**: Register a CSV file in the local storage index.
- **csv_read**: Read a CSV file from the local storage index.
- **storage_list**: List all files in the local storage index.
- **storage_get**: Get a file from the local storage index.

- **query_genes**: Query Indra database for gene-gene relationships.
- **count_edges**: Aggregate and count interactions by specified groupings.
- **gsea_pipe**: Perform Gene Set Enrichment Analysis following the enrichment analysis guidelines.
- **ora_pipe**: Perform Over-Representation Analysis following the enrichment analysis guidelines.
- **gene_info**: Get gene information, meaning symbol, name, summary and aliases.
- **literature_trends**: Plots a timeline of PubMed articles related to a term, showing research trends.
- **prioritize_genes**: Prioritize genes based on a scoring function.
When interacting with the Rummagene API, you can access any or all of the following functions, depending on the input:
- If the input is a **gene list**, use:
    - **enrich_rumma**: Run Over Representation Analysis using Rummagene’s curated gene sets.
- If the input is text or a search term (e.g., disease name, phenotype, biological process), use in this order:
    - **query_string_rumma**: Search Rummagene for articles with matching gene sets.
    - **query_table_rumma**: Search Rummagene's curated gene set tables using keywords.
- For functional summaries and metadata of any gene set, use:
    - **sets_info_rumm**: Retrieve detailed descriptions and biological context.

---

### Function-specific instructions

- **gsea_pipe**:
    - Follow the enrichment analysis guidelines above.
    - Show the output in this format:

        | Pathway  | Enrichment Score | Normalized Enrichment Score |
        | Nominal p-value | FDR q-value | FWER p-value |
        | Leading Edge Gene % | Pathway Overlap % | Lead_genes |
        |---------|----------------:|----------------:|
        |---------------:|----------------:|---------------:|
        |-----------------:|----------------:|------------|

    - Display:
        - First the overall top 3 results
        - Then top 3 results per gene set library used, in the same format.
    - Summarize:
        - Number of total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column.
    - Show the results to the user after each function runs. Include all intermediate prints.

- **ora_pipe**:
    - Follow the enrichment analysis guidelines above.
    - Show the output in this format:

        | Gene Set  | Term | Overlap | P-value | Adjusted P-value | Odds Ratio | Combined Score | Genes |
        |-----------|-----|--------|--------:|----------------:|-----------:|---------------:|------|

    - Display first the overall top 3 results, then top 3 results per gene set library used, in the same format.
    - Mention:
        - Total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column.
    - Show the results to the user after each function runs.

---
"""


# =========================
# Local storage helpers
# =========================
STORAGE_INDEX_PATH = os.path.join("output", "storage_index.json")


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


def _is_text_ext(ext: str) -> bool:
    return ext.lower() in {".txt", ".csv", ".tsv", ".json", ".md", ".yaml", ".yml"}


def _register_file(
    path: str,
    category: str,
    origin_tool: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    abs_path = str(Path(path).resolve())
    p = Path(abs_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found to register: {abs_path}")

    # Build entry
    stat = p.stat()
    file_hash = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:8]
    created_at = datetime.now().isoformat()
    entry_id = f"{int(stat.st_mtime)}_{file_hash}"

    entry = {
        "id": entry_id,
        "path": abs_path,
        "name": p.name,
        "ext": p.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "created_at": created_at,
        "category": category,
        "origin_tool": origin_tool,
        "tags": tags or [],
        "metadata": metadata or {},
    }

    entries = _load_storage_index()
    # Replace if same path exists
    entries = [e for e in entries if e.get("path") != abs_path]
    entries.append(entry)
    _save_storage_index(entries)
    return entry


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
    def _match(e: dict) -> bool:
        if origin_tool and e.get("origin_tool") != origin_tool:
            return False
        if tag and tag not in (e.get("tags") or []):
            return False
        if ext and e.get("ext") != (
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        ):
            return False
        if category and e.get("category") != category:
            return False
        if name_contains and name_contains.lower() not in e.get("name", "").lower():
            return False
        if since:
            try:
                if e.get("created_at") < since:
                    return False
            except Exception:
                pass
        if until:
            try:
                if e.get("created_at") > until:
                    return False
            except Exception:
                pass
        return True

    return [e for e in entries if _match(e)]


def _read_file_content(entry: dict, with_content: bool, max_bytes: int) -> dict:
    result = {**entry}
    if not with_content:
        return result
    try:
        path = entry.get("path")
        if not path or not os.path.exists(path):
            result["content_text"] = None
            result["content_base64"] = None
            result["truncated"] = False
            return result
        size = os.path.getsize(path)
        truncated = size > max_bytes
        mode = "r" if _is_text_ext(entry.get("ext", "")) else "rb"
        if mode == "r":
            with open(path, mode, encoding="utf-8", errors="ignore") as f:
                data = f.read(max_bytes)
            result["content_text"] = data
            result["content_base64"] = None
        else:
            with open(path, mode) as f:
                data = f.read(max_bytes)
            result["content_text"] = None
            result["content_base64"] = base64.b64encode(data).decode("utf-8")
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
        raise FileNotFoundError(f"No entry with id {file_id}")
    return None


def create_server():
    mcp = FastMCP(
        name="discovera_fastmcp",
        instructions=server_instructions,
        stateless_http=True
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
                model_obj = model_cls(**params)
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

    @mcp.tool()
    async def query_genes(params: QueryGenesInput) -> Dict[str, Any]:
        """
        Queries the Indra database for relationships between the listed genes.

        Returns:
            pd.DataFrame: A dataframe containing the results of querying indra including this columns:
                - 'nodes', 'type', 'subj.name', 'obj.name', 'belief', 'text', 'text_refs.PMID',
                  'text_refs.DOI', 'text_refs.PMCID', 'text_refs.SOURCE', 'text_refs.READER', 'url'
        """
        # Validate/construct params
        params = _validate_params(params, QueryGenesInput, "query_genes")

        nodes = params.genes
        results = []

        nodes = normalize_nodes(nodes)

        nodes_lists = list(combinations(nodes, r=int(params.size)))

        with ThreadPoolExecutor() as executor:
            # Map the nodes_lists to the executor for parallel processing
            for statements in tqdm(
                executor.map(nodes_batch, nodes_lists),
                total=len(nodes_lists),
                desc="Processing nodes",
            ):
                if statements is not None:
                    results.append(statements)

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
            combined_df["url"] = "https://doi.org/" + combined_df["text_refs.DOI"]

            # TODO: check if this is the best way to handle this
            # Step 1: Pick one row with highest belief per unique 'type'
            type_representatives = combined_df.sort_values(
                "belief", ascending=False
            ).drop_duplicates(subset="type", keep="first")

            # Step 2: Exclude already selected rows
            remaining_df = combined_df.drop(type_representatives.index)

            # Step 3: Track used subj and obj names
            used_subj = set(type_representatives["subj.name"])
            used_obj = set(type_representatives["obj.name"])

            # Step 4: Define mask to prioritize diverse subj/obj
            remaining_df = remaining_df.assign(
                is_new_subj=~remaining_df["subj.name"].isin(used_subj),
                is_new_obj=~remaining_df["obj.name"].isin(used_obj),
            )

            # Step 5: Sort by new subj/obj and belief
            remaining_df = remaining_df.sort_values(
                by=["is_new_subj", "is_new_obj", "belief"],
                ascending=[False, False, False],
            )

            # Step 6: Select additional rows to make total 20
            additional_needed = 20 - len(type_representatives)
            additional_rows = remaining_df.head(additional_needed)

            # Combine and reset index
            final_df = pd.concat([type_representatives, additional_rows]).reset_index(
                drop=True
            )

            return final_df.to_dict()
        else:
            return {}

    @mcp.tool()
    async def count_edges(
        edges: List[Dict[str, Any]],
        grouping: str,
    ) -> Dict[str, Any]:
        """
        Counts and groups interactions found in the dataset based on the specified grouping type.

        Args:
            edges (List[Dict[str, Any]]): The DataFrame containing edge relationships.
            grouping (str): The type of grouping to apply. Defaults to detailed. Options are:
                - "summary": Groups only by 'nodes'.
                - "detailed": Groups by 'nodes', 'type', 'subj.name', and 'obj.name'.
                - "view": Groups by 'nodes' and 'type'.
        Returns:
            pd.DataFrame: A grouped DataFrame with a count column, sorted by count and type.

        Raises:
            ValueError: If an invalid group_type is provided.
        """
        # Define grouping options
        group_options = {
            "summary": ["nodes"],
            "detailed": ["nodes", "type", "subj.name", "obj.name"],
            "view": ["nodes", "type"],
        }

        # Convert JSON-like input to DataFrame
        edges = edges or []
        edges_df = pd.DataFrame(edges)

        # Validate group_type
        if grouping not in group_options:
            raise ValueError(
                f"Invalid group_type '{grouping}'. Choose from {list(group_options.keys())}."
            )

        group_columns = group_options[grouping]

        # Check if the provided group columns exist in the DataFrame
        if not set(group_columns).issubset(edges_df.columns):
            raise ValueError(
                f"The DataFrame does not contain the required columns for grouping: {group_columns}"
            )

        # Group by the specified columns and count the occurrences
        grouped_df = edges_df.groupby(group_columns).size().reset_index(name="count")

        return grouped_df.to_dict()

    @mcp.tool()
    async def gsea_pipe(
        params: GseaPipeInput,
    ) -> Dict[str, Any]:
        """
        Performs Gene Set Enrichment Analysis (GSEA), a computational method
        used to determine whether a set of genes related to a biological function or pathway
        shows a consistent pattern of upregulation or downregulation between two conditions
        (e.g., healthy vs. diseased, treated vs. untreated). GSEA helps identify pathways
        that are significantly enriched in the data.

        Returns:
            A dictionary containing the GSEA top 10 high NES and low NES results, and the full result metadata.
            Key attributes:
                * Term: The biological pathway or process being tested for enrichment.
                * Enrichment Score (ES): A measure of how strongly the genes in this pathway
                    are enriched in your ranked gene list.
                    - Higher ES = Stronger enrichment (greater association with your data).
                * Normalized Enrichment Score (NES): The ES adjusted for differences in gene set size,
                    making it easier to compare across different gene sets.
                    - Higher NES = More statistically significant enrichment.
                * NOM p-val (Nominal p-value): The statistical significance of the enrichment score
                    before adjusting for multiple comparisons.
                    - Lower p-value (<0.05) = More significant result.
                * FDR q-val (False Discovery Rate q-value): Adjusted p-value to control for false positives
                    when testing multiple pathways.
                    - Lower FDR (<0.25) = Higher confidence in the result.
                * FWER p-val (Family-Wise Error Rate p-value): A stricter correction for multiple testing,
                    reducing the chance of false positives.
                    - Lower FWER (<0.05) = More reliable result.
                * Tag %: The percentage of genes in the pathway that appear in the leading edge subset
                    (most enriched genes).
                    - Higher % = More genes driving enrichment.
                * Gene %: The percentage of genes from your ranked list that belong to this pathway.
                    - Higher % = More overlap with the pathway.
                * Lead_genes: The key genes contributing to the enrichment signal in the pathway.
        Notes:
            - The `hit_col` must contain actual gene symbols (not booleans).
            - A high NES and low FDR indicate strong and reliable enrichment.
            - Smaller `min_size` includes more pathways but can reduce reliability.
            - Larger `max_size` can dilute pathway specificity.
            - If `hit_col` contains numeric or non-symbol gene identifiers (e.g., Entrez or Ensembl),
            they will be automatically mapped to gene symbols before running GSEA.
        """

        # Validate/construct params
        params = _validate_params(params, GseaPipeInput, "gsea_pipe")

        # Convert Pydantic models to DataFrame or load from CSV
        try:
            dataset_df = None
            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide dataset or csv_id/csv_path")
            dataset_df = pd.read_csv(csv_abs_path)
            data_sorted = dataset_df.sort_values(by=params.corr_col, ascending=False)
            data_sorted.reset_index(drop=True, inplace=True)
            rnk_data = data_sorted[[params.hit_col, params.corr_col]]

            # Perform the prerank analysis
            results = gp.prerank(
                rnk=rnk_data,
                gene_sets=params.gene_sets,
                min_size=params.min_size,
                max_size=params.max_size,
                outdir=None,
                verbose=True,
            )
            results = pd.DataFrame(results.res2d)
            results[["Data Base", "Pathway"]] = results["Term"].str.split(
                "__", expand=True
            )
            _ensure_dir("output")
            # raw_path = os.path.join("output", "gsea_results_raw.csv")
            # results.to_csv(raw_path, index=False)
            # try:
            #     _register_file(
            #         raw_path,
            #         category="generated",
            #         origin_tool="gsea_pipe",
            #         tags=["gsea", "raw"],
            #         metadata={
            #             "hit_col": params.hit_col,
            #             "corr_col": params.corr_col,
            #         },
            #     )
            # except Exception:
            #     pass

            pval_columns = [
                col for col in results.columns if "p-val" in col or "q-val" in col
            ]
            results = results[results[pval_columns].lt(params.threshold).any(axis=1)]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/filtered_results_{timestamp}.csv"
            results.to_csv(filename, index=False)
            try:
                gsea_result_metadata = _register_file(
                    filename,
                    category="generated",
                    origin_tool="gsea_pipe",
                    tags=["gsea", "filtered"],
                    metadata={"threshold": params.threshold},
                )
            except Exception:
                pass
            logger.info(
                (
                    f"[gsea_pipe] Number of matching results with p-val smaller than "
                    f"{params.threshold}: {len(results)}"
                )
            )
            top_high_nes = results.sort_values(by="NES", ascending=False).head(10)
            top_low_nes = results.sort_values(by="NES", ascending=True).head(10)
            top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()

            plot_gsea_results(top_combined, timestamp)

            return {
                "top_high_nes": top_high_nes.to_dict(),
                "top_low_nes": top_low_nes.to_dict(),
                "gsea_result_metadata": gsea_result_metadata,
            }
        except Exception as e:
            context = {
                "dataset": params.csv_id,
                "hit_col": params.hit_col,
                "corr_col": params.corr_col,
                "gene_sets": params.gene_sets,
                "min_size": params.min_size,
                "max_size": params.max_size,
                "threshold": params.threshold,
            }
            return _exception_payload("gsea_pipe", e, context)

    @mcp.tool()
    async def ora_pipe(params: OraPipeInput) -> Dict[str, Any]:
        """
        Performs Over Representation Analysis (ORA), a computational method
        used to determine whether a set of genes related to a biological function or pathway
        shows a consistent pattern of upregulation or downregulation between two conditions
        (e.g., healthy vs. diseased, treated vs. untreated). ORA helps identify pathways
        that are significantly enriched in your data.

        Returns:
            pd.DataFrame: A DataFrame containing the ORA results, including,
                - Term: The biological pathway or process being tested for enrichment.
                - Overlap: The number of genes in the gene set that are also in the dataset.
                - P-value: The statistical significance of the enrichment score.
                - Adjusted P-value: The adjusted p-value to control for multiple testing.
                - Odds Ratio: The ratio of the odds of the gene set being enriched
                  in the dataset to the odds of it not being enriched.
                - Combined Score: The combined score of the gene set.
                - Genes: The genes in the gene set.
        """
        # Validate/construct params
        params = _validate_params(params, OraPipeInput, "ora_pipe")

        # Drop NA values in the gene column
        try:
            genes = []
            csv_abs_path = _resolve_path_from_id(params.csv_id)
            if not csv_abs_path:
                raise ValueError("Provide csv_id")
            df_genes = pd.read_csv(csv_abs_path)
            gene_col = params.gene_col or "gene"
            if gene_col not in df_genes.columns:
                raise ValueError(f"Column '{gene_col}' not found in CSV")
            genes = df_genes[gene_col].dropna().astype(str).tolist()
            # Run enrichment
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=params.gene_sets,
                organism="human",
                outdir=None,
                verbose=True,
            )
            results = pd.DataFrame(enr.results)
            print(f"Number of matching results: {len(results)}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/ora_results_{timestamp}.csv"
            results.to_csv(filename, index=False)
            try:
                ora_result_metadata = _register_file(
                    filename,
                    category="generated",
                    origin_tool="ora_pipe",
                    tags=["ora", "results"],
                    metadata={"timestamp": timestamp},
                )
            except Exception:
                pass
            print("ORA results saved to", filename)
            return {
                "ora_results": results.to_dict(),
                "ora_result_metadata": ora_result_metadata,
            }
        except Exception as e:
            context = {
                "num_genes": len(genes),
                "gene_sets": params.gene_sets,
            }
            return _exception_payload("ora_pipe", e, context)

    @mcp.tool()
    async def enrich_rumma(params: EnrichRummaInput) -> Dict[str, Any]:
        """
        Performs enrichment using the Rummagene GraphQL API for a list of genes.
        """
        # Validate/construct params
        params = _validate_params(params, EnrichRummaInput, "enrich_rumma")

        df = enrich_query(
            params.genes, first=params.first, max_records=params.max_records
        )
        logger.info("🛠️[enrich_rumma] Enrichment fetched successfully")
        return df.to_dict()

    @mcp.tool()
    async def query_string_rumma(params: QueryStringRummaInput) -> Dict[str, Any]:
        """
        Searches PubMed and maps articles to gene sets in Rummagene.
        """
        # Validate/construct params
        params = _validate_params(params, QueryStringRummaInput, "query_string_rumma")

        pmc_ids = rumm_search_pubmed(params.term, retmax=params.retmax)
        pmcs_with_prefix = ["PMC" + pmc for pmc in pmc_ids]
        articles = fetch_pmc_info(pmcs_with_prefix)
        if articles.empty or "pmcid" not in articles.columns:
            return {}
        sets_art = gene_sets_paper_query(articles["pmcid"].tolist())
        if not isinstance(sets_art, pd.DataFrame) or sets_art.empty:
            return articles.to_dict()
        sets_art = sets_art.rename(columns={"pmc": "pmcid"})
        merged = articles.merge(sets_art, how="left", on="pmcid")
        # Preserve original order
        merged = merged.set_index("pmcid").loc[articles["pmcid"]].reset_index()
        logger.info("🛠️[query_string_rumm] Query string fetched successfully")
        return merged.to_dict()

    @mcp.tool()
    async def query_table_rumma(params: QueryTableRummaInput) -> Dict[str, Any]:
        """
        Searches gene set tables in Rummagene by a term and extracts PMCIDs.
        """
        # Validate/construct params
        params = _validate_params(params, QueryTableRummaInput, "query_table_rumma")

        df = table_search_query(params.term)
        if df.empty or "term" not in df.columns:
            return {}
        df["pmcid"] = df["term"].str.extract(r"(PMC\d+)")
        df["term"] = df["term"].str.replace(r"PMC\d+-?", "", regex=True)
        logger.info("🛠️[query_table_rumm] Query table fetched successfully")
        return df.to_dict()

    @mcp.tool()
    async def sets_info_rumm(params: SetsInfoRummInput) -> Dict[str, Any]:
        """
        Retrieves detailed gene membership and annotations for a gene set.
        """
        # Validate/construct params
        params = _validate_params(params, SetsInfoRummInput, "sets_info_rumm")

        df = genes_query(params.gene_set_id)
        logger.info("🛠️[sets_info_rumm] Sets info fetched successfully")
        return df.to_dict()

    @mcp.tool()
    async def literature_trends(params: LiteratureTrendsInput) -> Dict[str, Any]:
        """
        Plots a publication timeline and returns year-to-IDs map and image path.
        """
        # Validate/construct params
        params = _validate_params(params, LiteratureTrendsInput, "literature_trends")

        year_to_ids, save_path = literature_timeline(
            term=params.term,
            email=params.email,
            batch_size=params.batch_size,
        )
        # Register the saved plot if present
        try:
            if save_path and os.path.exists(save_path):
                _register_file(
                    save_path,
                    category="generated",
                    origin_tool="literature_trends",
                    tags=["literature", "plot"],
                    metadata={"term": params.term},
                )
        except Exception:
            pass
        logger.info("🛠️[literature_trends] Literature trends fetched successfully")
        return {"year_to_ids": year_to_ids or {}, "image_path": save_path}

    @mcp.tool()
    async def prioritize_genes(params: PrioritizeGenesInput) -> Dict[str, Any]:
        """
        Prioritizes genes in the context of a disease or phenotype.
        """
        # Validate/construct params
        params = _validate_params(params, PrioritizeGenesInput, "prioritize_genes")

        email = params.email or "test@example.com"
        df = prioritize_genes_fn(params.gene_list, params.context_term, email=email)
        logger.info("🛠️[prioritize_genes] Prioritized genes fetched successfully")
        return df.to_dict()

    @mcp.tool()
    async def gene_info(params: GeneInfoInput) -> Dict[str, Any]:
        """
        Fetches gene annotations from MyGene.info for a list of gene symbols.
        """
        # Validate/construct params
        params = _validate_params(params, GeneInfoInput, "gene_info")

        parsed = [g.strip() for g in params.gene_list if g.strip()]
        df = fetch_gene_annota(parsed)

        logger.info("🛠️[gene_info] Gene info fetched successfully")
        return df.to_dict()

    # =========================
    # Local storage tools
    # =========================

    @mcp.tool()
    async def storage_list(params: StorageListInput) -> Dict[str, Any]:
        """
        List stored files with optional filters; optionally include content preview.
        """
        try:
            entries = _load_storage_index()
            filtered = _filter_entries(
                entries,
                origin_tool=params.origin_tool,
                tag=params.tag,
                ext=params.ext,
                category=params.category,
                name_contains=params.name_contains,
                since=params.since,
                until=params.until,
            )
            if params.with_content:
                enriched = [
                    _read_file_content(
                        e,
                        with_content=True,
                        max_bytes=int(params.max_bytes or 1048576),
                    )
                    for e in filtered
                ]
                logger.info("🛠️[storage_list] %d enriched files found", len(enriched))
                return {"items": enriched}
            logger.info("🛠️[storage_list] %d files found", len(filtered))
            return {"items": filtered}
        except Exception as e:
            return _exception_payload("storage_list", e, {})

    @mcp.tool()
    async def storage_get(params: StorageGetInput) -> Dict[str, Any]:
        """
        Retrieve a stored file entry by id or path; optionally include content.
        """
        try:
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
                    entry, with_content=True, max_bytes=int(params.max_bytes or 1048576)
                )
                logger.info("🛠️[storage_get] Content read successfully")
                return content
            logger.info("🛠️[storage_get] Returning minimal entry for %s", entry["path"])
            return entry
        except Exception as e:
            return _exception_payload("storage_get", e, {"id": params.id})

    @mcp.tool()
    async def csv_record(params: CsvRecordInput) -> Dict[str, Any]:
        """
        Record a user-provided CSV into output/user_csvs and register it.
        """
        try:
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
                try:
                    df = pd.read_csv(io.StringIO(text))
                except Exception as e:
                    raise ValueError(f"csv_text is not a valid CSV: {e}")
                normalized_csv_text = df.to_csv(index=False)

            elif source == "csv_base64":
                try:
                    raw = base64.b64decode(params.csv_base64)
                    text = raw.decode("utf-8")
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
                    df = pd.read_csv(abs_in_path)
                except Exception as e:
                    raise ValueError(f"Failed to read CSV from path: {e}")
                normalized_csv_text = df.to_csv(index=False)

            elif source == "csv_url":
                try:
                    response = requests.get(params.csv_url, timeout=20)
                    response.raise_for_status()
                    df = pd.read_csv(io.StringIO(response.text))
                except Exception as e:
                    raise ValueError(f"Failed to read CSV from URL: {e}")
                normalized_csv_text = df.to_csv(index=False)

            # Ensure output directory exists
            base_dir = os.path.join("output", "user_csvs")
            _ensure_dir(base_dir)

            # Build filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_input_{timestamp}.csv"
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
                origin_tool="csv_record",
                tags=tags,
                metadata=metadata,
            )
            logger.info("🛠️[csv_record] CSV file registered successfully: %s", abs_path)
            return entry
        except Exception as e:
            return _exception_payload("csv_record", e, {"name": params.name})

    @mcp.tool()
    async def csv_read(params: CsvReadInput) -> Dict[str, Any]:
        """
        Read a stored CSV by id or path and return a preview of rows.
        """
        try:
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

    return mcp


def main():
    """Main function to start the MCP server."""
    # Verify OpenAI client is initialized
    if not openai_client:
        logger.error(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
        raise ValueError("OpenAI API key is required")

    logger.info(f"Using vector store: {VECTOR_STORE_ID}")

    # Create the MCP server
    server = create_server()

    try:
        server.run(transport="http", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
