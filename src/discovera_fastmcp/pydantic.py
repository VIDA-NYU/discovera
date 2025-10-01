from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class GseaRow(BaseModel):
    hit: str = Field(
        description="""
The gene symbol to be tested for enrichment.
"""
    )
    corr: float = Field(
        description="""
The correlation or scoring value of the gene.
"""
    )
    containment: Optional[float] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class GseaPipeInput(BaseModel):
    csv_id: str = Field(
        description="""
ID of a stored CSV entry (required).
""",
    )
    gene_sets: Optional[List[str]] = Field(
        default=[
            "KEGG_2016",
            "GO_Biological_Process_2023",
            "Reactome_Pathways_2024",
            "MSigDB_Hallmark_2020",
        ],
        description="""
A list of predefined gene set collections used
for enrichment analysis. Defaults to
`"KEGG_2016", "GO_Biological_Process_2023",
"Reactome_Pathways_2024", "MSigDB_Hallmark_2020"`:
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
""",
    )
    hit_col: Optional[str] = Field(
        default="hit",
        description="""
The column name in the dataset that contains gene symbols
(e.g., "VWA2", "TSC22D4", etc.). These will be used as identifiers
to match against gene sets during enrichment.
""",
    )
    corr_col: Optional[str] = Field(
        default="corr",
        description="""
The column in the ranked gene list containing correlation or scoring values,
which are used to rank genes by association with a condition.
""",
    )
    min_size: Optional[int] = Field(
        default=5,
        description="""
The minimum number of genes required for a gene set to be included
in the analysis.
""",
    )
    max_size: Optional[int] = Field(
        default=200,
        description="""
The maximum number of genes allowed in a gene set.
""",
    )
    threshold: Optional[float] = 0.05


class QueryGenesInput(BaseModel):
    genes: List[str] = Field(
        description="""
A list of gene names to query in Indra.
**Do not exceed 10 genes at a time for performance reasons!**
"""
    )
    size: int = Field(
        default=2,
        description="""
The size of the gene combinations. Defaults to 2.
""",
    )


class OraPipeInput(BaseModel):
    csv_id: str = Field(
        description="""
ID of a stored CSV entry (required).
""",
    )
    gene_col: Optional[str] = Field(
        default="gene",
        description="""
Column name in the CSV containing gene symbols (used when csv_id/csv_path is provided).
""",
    )
    gene_sets: Optional[List[str]] = Field(
        default=["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
        description="""
A list of predefined gene set collections used
for enrichment analysis. Defaults to
`"MSigDB_Hallmark_2020", "KEGG_2021_Human"`:
- MSigDB_Hallmark_2020: Hallmark gene sets representing broad biological themes
(e.g., inflammation, metabolism).
- KEGG_2021_Human: Kyoto Encyclopedia of Genes and Genomes (metabolic and signaling pathways).
""",
    )


class EnrichRummaInput(BaseModel):
    genes: List[str] = Field(
        description="""
A list of gene symbols (e.g., ["STAT3", "CTNNB1"]).
"""
    )
    first: int = Field(
        default=30,
        description="""
Number of entries per page when querying enrichment results.
""",
    )
    max_records: int = Field(
        default=100,
        description="""
Maximum number of records to return across pages.
""",
    )


class QueryStringRummaInput(BaseModel):
    term: str = Field(
        description="""
Keyword or phrase to search PubMed and map to gene sets.
"""
    )
    retmax: int = Field(
        default=5000,
        description="""
Maximum number of PubMed Central IDs to retrieve.
""",
    )


class QueryTableRummaInput(BaseModel):
    term: str = Field(
        description="""
Keyword or phrase to search within gene set tables.
"""
    )


class SetsInfoRummInput(BaseModel):
    gene_set_id: str = Field(
        description="""
UUID of the gene set to retrieve gene membership and annotations for.
"""
    )


class LiteratureTrendsInput(BaseModel):
    term: str = Field(
        description="""
Keyword or phrase for PubMed timeline analysis.
"""
    )
    email: str = Field(
        description="""
Contact email required by NCBI Entrez API.
"""
    )
    batch_size: Optional[int] = Field(
        default=500,
        description="""
Number of records fetched per Entrez batch.
""",
    )


class PrioritizeGenesInput(BaseModel):
    gene_list: List[str] = Field(
        description="""
A list of gene symbols to prioritize.
"""
    )
    context_term: str = Field(
        description="""
Context term (e.g., disease or phenotype) used for prioritization.
"""
    )
    email: Optional[str] = Field(
        default=None,
        description="""
Optional contact email for PubMed queries.
""",
    )


class GeneInfoInput(BaseModel):
    gene_list: List[str] = Field(
        description="""
A list of gene symbols for annotation lookup.
"""
    )


class CountEdgesInput(BaseModel):
    edges: List[Dict[str, Any]] = Field(
        description="""
List of edge records (rows) to group and count.
Each item should include columns matching the chosen grouping.
"""
    )
    grouping: Optional[str] = Field(
        default="detailed",
        description="""
Grouping mode. One of: "summary" | "detailed" | "view".
Defaults to "detailed".
""",
    )


# =========================
# Storage and CSV tool models
# =========================

# class StorageSaveInput(BaseModel):
#     filename: Optional[str] = Field(default=None, description="""
# Desired filename including extension. If omitted, a timestamped name is generated.
# """)
#     subdir: Optional[str] = Field(default="storage", description="""
# Subdirectory under output/ where the file will be saved (e.g., storage, user_csvs).
# """)
#     category: Optional[str] = Field(default="generated", description="""
# Logical category for the file (e.g., generated, user_input, report).
# """)
#     origin_tool: Optional[str] = Field(default=None, description="""
# Name of the tool that produced this file (if any).
# """)
#     tags: Optional[List[str]] = Field(default=None, description="""
# List of tags to attach for filtering (e.g., gsea, plot, raw).
# """)
#     metadata: Optional[Dict[str, Any]] = Field(default=None, description="""
# Arbitrary metadata to store alongside the entry (parameters, context, etc.).
# """)
#     content_text: Optional[str] = Field(default=None, description="""
# Text content to save. Mutually exclusive with content_base64.
# """)
#     content_base64: Optional[str] = Field(default=None, description="""
# Base64-encoded binary content to save. Mutually exclusive with content_text.
# """)


class StorageListInput(BaseModel):
    origin_tool: Optional[str] = None
    tag: Optional[str] = None
    ext: Optional[str] = None
    category: Optional[str] = None
    name_contains: Optional[str] = None
    since: Optional[str] = Field(
        default=None,
        description="""
ISO timestamp; include files created at or after this time.
""",
    )
    until: Optional[str] = Field(
        default=None,
        description="""
ISO timestamp; include files created at or before this time.
""",
    )
    with_content: Optional[bool] = Field(
        default=False,
        description="""
If true, include content (text or base64) up to max_bytes in the response.
""",
    )
    max_bytes: Optional[int] = Field(
        default=1048576,
        description="""
Maximum number of bytes to read when including content.
""",
    )


class StorageGetInput(BaseModel):
    id: str = Field(
        description="""
ID of the stored entry (required).
""",
    )
    with_content: Optional[bool] = Field(
        default=True,
        description="""
If true, include content (text or base64) up to max_bytes in the response.
""",
    )
    max_bytes: Optional[int] = Field(default=1048576)


class CsvRecordInput(BaseModel):
    name: str = Field(
        description="""
Logical name for the CSV (used to derive filename). If omitted, timestamped name is used.
""",
    )
    csv_text: Optional[str] = Field(
        default=None,
        description="""
CSV content as text. Mutually exclusive with csv_base64.
""",
    )
    csv_base64: Optional[str] = Field(
        default=None,
        description="""
CSV content as base64. Mutually exclusive with csv_text.
""",
    )
    csv_path: Optional[str] = Field(
        default=None,
        description="""
Absolute path to a CSV file to read. If the file is too big but can be accessed by local read, use this field.
""",
    )
    csv_url: Optional[str] = Field(
        default=None,
        description="""
URL to a CSV file to read. If the file is too big but can be accessed by web read, use this field.
""",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="""
Tags to associate (e.g., user_input, dataset).
""",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
Arbitrary metadata to store (e.g., source, description).
""",
    )


class CsvReadInput(BaseModel):
    id: str = Field(
        description="""
ID of a stored CSV entry (required).
""",
    )
    n_rows: Optional[int] = Field(
        default=20,
        description="""
Number of rows to return from the top of the CSV (preview).
""",
    )


class CsvFilterCondition(BaseModel):
    column: str = Field(
        description="""
Column to filter on.
"""
    )
    op: str = Field(
        description="""
Operator: one of ==, !=, >, >=, <, <=, in, not_in, contains, not_contains, startswith, endswith, isnull, notnull.
"""
    )
    value: Optional[Any] = Field(
        default=None,
        description="""
Right-hand value for comparison; for in/not_in provide list.
""",
    )


class CsvFilterInput(BaseModel):
    csv_id: str = Field(
        description="""
ID of a stored CSV entry to filter (required).
"""
    )
    conditions: List[CsvFilterCondition] = Field(
        description="""
List of filter conditions combined with AND logic.
"""
    )
    keep_columns: Optional[List[str]] = Field(
        default=None,
        description="""
Optional subset of columns to keep in the output.
""",
    )
    sort_by: Optional[List[str]] = Field(
        default=None,
        description="""
Optional columns to sort by (prefix with '-' for descending).
""",
    )
    drop_duplicates: Optional[bool] = Field(
        default=False,
        description="""
Whether to drop duplicate rows after filtering.
""",
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class CsvIntersectInput(BaseModel):
    left_csv_id: str = Field(
        description="""
ID of the left CSV (required).
"""
    )
    right_csv_id: str = Field(
        description="""
ID of the right CSV (required).
"""
    )
    on: str = Field(
        description="""
Column name to intersect on (must exist in both CSVs).
"""
    )
    left_keep: Optional[List[str]] = Field(
        default=None,
        description="""
Optional columns to keep from left CSV (defaults to all).
""",
    )
    right_keep: Optional[List[str]] = Field(
        default=None,
        description="""
Optional columns to keep from right CSV (defaults to none).
""",
    )
    distinct: Optional[bool] = Field(
        default=True,
        description="""
Return distinct rows by the join key.
""",
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class CsvSelectInput(BaseModel):
    csv_id: str = Field(
        description="""
ID of a stored CSV entry (required).
"""
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="""
Columns to keep (in order). If omitted, keep all.
""",
    )
    rename: Optional[Dict[str, str]] = Field(
        default=None,
        description="""
Optional mapping of old_name -> new_name for renaming.
""",
    )
    distinct: Optional[bool] = Field(
        default=False,
        description="""
Whether to drop duplicate rows after selection.
""",
    )
    sort_by: Optional[List[str]] = Field(
        default=None,
        description="""
Optional columns to sort by (prefix with '-' for descending).
""",
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class CsvMergeAverageInput(BaseModel):
    left_csv_id: str = Field(
        description="""
ID of the left CSV (required).
"""
    )
    right_csv_id: str = Field(
        description="""
ID of the right CSV (required).
"""
    )
    on: str = Field(
        description="""
Join key column present in both CSVs (e.g., "gene").
"""
    )
    left_cols: Optional[List[str]] = Field(
        default=None,
        description="""
Columns to include from the left CSV (must include the join key). If omitted, include all.
""",
    )
    right_cols: Optional[List[str]] = Field(
        default=None,
        description="""
Columns to include from the right CSV (must include the join key). If omitted, include all.
""",
    )
    avg_map: Dict[str, List[str]] = Field(
        description="""
Mapping of output_column -> [left_column, right_column] to average, e.g.,
{"log2FoldChange": ["log2FoldChange_left", "log2FoldChange_right"],
 "padj": ["padj_left", "padj_right"]}.
"""
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class CsvJoinInput(BaseModel):
    left_csv_id: str = Field(
        description="""
ID of the left CSV (required).
"""
    )
    right_csv_id: str = Field(
        description="""
ID of the right CSV (required).
"""
    )
    on: str = Field(
        description="""
Join key column present in both CSVs (e.g., "gene").
"""
    )
    how: Optional[str] = Field(
        default="inner",
        description="""
Join type: 'inner' | 'left' | 'right' | 'outer'. Defaults to 'inner'.
""",
    )
    select: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="""
Optional selection mapping: {"left": [cols...], "right": [cols...]}. If omitted, use all columns.
""",
    )
    suffixes: Optional[List[str]] = Field(
        default=["_left", "_right"],
        description="""
Suffixes for overlapping column names, e.g., ["_left", "_right"].
""",
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class CsvAggregateInput(BaseModel):
    csv_id: str = Field(
        description="""
ID of a stored CSV entry (required).
"""
    )
    aggregations: Dict[str, Dict[str, Union[str, List[str]]]] = Field(
        description="""
Aggregation mapping of output_col -> {"func": "mean|sum|min|max|median|first|last", "cols": [colA, colB, ...]}.
E.g.: {"log2FoldChange": {"func": "mean", "cols": ["log2FoldChange_left", "log2FoldChange_right"]},
       "padj": {"func": "mean", "cols": ["padj_left", "padj_right"]}}
"""
    )
    name: Optional[str] = Field(
        default=None,
        description="""
Optional logical name for the output CSV; defaults to auto timestamp.
""",
    )


class RunDeseq2GseaInput(BaseModel):
    raw_counts_csv_id: str = Field(
        description="""
ID of stored CSV containing raw counts (genes × samples). One column contains gene IDs.
"""
    )
    sample_groups: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="""
Mapping of {group_name: [sample_id, ...]} used to construct sample metadata directly
from raw count column names (no separate metadata CSV required).
""",
    )
    gene_column: Optional[str] = Field(
        default="Unnamed: 0",
        description="""
Column name in raw counts holding gene identifiers.
""",
    )
    gene_sets: Optional[List[str]] = Field(
        default=[
            "KEGG_2016",
            "GO_Biological_Process_2023",
            "Reactome_Pathways_2024",
            "MSigDB_Hallmark_2020",
        ],
        description="""
Gene set libraries to use in downstream GSEA.
""",
    )
    threshold: Optional[float] = Field(
        default=0.05,
        description="""
Threshold GSEA p-value.
""",
    )
