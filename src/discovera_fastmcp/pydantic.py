from typing import List, Optional
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
    dataset: List[GseaRow] = Field(
        description="""
The dataset to analyze in pandas DataFrame dict format, e.g.
[
    {
        "hit": "VWA2",
        "corr": 0.657806,
        "containment": 1.0,
        "score": 77.0,
        "rerank_score": 1.0
    },
    {
        "hit": "TSC22D4",
        "corr": -0.40405,
        "containment": 1.0,
        "score": 67.0,
        "rerank_score": 2.0
    },
    ...
]
"""
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
        default=50,
        description="""
The maximum number of genes allowed in a gene set.
""",
    )
    threshold: Optional[float] = 0.05


class QueryGenesInput(BaseModel):
    genes: List[str] = Field(
        description="""
A list of gene names to query in Indra.
"""
    )
    size: int = Field(
        description="""
The size of the gene combinations. Defaults to 2.
"""
    )


class OraPipeInput(BaseModel):
    genes: List[str] = Field(
        description="""
A list of genes to be tested for enrichment.
e.g. ["VWA2", "TSC22D4", ...]
"""
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
