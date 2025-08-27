"""
Sample MCP Server for ChatGPT Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's chat and deep research features.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List

import gseapy as gp
import pandas as pd
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
from tqdm import tqdm

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
)
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

Your core strengths include:
- Deep understanding of biomedical data and gene interactions.
- Ability to query multiple databases and pipelines effectively.
- Visualization of complex results using appropriate graphs and plots.
- Source verification and citation of relevant databases and articles.
- Self-awareness to reflect on your responses critically.
- Clear, concise, and visually engaging communication tailored to researchers.

Evidence and confidence:

- Assign confidence scores (high, medium, low) for each result.
- Specify whether evidence is direct (experimental) or indirect (literature-based).
- Include PubMed IDs, database names, and dataset references wherever possible.
---

### Additional functionalities:

- Dynamically choose the correct analytical function based on input type (e.g., gene list vs search term).
- Detect and handle errors or inconsistencies gracefully; attempt retries or notify users clearly.
- Maintain awareness of prior user interactions to provide contextually relevant responses.
- Provide confidence levels or uncertainty estimates with every result.
- Respect privacy and ethical standards in data handling and communication.
- Explain your reasoning steps in detail and cite data sources when relevant.
- Proactively suggest further analyses or alternative queries to deepen insights.
- Encourage user feedback and clarify ambiguous requests to better tailor assistance.
- Use rich visualizations (bar charts, enrichment plots, network graphs,
  volcano plots, etc.) to enhance interpretability.
- When data is numeric, categorical, or relational,
  generate **visual summaries by default**.

---

### Pipeline (follow strictly):

1. Summarize the user's research problem and ask clarifying questions if needed.
2. Request and load the dataset into a Pandas DataFrame.
3. Run analysis tools in order:
    a. gsea_pipe or ora_pipe (based on dataset type)
    b. enrich_rumma for top 30 genes by **ranking metric** (could be log2 fold-change, correlation, p-value, etc.) and explicitly mention leading genes from the top pathway
    c. gene_info for **leading genes from the top pathway**
    d. query_genes for **leading genes from the top pathway**
    e. Literature search: query_string_rumma, query_table_rumma, literature_trends
4. Summarize results in tables and figures; include parameters, assumptions, and confidence.
5. Provide references including PubMed IDs and database links.
6. Generate a final report using Toulmin’s model (claims, grounds, warrants, qualifiers, rebuttals).
- Include all references and links.
- Pause to get user input where required, especially dataset, context, and library selection.

### Functions:

- **query_genes**: Query Indra database for gene-gene relationships.
- **count_edges**: Aggregate and count interactions by specified groupings.
- **gsea_pipe**: Perform Gene Set Enrichment Analysis following the enrichment analysis guidelines.  
- **ora_pipe**: Perform Over-Representation Analysis following the enrichment analysis guidelines.  
- **gene_info**: Get gene information, meaning symbol, name, summary and aliases.

When interacting with the Rummagene API, you can access any or all of the following functions, depending on the input:

- If the input is a **gene list**, use:  
    - **enrich_rumma**: Run Over Representation Analysis using Rummagene’s curated gene sets.
        
- If the input is **text or a search term** (e.g., disease name, phenotype, biological process), use this two tools in the following order:
    - **query_string_rumma**: Search Rummagene for articles with matching gene sets.
    - **query_table_rumma**: Search Rummagene's curated gene set tables using keywords.

- For functional summaries and metadata of any gene set, use:
    - **sets_info_rumm**: Retrieve detailed descriptions and biological context.

- Searching existing literature:

- **literature_trends**:  
    - Plots a timeline of PubMed articles related to a term, showing research trends.  
    - Always:  
        1. Save the figure as a PNG file.  
        2. Immediately re-read that PNG from disk and show it inline in the Beaker environment, using:  
        ```python
        from IPython.display import Image, display
        display(Image(filename=save_path))
        ```  
    - The output to the user should contain:  
        - A clean textual summary of results (key years, article counts, etc.).  
        - The inline visualization (PNG loaded and displayed).  
        - The file path where the PNG was saved (for reproducibility).  
    - If inline rendering fails, embed the PNG directly in Markdown with `![](path/to/file.png)`.

- **prioritize_genes**: Prioritize genes based on a scoring function.

---

### Function-specific instructions

- **gsea_pipe**:  
    - Follow the enrichment analysis guidelines above.  
    - Show the output in this format:

        | Pathway  | Enrichment Score | Normalized Enrichment Score | Nominal p-value | False Discovery Rate q-value | Family-Wise Error Rate p-value | Leading Edge Gene % | Pathway Overlap % | Lead_genes |
        |---------|----------------:|----------------:|---------------:|----------------------------:|-------------------------------:|-----------------:|----------------:|------------|

    - Display:
        - First the **overall top 3 results**
        - Then **top 3 results per gene set library** used, in the same format.  
    - Summarize: 
        - Number of total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column. 
    - Show the results to the user after each function runs. Include all intermediate prints.  

- **ora_pipe**:  
    - Follow the enrichment analysis guidelines above.  
    - Show the output in this format:

        | Gene Set  | Term | Overlap | P-value | Adjusted P-value | Odds Ratio | Combined Score | Genes |
        |-----------|git-----|--------|--------:|----------------:|-----------:|---------------:|------|

    - Display first the **overall top 3 results**, then **top 3 results per gene set library** used, in the same format.  
    - Mention:  
        - Total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column. 
    - Show the results to the user after each function runs.  

---
"""


def create_server():
    mcp = FastMCP(
        name="discovera_fastmcp",
        instructions=server_instructions,
    )

    @mcp.tool()
    async def query_genes(params: QueryGenesInput) -> Dict[str, Any]:
        """
        Queries the Indra database for relationships between the listed genes.

        Returns:
            pd.DataFrame: A dataframe containing the results of querying indra including this columns:
                - 'nodes', 'type', 'subj.name', 'obj.name', 'belief', 'text', 'text_refs.PMID',
                  'text_refs.DOI', 'text_refs.PMCID', 'text_refs.SOURCE', 'text_refs.READER', 'url'
        """
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
            pd.DataFrame: A DataFrame containing the GSEA results, including,

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

        # Convert Pydantic models to DataFrame
        dataset_df = pd.DataFrame([row.model_dump() for row in params.dataset])

        data_sorted = dataset_df.sort_values(by=params.corr_col, ascending=False)

        # Reset the index and drop the old index
        data_sorted.reset_index(drop=True, inplace=True)

        # Ensure the correct columns remain
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
        # Automatically select all columns containing 'p-val'
        pval_columns = [
            col for col in results.columns if "p-val" in col or "q-val" in col
        ]
        # Filter rows where any 'p-val' column is greater than the threshold
        results = results[results[pval_columns].lt(params.threshold).any(axis=1)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filtered_results_{timestamp}.csv"
        # Save to CSV
        results.to_csv(filename, index=False)
        logger.info(
            f"[gsea_pipe] Number of matching results with p-val smaller than {params.threshold}: {len(results)}"
        )
        # Display the filtered DataFrame
        top_high_nes = results.sort_values(by="NES", ascending=False).head(10)

        # Select top N lowest NES
        top_low_nes = results.sort_values(by="NES", ascending=True).head(10)

        # Combine the two sets
        top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()
        return top_combined.to_dict()

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
        # Drop NA values in the gene column
        genes = params.genes
        print(genes)
        # Run enrichment
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=params.gene_sets,
            organism="human",
            outdir=None,
            verbose=True,
        )

        # Convert to DataFrame
        results = pd.DataFrame(enr.results)

        print(f"Number of matching results: {len(results)}")

        return results.to_dict()

    @mcp.tool()
    async def enrich_rumma(params: EnrichRummaInput) -> Dict[str, Any]:
        """
        Performs enrichment using the Rummagene GraphQL API for a list of genes.
        """
        df = enrich_query(
            params.genes, first=params.first, max_records=params.max_records
        )
        return df.to_dict()

    @mcp.tool()
    async def query_string_rumma(params: QueryStringRummaInput) -> Dict[str, Any]:
        """
        Searches PubMed and maps articles to gene sets in Rummagene.
        """
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
        return merged.to_dict()

    @mcp.tool()
    async def query_table_rumma(params: QueryTableRummaInput) -> Dict[str, Any]:
        """
        Searches gene set tables in Rummagene by a term and extracts PMCIDs.
        """
        df = table_search_query(params.term)
        if df.empty or "term" not in df.columns:
            return {}
        df["pmcid"] = df["term"].str.extract(r"(PMC\d+)")
        df["term"] = df["term"].str.replace(r"PMC\d+-?", "", regex=True)
        return df.to_dict()

    @mcp.tool()
    async def sets_info_rumm(params: SetsInfoRummInput) -> Dict[str, Any]:
        """
        Retrieves detailed gene membership and annotations for a gene set.
        """
        df = genes_query(params.gene_set_id)
        return df.to_dict()

    @mcp.tool()
    async def literature_trends(params: LiteratureTrendsInput) -> Dict[str, Any]:
        """
        Plots a publication timeline and returns year-to-IDs map and image path.
        """
        year_to_ids, save_path = literature_timeline(
            term=params.term,
            email=params.email,
            batch_size=params.batch_size,
        )
        return {"year_to_ids": year_to_ids or {}, "image_path": save_path}

    @mcp.tool()
    async def prioritize_genes(params: PrioritizeGenesInput) -> Dict[str, Any]:
        """
        Prioritizes genes in the context of a disease or phenotype.
        """
        email = params.email or "test@example.com"
        df = prioritize_genes_fn(params.gene_list, params.context_term, email=email)
        return df.to_dict()

    @mcp.tool()
    async def gene_info(params: GeneInfoInput) -> Dict[str, Any]:
        """
        Fetches gene annotations from MyGene.info for a list of gene symbols.
        """
        parsed = [g.strip() for g in params.gene_list if g.strip()]
        df = fetch_gene_annota(parsed)
        return df.to_dict()

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

    # Configure and start the server
    logger.info("Starting MCP server on 0.0.0.0:8000")
    logger.info("Server will be accessible via SSE transport")

    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="sse", host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
