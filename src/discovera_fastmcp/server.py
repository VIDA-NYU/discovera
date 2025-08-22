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
from typing import Any, Dict, List, Optional

import gseapy as gp
import pandas as pd
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI
from tqdm import tqdm

from src.discovera.bkd.query_indra import nodes_batch, normalize_nodes

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
This MCP server helps biomedical researchers discover knowledge about genes and proteins.
"""


def create_server():
    mcp = FastMCP(
        name="discovera_fastmcp",
        instructions=server_instructions,
    )

    @mcp.tool()
    async def query_genes(genes: str, size: int) -> pd.DataFrame:
        """
        Queries the Indra database for relationships between the listed genes.

        Args:
            genes (str): A comma-separated string of gene names to query in Indra.
            size (int): The size of the gene combinations. Defaults to 2.

        Returns:
            pd.DataFrame: A dataframe containing the results of querying indra including this columns:
                - 'nodes', 'type', 'subj.name', 'obj.name', 'belief', 'text', 'text_refs.PMID',
                  'text_refs.DOI', 'text_refs.PMCID', 'text_refs.SOURCE', 'text_refs.READER', 'url'
        """
        nodes = genes.split(",")
        results = []

        nodes = normalize_nodes(nodes)

        nodes_lists = list(combinations(nodes, r=int(size)))

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

            return final_df
        else:
            return pd.DataFrame()

    @mcp.tool()
    async def count_edges(
        edges: List[Dict[str, Any]],
        grouping: str,
    ) -> pd.DataFrame:
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

        return grouped_df

    @mcp.tool()
    async def gsea_pipe(
        dataset: List[Dict[str, Any]],
        gene_sets: Optional[str] = None,
        hit_col: Optional[str] = None,
        corr_col: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        threshold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Performs Gene Set Enrichment Analysis (GSEA), a computational method
        used to determine whether a set of genes related to a biological function or pathway
        shows a consistent pattern of upregulation or downregulation between two conditions
        (e.g., healthy vs. diseased, treated vs. untreated). GSEA helps identify pathways
        that are significantly enriched in the data.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.

            gene_sets (str, optional): A list of predefined gene set collections used for enrichment analysis.
                Defaults to `"KEGG_2016", "GO_Biological_Process_2023", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"`:
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

            hit_col (str, optional): The column name in the dataset that contains gene symbols
                (e.g., "VWA2", "TSC22D4", etc.). These will be used as identifiers
                to match against gene sets during enrichment.

            hit_col (str, optional): The column name in the dataset that contains gene symbols
                (e.g., "VWA2", "TSC22D4", etc.). These will be used as identifiers
                to match against gene sets during enrichment.

            corr_col (str, optional): The column in the ranked gene list containing correlation or scoring values,
                which are used to rank genes by association with a condition.

            min_size (int, optional): The minimum number of genes required for a gene set to be included
                in the analysis.
                - Default is `5` (gene sets with fewer than 5 genes are excluded).

            max_size (int, optional): The maximum number of genes allowed in a gene set for it to be tested.
                - Default is `50` (gene sets with more than 50 genes are excluded to maintain specificity).

            threshold (int, optional): Defaults to 0.05.

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

        # Convert JSON-like input to DataFrame
        dataset_df = pd.DataFrame(dataset or [])

        data_sorted = dataset_df.sort_values(by=corr_col, ascending=False)

        # Reset the index and drop the old index
        data_sorted.reset_index(drop=True, inplace=True)

        # Ensure the correct columns remain
        rnk_data = data_sorted[[hit_col, corr_col]]

        # Perform the prerank analysis
        results = gp.prerank(
            rnk=rnk_data,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            outdir=None,
            verbose=True,
        )
        results = pd.DataFrame(results.res2d)
        # Automatically select all columns containing 'p-val'
        pval_columns = [
            col for col in results.columns if "p-val" in col or "q-val" in col
        ]
        # Filter rows where any 'p-val' column is greater than the threshold
        results = results[results[pval_columns].lt(threshold).any(axis=1)]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filtered_results_{timestamp}.csv"
        # Save to CSV
        results.to_csv(filename, index=False)
        logger.info(
            f"[gsea_pipe] Number of matching results with p-val smaller than {threshold}: {len(results)}"
        )
        # Display the filtered DataFrame
        top_high_nes = results.sort_values(by="NES", ascending=False).head(10)

        # Select top N lowest NES
        top_low_nes = results.sort_values(by="NES", ascending=True).head(10)

        # Combine the two sets
        top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()
        return top_combined

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
