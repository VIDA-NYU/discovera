from archytas.tool_utils import AgentRef, LoopControllerRef, is_tool, tool, toolset
from beaker_kernel.lib.agent import BeakerAgent
from beaker_kernel.lib.context import BeakerContext
import pandas as pd
from typing import Optional

class BKDAgent(BeakerAgent):
    """
    An agent that will help discover knowledge about genes and protein lists.
    """

    @tool()
    async def query_gene_pair(self, gene_pair: list, agent: AgentRef) -> str:
        """
        Queries the Indra database for relationships between a pair of genes.

        This function retrieves known interactions between two specified genes from the Indra database.

        Args:
            gene_pair (list[str]): A list containing exactly two gene names to query, e.g., ["CTNNB1", "CDH1"].

        Returns:
            
        """

        code = agent.context.get_code(
            "query_gene_pair",
            {
                "gene_pair": gene_pair,
            },
        )
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")

        return result


    @tool()
    async def multi_hop_query(self, gene_pair: tuple, agent: AgentRef) -> str:
        """
        This function query the Indra database for indirect evidence documented between a pair of genes, transversing hops.
        The gene_pair is a tuple of two gene names to query for papers documented some sort of indirect relationship, e.g. ("CTNNB1", "CDH1").

        Args:
            gene_pair (tuple): A pair of gene names to query for indirect relationships.

        Returns:
            str: returns the matched columns

        You should show the user the result after this function runs.
        """

        code = agent.context.get_code(
            "multi_hop_query",
            {

                "gene_pair": gene_pair,
            },
        )
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")

        return result


    @tool()
    async def run_gsea(
        self, 
        dataset: str,  # This is the variable name, not the actual data
        gene_sets: str,
        hit_col: str,
        corr_col: str,
        min_size: int,
        max_size: int,
        agent: AgentRef
    ) -> str:
        """
        Performs Gene Set Enrichment Analysis (GSEA) using a dataset stored in the agent's memory.
        This dataset is usually gene expression data ranked. 
        Args:
            dataset (str): The name of the dataset variable stored in the agent.
            gene_sets (str): Multi-libraries names supported, separate each name by comma or input a list.
            hit_col (str): Column with name of genes.
            corr_col (str): Column with ranks.
            min_size (int): 
            max_size (int): 
        Returns:
            str: Analysis results.
        """
        # Set default gene sets if not provided
        gene_sets = ["KEGG_2016", "GO_Biological_Process_2023", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"]
        hit_col = "hit"
        corr_col = "corr"
        min_size = 5
        max_size = 2000
        #if not gene_sets:  
        #    gene_sets = default_gene_sets  # Use default values

        # Generate the code execution context
        code = agent.context.get_code(
            "run_gsea",
            {
                "dataset": dataset, 
                "gene_sets": gene_sets,
                "hit_col": hit_col,
                "corr_col": corr_col,
                "min_size": min_size,
                "max_size": max_size,
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        
        return result  # Return the processed results
