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
    async def query_genes(self, genes: str, size: int, agent: AgentRef) -> str:
        """
        Queries the Indra database for relationships between the listed genes.

        Args:
            genes (str): A string containing the genes we want to query in Indra. This list is separated by commas.
            size (int): Size is the size of the combination. The default value is 2.

        Returns:
            pd.DataFrame(): containing the following columns: 'nodes', 'type', 'subj.name', 'obj.name', 'belief', 'text', 'text_refs.PMID', 'text_refs.DOI', 'text_refs.PMCID', 'text_refs.SOURCE', 'text_refs.READER'         
        """
        code = agent.context.get_code(
            "query_genes",
            {
                "genes": genes,
                "size": size
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
        
        return result 

