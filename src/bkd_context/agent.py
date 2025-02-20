from archytas.tool_utils import AgentRef, LoopControllerRef, is_tool, tool, toolset
from beaker_kernel.lib.agent import BeakerAgent
from beaker_kernel.lib.context import BeakerContext


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
    async def run_gsea(self, dataset: str, gene_sets: str, agent: AgentRef) -> str:
        """
        Run Gene Set Enrichment Analysis (GSEA).

        Args:
            dataset (str): The dataset to analyze.
            gene_sets (list): A list of gene sets.

        Returns:
            str: Analysis results.
        """
        code = agent.context.get_code(
            "run_gsea",
            {   "dataset": dataset,
                "gene_sets": gene_sets,
            },
        )
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")

        return result