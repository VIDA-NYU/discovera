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
        This function query the Indra database for relationships between a pair of genes.
        The gene_pair is a list of two gene names to query for relationships, e.g. ["CTNNB1", "CDH1"].

        Args:
            gene_pair (list): A pair of gene names to query for relationships.

        Returns:
            str: returns the matched columns

        You should show the user the result after this function runs.
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
