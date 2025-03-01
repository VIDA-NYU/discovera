from archytas.tool_utils import AgentRef, LoopControllerRef, is_tool, tool, toolset
from beaker_kernel.lib.agent import BeakerAgent
from beaker_kernel.lib.context import BeakerContext


class BKDAgent(BeakerAgent):
    """
    An agent that will help biomedical researchers discover knowledge about genes and proteins
    """
    
    @tool()
    async def query_genes(self, genes: str, size: int, agent: AgentRef) -> str:
        """
        Queries the Indra database for relationships between the listed genes.

        Args:
            genes (str): A comma-separated string of gene names to query in Indra.
            size (int): The size of the gene combinations. Defaults to 2.

        Returns:
            pd.DataFrame(): containing the following columns: 
                - 'nodes', 'type', 'subj.name', 'obj.name', 'belief', 'text', 'text_refs.PMID',
                  'text_refs.DOI', 'text_refs.PMCID', 'text_refs.SOURCE', 'text_refs.READER', 'url'         
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
    async def count_edges(self, dataset: str, grouping: str, agent: AgentRef) -> str:
        """
        Counts and groups interactions found in the dataset based on the specified grouping type.

        Args:
            dataset (str): The name of the dataset to analyze.
            grouping (str): The type of grouping to apply. Defaults to detailed. Options are:
                - "summary": Groups only by 'nodes'.
                - "detailed": Groups by 'nodes', 'type', 'subj.name', and 'obj.name'.
                - "view": Groups by 'nodes' and 'type'.
        Returns:
            pd.DataFrame: A DataFrame containing the grouped interaction counts.
        """
        
        code = agent.context.get_code(
            "count_edges",
            {
                "dataset": dataset,
                "grouping": grouping,
            },
        )

        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")

        return result
    
    
    @tool()
    async def gsea_pipe(
        self, 
        dataset: str,  # This is the variable name, not the actual data
        gene_sets: str,
        hit_col: str,
        corr_col: str,
        min_size: int,
        max_size: int,
        threshold: int,
        agent: AgentRef
    ) -> str:
        """
        Performs Gene Set Enrichment Analysis (GSEA), a computational method 
        used to determine whether a set of genes related to a biological function or pathway 
        shows a consistent pattern of upregulation or downregulation between two conditions 
        (e.g., healthy vs. diseased, treated vs. untreated). GSEA helps identify pathways 
        that are significantly enriched in your data.

        Args:
            dataset (str): The name of the dataset variable stored in the agent.
            gene_sets (list, optional): A list of predefined gene set collections used for enrichment analysis.  
                Defaults to `["KEGG_2016", "GO_Biological_Process_2023", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020", "GO_Molecular_Function_2015", "GO_Cellular_Component_2015"]`:
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

            hit_col (str, optional): The column in the ranked gene list that indicates whether a gene  
                is part of a gene set. Default is `"hit"`.  

            corr_col (str, optional): The column in the ranked gene list containing correlation values,  
                which are used to rank genes based on their association with a condition. Default is `"corr"`.  

            min_size (int, optional): The minimum number of genes required for a gene set to be included
                in the analysis.  
                - Default is `5` (gene sets with fewer than 5 genes are excluded).  

            max_size (int, optional): The maximum number of genes allowed in a gene set for it to be tested.  
                - Default is `2000` (gene sets with more than 2000 genes are excluded to maintain specificity). 

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
            - Smaller `min_size` values allow more gene sets but may include unreliable results.  
            - Larger `max_size` values include broader pathways but may be too general.  
            - Look for pathways with high NES and low FDR q-values (< 0.25, ideally < 0.05) 
              for biologically significant results. 
        """

        # Generate the code execution context
        code = agent.context.get_code(
            "gsea_pipe",
            {
                "dataset": dataset, 
                "gene_sets": gene_sets,
                "hit_col": hit_col,
                "corr_col": corr_col,
                "min_size": min_size,
                "max_size": max_size,
                "threshold": threshold
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        
        return result 

