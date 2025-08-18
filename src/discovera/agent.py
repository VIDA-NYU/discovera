from archytas.tool_utils import AgentRef, LoopControllerRef, is_tool, tool, toolset
from beaker_kernel.lib.agent import BeakerAgent


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
            pd.DataFrame(): A dataframe containing the results of querying indra including this columns:
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
        that are significantly enriched in the data.

        Args:
            dataset (str): The name of the dataset variable stored in the agent.
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
    
    @tool()
    async def ora_pipe(
        self, 
        dataset: str,  # This is the variable name, not the actual data
        gene_sets: str, 
        gene_col: str,
        agent: AgentRef
    ) -> str:
        """
        Performs Over Representation Analysis (ORA), a computational method 
        used to determine whether a set of genes related to a biological function or pathway 
        shows a consistent pattern of upregulation or downregulation between two conditions 
        (e.g., healthy vs. diseased, treated vs. untreated). ORA helps identify pathways 
        that are significantly enriched in your data.

        Args:
            dataset (str): The name of the dataset variable stored in the agent.
            gene_sets (str, optional): A list of predefined gene set collections used for enrichment analysis.  
                Defaults to `"MSigDB_Hallmark_2020","KEGG_2021_Human"`
            gene_col (str, optional): The column in the dataset that contains the gene symbols. Default is `"gene"`.

        Returns:
            pd.DataFrame: A DataFrame containing the ORA results, including,
                - Term: The biological pathway or process being tested for enrichment.
                - Overlap: The number of genes in the gene set that are also in the dataset.
                - P-value: The statistical significance of the enrichment score.
                - Adjusted P-value: The adjusted p-value to control for multiple testing.
                - Odds Ratio: The ratio of the odds of the gene set being enriched in the dataset to the odds of it not being enriched.
                - Combined Score: The combined score of the gene set.
                - Genes: The genes in the gene set.
        """

        # Generate the code execution context
        code = agent.context.get_code(
            "ora_pipe",
            {
                "dataset": dataset, 
                "gene_sets": gene_sets,
                "gene_col": gene_col
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")


    @tool()
    async def enrich_rumma(
        self,
        genes: str,           # Comma-separated string of gene names
        first: int,
        max_records: int,
        agent: AgentRef
    ) -> str:
        """
        Performs Over Representation Analysis (ORA) using the Rummagene GraphQL API.

        This function uses a curated gene enrichment engine to find biological pathways,
        gene sets, and literature associations that are significantly enriched in the 
        input gene list. It automatically paginates through available results.

        Args:
            genes (str): A comma-separated string of gene symbols (e.g., "STAT3,CTNNB1").
                These will be parsed and submitted as a list to the enrichment engine.
            first (int): Defaults to 30
            max_records (int): Defaults to 100.

        Returns:
            str: A formatted string of the top enrichment results,
                including:
                    - Gene Set ID
                    - Term
                    - Description
                    - # Genes in Set
                    - p-value
                    - adj. p-value
                    - Odds Ratio
                    - # Overlap
                    - PubMed Title
                    - PMCID

                If no results are found or if an error occurs, a message will be returned instead.
        """

        # Generate the code execution context
        code = agent.context.get_code(
            "enrich_rumma",
            {
                "genes": genes,
                "first": first,
                "max_records": max_records,
 
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        
        return result 
        
    @tool()
    async def query_string_rumma(
        self,
        term: str,           # Search term to query PubMed and retrieve PMC articles
        agent: AgentRef
    ) -> str:
        """
        Searches PubMed for literature using a given search term and retrieves information corresponding to 
        gene sets found in articles matching the search term.

        This function performs the following steps:
        1. Uses the `search_pubmed` function to search PubMed with the provided term 
        and retrieve up to 5000 matching PMC IDs.
        2. Adds the 'PMC' prefix to each ID to conform to PMC article identifiers.
        3. Uses the `fetch_pmc_info` function to retrieve article metadata for the 
        identified PMC articles.
        4. Retrieves metadata corresponding to gene sets associated to articles.

        Args:
            term (str): A query string used to search PubMed.

        Returns:
            str: A serialized representation of a dataframe including:
                - pmcid (str): The PubMed Central ID of the article.
                - title (str): The title of the article.
                - yr (int): The publication year.
                - doi (str): The Digital Object Identifier.
                - id (str): Identifier of the gene set.
                - term (str): String in file
                - count (int): Number of genes in the gene set.

            If no results are found or an error occurs, an informative message is returned.
        """
        # Generate the code execution context
        code = agent.context.get_code(
            "query_string_rumma",
            {
                "term": term,
 
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        

    @tool()
    async def query_table_rumma(
        self,
        term: str,          
        agent: AgentRef
    ) -> str:
        """
        Searches gene set tables in the Rummagene knowledge base using a given search term,
        and extracts metadata including associated PMC article IDs.

        Args:
            term (str): A keyword or phrase to search within gene set tables.

        Returns:
            str: A serialized representation of a dataframe including:
                - id (str): Identifier of the gene set, not of the table.
                - term (str): Cleaned table name or title without PMC prefix.
                - nGeneIds (int): Number of genes in the gene set.
                - pmcid (str): Extracted PubMed Central ID (if available).

            If no results are found or an error occurs, an informative message is returned.
        """
        # Generate the code execution context
        code = agent.context.get_code(
            "query_table_rumma",
            {
                "term": term,
 
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        return result 


    @tool()
    async def sets_info_rumm(
        self,
        gene_set_id: str,          
        agent: AgentRef
    ) -> str:
        """
        Retrieves detailed information about a specific gene set, including all genes it contains,
        along with descriptions and functional summaries of each gene. Needs as input a string

        Args:
            gene_set_id (str): Identifier of the gene set to retrieve information for. If not provided, look into the dataset referenced, the value from the column containing gene id.
                Expects a valid UUID, typically used as an ID or reference key in databases or APIs to referred to gene sets. 

        Returns:
            str: A serialized representation of a dataframe containing:
                - symbol (str): Gene symbol.
                - ncbiGeneId (str): NCBI gene identifier.
                - description (str): Brief description of the gene.
                - summary (str): Summary of the gene's function.

        """
        # Generate the code execution context
        code = agent.context.get_code(
            "sets_info_rumm",
            {
                "gene_set_id": gene_set_id,
 
            },
        )

        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        return result
    
    @tool()
    async def literature_trends(
        self,
        term: str,
        email: str,
        agent: AgentRef
    ) -> str:
        """
        Plots a timeline of PubMed articles related to a term, showing research trends.

        Args:
            term (str): Keyword or phrase (e.g., "autophagy", "TP53", "CTNNB1 and DKK are upregulated in tumors").
            email (str): Email, required to query pubmed. If not provided, ask user kindly to provide it.


        Returns:
            str: Years and Ids.
        """

        # Generate the code execution context
        code = agent.context.get_code(
            "literature_trends",
            {
                "term": term,
                "email": email
 
            },
        )
        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )

        result = result.get("return")
        return result
    
    @tool()
    async def prioritize_genes(
        self,
        gene_list: str,
        context_term: str,
        agent: AgentRef
    ) -> str:
        """
        Prioritize genes in context of disease or phenotype.
        Args:
            gene_list (str): String of gene lists.
            context_term (str): Context term which can be disease
        Returns:
            str: Base64 image or markdown with year-wise publication chart.
        """
        # Generate the code execution context
        code = agent.context.get_code(
            "prioritize_genes",
            {
                "gene_list": gene_list,
                "context_term": context_term

            },
        )
        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )
        result = result.get("return")
        return result
    
    @tool()
    async def gene_info(
        self,
        gene_list: str,
        agent: AgentRef
    ) -> str:
        """
        Prioritize genes in context of disease or phenotype.
        Args:
            gene_list (str): String of gene lists.
        Returns:
            str: Dataframe with gene information
        """
        # Generate the code execution context
        code = agent.context.get_code(
            "gene_info",
            {
                "gene_list": gene_list,

            },
        )
        # Evaluate the code asynchronously
        result = await agent.context.evaluate(
            code,
            parent_header={},
        )
        result = result.get("return")
        return result