# bkd-example

First, add your OpenAI API key to the environment:

```
export OPENAI_API_KEY=your key goes here
```

Then use `docker compose` to build and run the BKDKit Beaker context:

```
docker compose build
docker compose up -d
```

Navigate to `localhost:8888` and select the `bkd_context`. You can experiment with the following script:

```
1.- Load Gene Expression Data
    - Load gene_expression.csv into Pandas DataFrame.
2.- Gene Set Enrichment Analysis (GSEA)
    - Perform GSEA using this dataset, with:
        - GO_Biological_Process_2023 as gene_sets
        - hit as hit_column
        - corr as corr_column
        - 5 as min_set
        - 2000 as max_set
3.- Identify Lead Genes from Top Pathway
    - From the most statistically significant pathway identified by the GSEA, extract the lead genes.
4.- Refine Gene List Based on Correlation
    - Identify top {20} most correlated genes in the original dataset. Remove any genes already present in the lead gene set. Create a refined list of candidate genes for further analysis
5.- Retrieve Documented Gene Relationships
    - Retrieve relationships between  ["CTNNB1", "GLCE"], and save it into dataframe
6.- Summarize Gene Pair Relationship Types and Frequencies   ["CTNNB1", "GLCE"]
    ["CTNNB1", "NOTUM"],
8 .- Construct Gene Network Graph
    - Build a network graph where:
        - Nodes = Genes
        - Edges = Documented relationships (weighted by frequency/strength)
7 .- Extract Gene Relationship Excerpts
    - Extract excerpt where documented evidence/relationships was mentioned between these genes
9.-  Contextualize Gene Relationships in Disease {(Endometrial Carcinoma)}. 
    - Analyze how these gene relationships might relate to prospective endometrial carcinoma.
    - Hypothesis Generation:
        - Integrate GSEA results, known pathway involvement, and extracted literature.
        - Use LLM-generated biological interpretations. Can you summarize these excerpts relationships in the context of {prospective endometrial carcinoma}? / Hyphotesis of how these excerpts relate to {prospective endometrial carcinoma} and the output of the GSEA anaylsis you ran before.
10.- Suggest future research directions. Based on GSEA findings, network analysis, and literature mining, suggest:
    - Novel hypotheses for experimental validation.
    - Potential drug targets or pathways for intervention.
    - Follow-up bioinformatics analyses (e.g., transcriptomic validation, single-cell analysis).

```

## Adding tools for the agent

Currently the agent only has one tool: `query_gene_pair`. This is defined in `src/bkd_context/agent.py`. Additional tools can easily be added by copying the template for the `query_gene_pair` tool. One thing to note is that `@tools` are managed by [Archytas](https://github.com/jataware/archytas). Archytas allows somewhat restricted argument types and does not allow direct passing of `pandas.DataFrame`. Instead, dataframes should be referenced by their variable names as a `str`. The actual code procedure that is executed (see `procedures/python3/query_gene_pair.py`) treats the arguments from the `@tool` as variable names; when they should actually _be strings_ they should be wrapped in quotes as in the `query_gene_pair.py` example. Procedures invoked by tools can have their arguments passed in using Jinja templating. For example:

```
query_gene_pair({{ dataset }}, target="{{ target }}", method="{{ method }}")
```

Here `{{ dataset }}` is the string name of a `pandas.DataFrame` and is interpreted as a variable, where as `"{{ target }}"` is treated as a string such as `"gdc"`.

## Prompt modification
There are two main places to edit the agent's prompt. In `src/bkd_context/context.py` the `auto_context` is a place to provide additional context. Currently the tools are enumerated here though this isn't strictly necessary. Additionally, prompt can be edited/managed in the `agent.py` `BKDAgent` docstring.

