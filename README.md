# bkd-example

First, add your OpenAI API key to the environment:

```
export OPENAI_API_KEY=your key goes here
```

Then use `docker compose` to build and run the BDIKit Beaker context:

```
docker compose build
docker compose up -d
```

Navigate to `localhost:8888` and select the `bkd_context`. You can experiment with the following script:

```
1. Load the file dou.csv as a dataframe and subset it to the following columns: Country, Histologic_type, FIGO_stage, BMI, Age, Race, Ethnicity, Gender, Tumor_Focality, Tumor_Size_cm.
2. Please match this to the gdc schema using the two_phase method and check any results that don't look correct.
3. Can you show the top matches for Histologic_type?
```

## Adding tools for the agent

Currently the agent only has one tool: `query_gene_pair`. This is defined in `src/bkd_context/agent.py`. Additional tools can easily be added by copying the template for the `query_gene_pair` tool. One thing to note is that `@tools` are managed by [Archytas](https://github.com/jataware/archytas). Archytas allows somewhat restricted argument types and does not allow direct passing of `pandas.DataFrame`. Instead, dataframes should be referenced by their variable names as a `str`. The actual code procedure that is executed (see `procedures/python3/query_gene_pair.py`) treats the arguments from the `@tool` as variable names; when they should actually _be strings_ they should be wrapped in quotes as in the `query_gene_pair.py` example. Procedures invoked by tools can have their arguments passed in using Jinja templating. For example:

```
query_gene_pair({{ dataset }}, target="{{ target }}", method="{{ method }}")
```

Here `{{ dataset }}` is the string name of a `pandas.DataFrame` and is interpreted as a variable, where as `"{{ target }}"` is treated as a string such as `"gdc"`.

## Prompt modification
There are two main places to edit the agent's prompt. In `src/bkd_context/context.py` the `auto_context` is a place to provide additional context. Currently the tools are enumerated here though this isn't strictly necessary. Additionally, prompt can be edited/managed in the `agent.py` `BKDAgent` docstring.