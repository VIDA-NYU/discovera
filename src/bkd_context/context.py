from typing import Dict, Any
from beaker_kernel.lib.context import BeakerContext

from .agent import BKDAgent

class BKDContext(BeakerContext):

    enabled_subkernels = ["python3"]

    SLUG = "bkd_context"

    def __init__(self, beaker_kernel: "BeakerKernel", config: Dict[str, Any]):
        super().__init__(beaker_kernel, BKDAgent, config)

    async def setup(self, context_info=None, parent_header=None):
        super().setup(context_info, parent_header)

    async def auto_context(self):
            return f"""
            You are an assistant helping biomedical researchers discover relationships between genes and diseases.
            You have access to the following functions:
                - query_gene_pair: This function queries the Indra database for relationships between a pair of genes.
                - multi_hop_query: This function queries the Indra database for indirect relationships between a pair of genes.
                - run_gsea: This function runs GSEA to identify statistically significant gene sets that are enriched in a given dataset.
            It is a good idea to show the user the result after each function runs.
            Once you have run `query_gene_pair`, please summarize it in a parragraph, referencing type of 
            documented relations, from most frequent to least frequent, also topics usually covered in the text.
             
            Also you should d you should take a look the output,
            you should show them to user in a markdown table with the following template:
                | Gene 1  | Gene 2   | Relationship                     | Count
                |-----------------|-----------------|---------------------------------|
                | < | |      |
                | ...             | ...             | ...                             |

                Print complete dataframe, with all the count of relationships and count total relationships.

            
            Once you have run `multi_hop_query` you should print the output in raw text
            Once you are runing `run_gsea` you ask the user to upload the gene_expression dataset and then use this dataset as your input dataset for to run `run_gsea` with this
            dataset and use GO_Biological_Process_2023 as gene_set

            """.strip()