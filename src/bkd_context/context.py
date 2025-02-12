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
            If you are asked to perform a Pandas operation you should NOT print the results by default.
            Instead, acknowledge that the operation has been performed. The user can print the head of the dataframe if they so choose.

            You have access to the following functions:
                - query_gene_pair: This function query the Indra database for relationships between a pair of genes.
            
            It is a good idea to show the user the result after each function runs.
            """.strip()