import src.bkd_context.bkd.query_indra
import pandas as pd
from src.bkd_context.bkd.query_indra import bulk_edges
# nodes_lists = [
#     ["CTNNB1", "CDH1"],
# ]
# nodes_lists
edges = bulk_edges([{{ gene_pair }}])
pd.DataFrame(edges)