import src.bkd_context.bkd.query_indra
import pandas as pd
from src.bkd_context.bkd.query_indra import bulk_edges
from itertools import combinations

# genes =  [
# ('NOTUM', 'VWA2'), ('NOTUM', 'CTNNB1'),
# ('NOTUM', 'VPS26C'),
# ('VWA2', 'CTNNB1'),
# ('VWA2', 'VPS26C'),
# ('CTNNB1', 'VPS26C')
# ]

edge_data = bulk_edges([{{ genes }}])
edge_data