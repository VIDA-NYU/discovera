from src.discovera.bkd.query_indra import bulk_edges

edge_data = bulk_edges(" {{ genes }} ", size={{ size }})
edge_data.to_markdown()