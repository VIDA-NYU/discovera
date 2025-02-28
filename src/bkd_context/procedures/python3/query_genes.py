from src.bkd_context.bkd.query_indra import bulk_edges


edge_data = bulk_edges("{{ genes }}", size={{size}})
edge_data.to_markdown()
#edge_data.to_json()