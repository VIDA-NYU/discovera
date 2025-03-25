from bkd_context.bkd.processing import group_edges


edges_types = group_edges({{ dataset }}, grouping="{{ grouping }}" )
edges_types.to_markdown()