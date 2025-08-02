from src.discovera.bkd.rummagene import enrich_query

# genes = ["CTNNB1", "STAT3"]


enrichment = enrich_query("{{ genes }}", first={{ first }}, max_records={{ max_records }})
enrichment