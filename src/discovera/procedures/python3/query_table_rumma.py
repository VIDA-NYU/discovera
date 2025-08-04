from src.discovera.bkd.rummagene import table_search_query

# term = "CRISPR"

table_article = table_search_query("{{ term }}")
table_article['pmcid'] = table_article['term'].str.extract(r'(PMC\d+)')
table_article['term'] = table_article['term'].str.replace(r'PMC\d+-?', '', regex=True)
table_article