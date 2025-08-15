from src.discovera.bkd.rummagene import table_search_query

# term = "CRISPR"

table_article = table_search_query("{{ term }}")
# If DataFrame is empty or missing 'term' column, stop
if table_article.empty or 'term' not in table_article.columns:
    print("No item matched")
else:
    table_article['pmcid'] = table_article['term'].str.extract(r'(PMC\d+)')
    table_article['term'] = table_article['term'].str.replace(r'PMC\d+-?', '', regex=True)
    table_article.to_markdown()