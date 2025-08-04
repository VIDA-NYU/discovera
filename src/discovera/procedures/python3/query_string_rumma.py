from src.discovera.bkd.rummagene import search_pubmed, fetch_pmc_info, gene_sets_paper_query

# term = erythrocyte
pmc_ids = search_pubmed("{{ term }}", retmax=5000)
pmcs_with_prefix = ['PMC' + pmc for pmc in pmc_ids]
articles = fetch_pmc_info(pmcs_with_prefix)
sets_art= articles.merge(gene_sets_paper_query(articles["pmcid"].tolist()), how='left', on='pmcid')
sets_art = sets_art.set_index('pmcid').loc[articles['pmcid']].reset_index()
sets_art
