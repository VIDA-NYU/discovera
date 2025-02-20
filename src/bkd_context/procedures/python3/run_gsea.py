import src.bkd_context.bkd.gsea

from src.bkd_context.bkd.gsea import rank_gsea

# gene_sets="GO_Biological_Process_2023"


gene_pair_edges = rank_gsea(df={{ dataset }}, gene_sets = {{ gene_sets }})
print(gene_pair_edges)

#df = pd.read_("gene_expression.xlsx")


# automating the reliability check,
# grounding find the related papers, using something
# rag, relevant references to try to coome up with trustworthy explanations.
# recomend which gene set to use!
# fact checking the infromation of the
# which way would be the best way to go, first query or the other way around


# reasoning or just repeating whta the llm read
# brand new papers, test whether is just repeating things that it has read
