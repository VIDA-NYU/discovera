import src.bkd_context.bkd.gsea

from src.bkd_context.bkd.gsea import rank_gsea


leading_genes = rank_gsea({{ dataset }}, gene_sets={{ gene_sets }}, hit_col="{{ hit_col }}", corr_col="{{ corr_col }}", min_size={{ min_size }}, max_size={{ max_size }})
leading_genes.to_markdown()


# automating the reliability check,
# grounding find the related papers, using something
# rag, relevant references to try to coome up with trustworthy explanations.
# recomend which gene set to use!
# fact checking the infromation of the
# which way would be the best way to go, first query or the other way around


# reasoning or just repeating whta the llm read
# brand new papers, test whether is just repeating things that it has read
# load the file gene_expression.csv as a dataframe
       # The source dataset is a DataFrame
       # For this step you should load gene_expression.csv into dataframe and use this dataset and gene_sets equal to GO_Biological_Process_2023 to run the following code. print in idle.

       #≈