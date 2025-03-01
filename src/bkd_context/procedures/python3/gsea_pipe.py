from src.bkd_context.bkd.gsea import rank_gsea


leading_genes = rank_gsea(
    {{ dataset }}, 
    gene_sets={{ gene_sets }},
    hit_col="{{ hit_col }}", 
    corr_col="{{ corr_col }}",
    min_size={{ min_size }}, 
    max_size={{ max_size }},
    threshold={{ threshold }}
    )
leading_genes.to_markdown()