from src.discovera.bkd.gsea import nrank_ora


enr = nrank_ora(
    {{ dataset }}, 
    gene_sets="{{ gene_sets }}",
    gene_col="{{ gene_col }}"
    )
    
enr.to_markdown()
