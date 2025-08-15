
from src.discovera.bkd.mygeneinfo import fetch_gene_annota

# Example usage
# gene_list = "TP53, BRCA1, EGFR, APOE"
genes = fetch_gene_annota("{{ gene_list }}")
genes.to_markdown()