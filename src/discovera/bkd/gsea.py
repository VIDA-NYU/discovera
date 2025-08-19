import gseapy as gp
import pandas as pd

from datetime import datetime

def rank_gsea(
        dataset, 
        gene_sets,
        hit_col,
        corr_col,
        min_size, 
        max_size,
        threshold
        ):

    data_sorted = dataset.sort_values(by=corr_col, ascending=False)

    # Reset the index and drop the old index
    data_sorted.reset_index(drop=True, inplace=True)

    # Ensure the correct columns remain
    rnk_data = data_sorted[[hit_col, corr_col]]

    # Perform the prerank analysis
    results = gp.prerank(
        rnk=rnk_data,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        outdir=None,
        verbose=True
    )
    results = pd.DataFrame(results.res2d)
    # Automatically select all columns containing 'p-val'
    pval_columns = [col for col in results.columns if "p-val" in col or "q-val" in col]
    # Filter rows where any 'p-val' column is greater than the threshold
    results = results[results[pval_columns].lt(threshold).any(axis=1)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"filtered_results_{timestamp}.csv"
    # Save to CSV
    results.to_csv(filename, index=False)
    print(f"Number of matching results with p-val smaller than {threshold}: {len(results)}")
    # Display the filtered DataFrame
    top_high_nes = results.sort_values(by="NES", ascending=False).head(10)

    # Select top N lowest NES
    top_low_nes = results.sort_values(by="NES", ascending=True).head(10)

    # Combine the two sets
    top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()
    return top_combined

 # TODO: check how to handle this    


def nrank_ora(
        dataset,
        gene_sets,
        gene_col,
        organism='human'
    ):
    """
    Perform Enrichr enrichment analysis on a gene list from a DataFrame column.

    Parameters:
    - dataset (pd.DataFrame): Input data containing a column with gene symbols.
    - gene_sets (str or list): Enrichr gene set libraries (e.g., 'KEGG_2016', 'GO_Biological_Process_2021').
    - gene_col (str): Column name in dataset containing gene symbols.
    - organism (str): Organism name, default is 'human'.
    - threshold (float): Significance threshold for p-value/q-value filtering.

    Returns:
    - pd.DataFrame: Filtered enrichment results.
    """
    # Drop NA values in the gene column
    genes = dataset[gene_col].dropna().astype(str).tolist()
    print(genes)
    # Run enrichment
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        verbose=True
    )

    # Convert to DataFrame
    results = pd.DataFrame(enr.results)

    print(f"Number of matching results: {len(results)}")

    return results.head(20) # TODO: check how to handle this



