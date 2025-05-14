import gseapy as gp
import pandas as pd

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

    # Display the filtered DataFrame
    return pd.DataFrame(results)
