import gseapy as gp
import pandas as pd

def rank_gsea(
        dataset, 
        gene_sets,
        hit_col,
        corr_col,
        min_size, 
        max_size
        ):

    dataset['rank'] = dataset[corr_col].abs()
    data_sorted = dataset.sort_values(by='rank', ascending=False)

    data_sorted = data_sorted.drop(columns=[corr_col])

    # Reset the index and drop the old index
    data_sorted.reset_index(drop=True, inplace=True)

    # Ensure the correct columns remain
    rnk_data = data_sorted[[hit_col, 'rank']]

    # Perform the prerank analysis
    results = gp.prerank(
        rnk=rnk_data,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        outdir=None,
        verbose=True
    )
    
    return pd.DataFrame(results.res2d)
