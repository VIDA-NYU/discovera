import gseapy as gp


def rank_gsea(
        df, 
        gene_sets
        ):
    
    # Sort the correlation column by its absolute value
    hit_col="hit"
    corr_col="corr"
    top_n=50
    min_size=5 
    max_size=2000
    df['rank'] = df[corr_col].abs()
    df_sorted = df.sort_values(by='rank', ascending=False)

    df_sorted = df_sorted.drop(columns=['rank'])

    # Reset the index and drop the old index
    df_sorted.reset_index(drop=True, inplace=True)

    # Ensure the correct columns remain
    rnk_df = df_sorted[[hit_col, corr_col]]
    rnk_df[corr_col] = rnk_df[corr_col].astype(float)

    # Select top N rows
    rnk_df = rnk_df.head(top_n)

    # Perform the prerank analysis
    results = gp.prerank(
        rnk=rnk_df,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        outdir=None,
        verbose=True
    )
    
    return results.res2d
