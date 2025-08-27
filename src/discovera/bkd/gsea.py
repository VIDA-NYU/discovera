import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# =========================
# Utility helpers
# =========================
def get_timestamp():
    """Generate a single timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_output_dir(path="output"):
    os.makedirs(path, exist_ok=True)
    return path

def save_with_timestamp(df, prefix, timestamp, ext="csv", folder="output"):
    """Save DataFrame with consistent timestamped filename."""
    ensure_output_dir(folder)
    filename = os.path.join(folder, f"{prefix}_{timestamp}.{ext}")
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    return filename

def pval_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''

# =========================
# Plotting
# =========================
def plot_gsea_results(
        df, 
        timestamp=get_timestamp(),
        nes_col='NES', 
        pval_col='FDR q-val', 
        pathway_col='Pathway', 
        db_col='Data Base',
        save=True,
        folder="output",
    ):
    """
    Plot GSEA results with NES values and significance stars for multiple databases.
    """
    df = df.copy()
    df['stars'] = df[pval_col].apply(pval_to_stars)

    databases = df[db_col].unique()
    n_db = len(databases)

    fig, axes = plt.subplots(n_db, 1, figsize=(8, 4 * n_db), sharex=True)
    if n_db == 1:
        axes = [axes]

    for ax, db in zip(axes, databases):
        df_db = df[df[db_col] == db].sort_values(nes_col)
        bars = ax.barh(
            df_db[pathway_col], 
            df_db[nes_col], 
            color=['#d73027' if x < 0 else '#4575b4' for x in df_db[nes_col]]
        )
        # Add stars
        for bar, star in zip(bars, df_db['stars']):
            if star:
                width = bar.get_width()
                ax.text(
                    width + 0.05 if width > 0 else width - 0.05,
                    bar.get_y() + bar.get_height()/2,
                    star,
                    va='center',
                    ha='left' if width > 0 else 'right',
                    fontsize=12,
                    fontweight='bold'
                )
        
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_ylabel('Pathway')
        ax.set_title(f'{db} Pathways')

    axes[-1].set_xlabel('Normalized Enrichment Score (NES)')
    plt.tight_layout()

    if save:
        ensure_output_dir(folder)
        outpath = os.path.join(folder, f"gsea_results_{timestamp}.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {outpath}")
    plt.show()

    return fig


# =========================
# Analysis functions
# =========================
def rank_gsea(
        dataset, 
        gene_sets,
        hit_col,
        corr_col,
        min_size, 
        max_size,
        threshold,
        timestamp=get_timestamp()
    ):
    data_sorted = dataset.sort_values(by=corr_col, ascending=False).reset_index(drop=True)
    rnk_data = data_sorted[[hit_col, corr_col]]

    # Run prerank analysis
    results = gp.prerank(
        rnk=rnk_data,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        outdir=None,
        verbose=True
    )
    results = pd.DataFrame(results.res2d)
    results[['Data Base', 'Pathway']] = results['Term'].str.split('__', expand=True)

    # Filter by threshold
    pval_columns = [col for col in results.columns if "p-val" in col or "q-val" in col]
    results = results[results[pval_columns].lt(threshold).any(axis=1)]

    save_with_timestamp(results, "gsea_results", timestamp)

    print(f"Number of matching results with p-val < {threshold}: {len(results)}")

    # Take top high/low NES
    sorted_results = results.sort_values(by="NES", ascending=True)
    top_low_nes = sorted_results.head(10)
    top_high_nes = sorted_results.tail(10)
    top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()

    plot_gsea_results(top_combined, timestamp)
    return top_combined


def nrank_ora(
        dataset,
        gene_sets,
        gene_col,
        organism,
        timestamp=get_timestamp()
    ):
    """Perform Enrichr enrichment analysis on a gene list."""
    genes = dataset[gene_col].dropna().astype(str).tolist()
    print(f"Using {len(genes)} genes")

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,
        verbose=True
    )

    results = pd.DataFrame(enr.results)
    save_with_timestamp(results, "ora_results", timestamp)
    print(f"Number of matching results: {len(results)}")

    return results.head(20)
