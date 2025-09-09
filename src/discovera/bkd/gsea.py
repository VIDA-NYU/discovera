import gseapy as gp
import pandas as pd
import matplotlib.pyplot as plt
import os
import mygene
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime

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
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
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
    # plt.show()
    return fig
# =========================
# Analysis functions
# =========================


def detect_id_type(ids):
    """Quick heuristic — mainly for info, not strict enforcement."""
    sample = list(map(str, ids[:10]))
    if any(x.startswith(("ENSG", "ENSMUSG", "ENST", "ENSMUST", "ENSP", "ENSMUSP")) for x in sample):
        return "ensembl.gene"
    elif all(x.isdigit() for x in sample):
        return "entrezgene"
    elif any(x.startswith(("NM_", "NR_")) for x in sample):
        return "refseq"
    elif any(re.match(r"^[OPQ][0-9][A-Z0-9]{3,}", x) for x in sample):
        return "uniprot"
    else:
        return "symbol"


def map_to_symbol(df, gene_col):
    # strip Ensembl version suffix if present
    df[gene_col] = df[gene_col].astype(str).str.replace(r"\.\d+$", "", regex=True)
    ids = df[gene_col].dropna().astype(str).tolist()

    # Instead of detecting a single type, allow mixed IDs
    possible_scopes = ["ensembl.gene", "entrezgene", "refseq", "uniprot", "symbol"]

    # Deduplicate to reduce API load
    unique_ids = list(pd.unique(ids))
    if not unique_ids:
        return df.assign(symbol=pd.NA)

    # Chunking and parallel querying
    chunk_size = int(os.environ.get("MYGENE_CHUNK_SIZE", "1000"))
    max_workers = int(os.environ.get("MYGENE_WORKERS", "4"))
    pause_between_chunks_s = float(os.environ.get("MYGENE_THROTTLE_SECONDS", "0.0"))

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    mg = mygene.MyGeneInfo()

    def fetch_chunk(id_chunk):
        try:
            res = mg.querymany(
                id_chunk,
                scopes=possible_scopes,
                fields="symbol",
                as_dataframe=True,
            )
            # Ensure DataFrame with 'query' as a column
            res = res.reset_index() if hasattr(res, "reset_index") else pd.DataFrame()
            return res
        except Exception:
            return pd.DataFrame(columns=["query", "symbol"])  # fail-soft

    results = []
    id_chunks = list(chunks(unique_ids, chunk_size))

    # Use a small pool to avoid rate limits; throttle between completions if requested
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(fetch_chunk, c): idx for idx, c in enumerate(id_chunks)}
        for future in as_completed(future_to_idx):
            df_chunk = future.result()
            if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                results.append(df_chunk)
            if pause_between_chunks_s > 0:
                time.sleep(pause_between_chunks_s)

    if results:
        mapping = pd.concat(results, ignore_index=True)
    else:
        mapping = pd.DataFrame(columns=["query", "symbol"])

    # keep only useful mappings
    if not mapping.empty:
        mapping = mapping[["query", "symbol"]].dropna().drop_duplicates()

    # merge back with df
    df_merged = df.merge(mapping, left_on=gene_col, right_on="query", how="left")

    return df_merged


def rank_gsea(dataset, gene_sets, hit_col, corr_col,
              min_size=5, max_size=200, threshold=0.05, timestamp=None):
    if timestamp is None:
        timestamp = get_timestamp()

    # --- Auto-detect and map IDs if not symbols ---
    sample = dataset[hit_col].dropna().astype(str).tolist()
    id_type = detect_id_type(sample)
    if id_type != "symbol":
        print(f"Converting {id_type} IDs to gene symbols...")
        dataset = map_to_symbol(dataset, gene_col=hit_col)
        hit_col = "symbol"  # mapped column

    data_sorted = dataset.sort_values(by=corr_col, ascending=False).reset_index(drop=True)
    rnk_data = data_sorted[[hit_col, corr_col]].dropna()
    # Set index to hit_col
    rnk_data.set_index(hit_col, inplace=True)

    try:
        results = gp.prerank(
            rnk=rnk_data,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            outdir=None,
            verbose=True
        )
    except Exception as e:
        print(f"GSEA prerank failed: {e}")
        return pd.DataFrame()

    results_df = pd.DataFrame(results.res2d)
    if results_df.empty:
        print("GSEA returned no results.")
        return results_df

    results_df[['Data Base', 'Pathway']] = results_df['Term'].str.split('__', expand=True)
    pval_columns = [col for col in results_df.columns if "p-val" in col or "q-val" in col]
    results_df = results_df[results_df[pval_columns].lt(threshold).any(axis=1)]

    save_with_timestamp(results_df, "gsea_results", timestamp)
    print(f"Number of matching results with p-val < {threshold}: {len(results_df)}")

    sorted_results = results_df.sort_values(by="NES", ascending=True)
    top_low_nes = sorted_results.head(10)
    top_high_nes = sorted_results.tail(10)
    top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()

    plot_gsea_results(top_combined, timestamp)
    return sorted_results


def nrank_ora(
    dataset,
    gene_sets,
    gene_col,
    # organism="human",
    timestamp=None
):
    """
    Perform Enrichr enrichment analysis on a gene list.
    Automatically maps Ensembl/Entrez/RefSeq/UniProt IDs to symbols.
    """
    if timestamp is None:
        timestamp = get_timestamp()

    # --- Auto-detect and map IDs ---
    sample_ids = dataset[gene_col].dropna().astype(str).tolist()
    id_type = detect_id_type(sample_ids)
    if id_type != "symbol":
        print(f"Converting {id_type} IDs to gene symbols...")
        dataset = map_to_symbol(dataset, gene_col=gene_col)
        gene_col = "symbol"  # mapped column
    else:
        print("Detected input as gene symbols, no conversion needed.")

    # --- Build gene list ---
    genes = dataset[gene_col].dropna().astype(str).unique().tolist()
    print(f"Using {len(genes)} unique genes for ORA")

    if len(genes) == 0:
        print("No valid genes to run ORA.")
        return pd.DataFrame()

    # --- Run Enrichr --
    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_sets,
            # organism=organism,
            outdir=None,
            verbose=True
        )
    except Exception as e:
        print(f"ORA (Enrichr) failed: {e}")
        return pd.DataFrame()

    results = pd.DataFrame(enr.results)
    if results.empty:
        print("Enrichr returned no results.")
        return results

    results = results[results["Adjusted P-value"] < 0.05]
    results = results.sort_values(by="Combined Score", ascending=False)
    save_with_timestamp(results, "ora_results", timestamp)
    print(f"Number of matching results: {len(results)}")

    return results

