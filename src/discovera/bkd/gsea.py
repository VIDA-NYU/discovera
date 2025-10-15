import functools
import os
import pickle
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import blitzgsea as blitz
import gseapy as gp
import matplotlib.pyplot as plt
import mygene
import pandas as pd
from gseapy import GSEA
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from rapidfuzz import process

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
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# =========================
# Plotting
# =========================


def plot_gsea_results(
    df,
    timestamp=get_timestamp(),
    nes_col="NES",
    pval_col="FDR q-val",
    pathway_col="Pathway",
    db_col="Data Base",
    save=True,
    folder="output",
):
    """
    Plot GSEA results with NES values and significance stars for multiple databases.
    """
    df = df.copy()
    df["stars"] = df[pval_col].apply(pval_to_stars)

    databases = df[db_col].unique()
    n_db = len(databases)

    plt.ioff()
    fig, axes = plt.subplots(n_db, 1, figsize=(8, 4 * n_db), sharex=True)
    if n_db == 1:
        axes = [axes]

    for ax, db in zip(axes, databases):
        df_db = df[df[db_col] == db].sort_values(nes_col)
        bars = ax.barh(
            df_db[pathway_col],
            df_db[nes_col],
            color=["#d73027" if x < 0 else "#4575b4" for x in df_db[nes_col]],
        )
        # Add stars
        for bar, star in zip(bars, df_db["stars"]):
            if star:
                width = bar.get_width()
                ax.text(
                    width + 0.05 if width > 0 else width - 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    star,
                    va="center",
                    ha="left" if width > 0 else "right",
                    fontsize=12,
                    fontweight="bold",
                )
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_ylabel("Pathway")
        ax.set_title(f"{db} Pathways")

    axes[-1].set_xlabel("Normalized Enrichment Score (NES)")
    plt.tight_layout()

    if save:
        ensure_output_dir(folder)
        outpath = os.path.join(folder, f"gsea_results_{timestamp}.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {outpath}")

    plt.close()
    return fig


# =========================
# Analysis functions
# =========================


@functools.lru_cache(maxsize=1)
def get_valid_hgnc_symbols():
    """
    Download HGNC reference table and cache it for validation.
    """
    url = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
    hgnc_df = pd.read_csv(url, sep="\t", low_memory=False)
    return set(hgnc_df["symbol"].dropna().str.upper())


def detect_id_type(ids):
    """
    Heuristic for gene ID type detection.
    Returns one of:
      - "ensembl.gene.human"
      - "ensembl.transcript.human"
      - "ensembl.protein.human"
      - "ensembl.gene.mouse"
      - "ensembl.transcript.mouse"
      - "ensembl.protein.mouse"
      - "entrezgene"
      - "refseq.human"
      - "refseq.mouse"
      - "uniprot"
      - "symbol"
      - "mixed" (if inconsistent)
    """
    sample = list(map(str, ids[:20]))  # check first 20 IDs

    patterns = {
        "ensembl.gene.human": re.compile(r"^ENSG\d+"),
        "ensembl.transcript.human": re.compile(r"^ENST\d+"),
        "ensembl.protein.human": re.compile(r"^ENSP\d+"),
        "ensembl.gene.mouse": re.compile(r"^ENSMUSG\d+"),
        "ensembl.transcript.mouse": re.compile(r"^ENSMUST\d+"),
        "ensembl.protein.mouse": re.compile(r"^ENSMUSP\d+"),
        "entrezgene": re.compile(r"^\d+$"),
        "refseq.human": re.compile(r"^(NM_|NR_)\d+"),
        "refseq.mouse": re.compile(r"^(XM_|XR_)\d+"),
        "uniprot": re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3,}"),  # broad UniProt
    }

    matched_types = set()
    for x in sample:
        for id_type, regex in patterns.items():
            if regex.match(x):
                matched_types.add(id_type)
                break
        else:
            matched_types.add("symbol")  # fallback

    return matched_types.pop() if len(matched_types) == 1 else "mixed"


def map_to_symbol(df, gene_col, cache_file=None):
    """
    Map any gene IDs (mouse/human/Ensembl/Entrez/symbol) to human gene symbols.
    """
    df[gene_col] = df[gene_col].astype(str).str.replace(r"\.\d+$", "", regex=True)
    ids = df[gene_col].dropna().astype(str).tolist()
    if not ids:
        return df.assign(symbol=pd.NA)

    unique_ids = list(pd.unique(ids))
    possible_scopes = ["ensembl.gene", "entrezgene", "refseq", "uniprot", "symbol"]

    chunk_size = int(os.environ.get("MYGENE_CHUNK_SIZE", "2000"))
    max_workers = int(os.environ.get("MYGENE_WORKERS", "8"))
    pause_between_chunks_s = float(os.environ.get("MYGENE_THROTTLE_SECONDS", "0.1"))

    mg = mygene.MyGeneInfo()

    # Load cache if provided
    cache = {}
    if cache_file:
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
        except FileNotFoundError:
            pass

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def fetch_chunk(id_chunk):
        results = []
        id_type = detect_id_type(id_chunk)

        if "mouse" in id_type:
            species = "mouse"
            fields = "symbol,homologene"
        else:
            species = "human"
            fields = "symbol"

        try:
            res = mg.querymany(
                id_chunk,
                scopes=possible_scopes,
                fields=fields,
                species=species,
                as_dataframe=True,
            ).reset_index()

            for _, row in res.iterrows():
                human_symbol = None
                if species == "mouse":
                    if "homologene" in row and isinstance(row["homologene"], dict):
                        for taxid, entrez, symbol in row["homologene"].get("genes", []):
                            if taxid == 9606:
                                human_symbol = symbol
                                break
                if human_symbol is None:
                    human_symbol = row.get("symbol")
                results.append({"query": row["query"], "human_symbol": human_symbol})

        except Exception:
            for gid in id_chunk:
                results.append({"query": gid, "human_symbol": None})

        return pd.DataFrame(results)

    # Parallel querying
    results = []
    id_chunks = list(chunks(unique_ids, chunk_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(fetch_chunk, c): idx for idx, c in enumerate(id_chunks)
        }
        for future in as_completed(future_to_idx):
            df_chunk = future.result()
            if isinstance(df_chunk, pd.DataFrame) and not df_chunk.empty:
                results.append(df_chunk)
            if pause_between_chunks_s > 0:
                time.sleep(pause_between_chunks_s)
            print(f"[map_to_symbol] Processed {len(results)} chunks")

    # Combine results
    if results:
        mapping = pd.concat(results, ignore_index=True)
    else:
        mapping = pd.DataFrame(columns=["query", "human_symbol"])

    # Merge with cache
    if cache:
        cached_df = pd.DataFrame(
            [{"query": k, "human_symbol": v} for k, v in cache.items()]
        )
        mapping = pd.concat([mapping, cached_df], ignore_index=True).drop_duplicates(
            subset="query"
        )

    # Update cache
    if cache_file:
        cache_dict = dict(zip(mapping["query"], mapping["human_symbol"]))
        with open(cache_file, "wb") as f:
            pickle.dump(cache_dict, f)

    # Merge back with original DataFrame
    df_merged = df.merge(mapping, left_on=gene_col, right_on="query", how="left").drop(
        columns=["query"]
    )
    df_merged.rename(columns={"human_symbol": "symbol"}, inplace=True)
    df_merged["symbol"] = df_merged["symbol"].str.upper()

    return df_merged


def prepare_gene_symbols(df, gene_col):
    """
    Map to human gene symbols only if necessary; otherwise, keep original symbols.
    Filters against HGNC to keep valid human symbols.
    """
    valid_symbols = get_valid_hgnc_symbols()

    # Check if input is already valid human symbols
    input_ids = df[gene_col].dropna().astype(str).str.upper()
    n_valid = input_ids.isin(valid_symbols).sum()
    fraction_valid = n_valid / len(input_ids)
    if fraction_valid >= 0.9:  # if most IDs are valid human symbols, skip mapping
        df = df.copy()
        df["symbol"] = input_ids
        print(
            f"Skipping mapping: {n_valid}/{len(input_ids)} IDs are valid human symbols."
        )
    else:
        df = map_to_symbol(df, gene_col)
        print(f"Mapping performed: {len(df)} IDs mapped to human symbols.")

    # Filter against HGNC
    df = df[df["symbol"].isin(valid_symbols)].copy()
    df = df.drop(columns=[gene_col], errors="ignore")
    df[gene_col] = df["symbol"]
    return df


def fuzzy_rename_column(df, column, reference_values, threshold=50):
    """
    Rename values in a DataFrame column to closest matches in reference_values using fuzzy matching.

    Parameters:
    - df: DataFrame containing the column to rename
    - column: column name in df to rename
    - reference_values: list of strings to match against
    - threshold: minimum fuzzy match score to apply renaming

    Returns:
    - df with updated column values
    """
    new_values = []
    for val in df[column]:
        match, score, _ = process.extractOne(val, reference_values)
        if score >= threshold:
            new_values.append(match)
        else:
            new_values.append(val)  # keep original if no good match
    df[column] = new_values
    return df


def run_deseq2(
    raw_counts,
    sample_conditions,
    gene_column="Unnamed: 0",
    sample_column="SampleID",
    group_column="Group",
):
    """
    Full DESeq2 workflow (preprocessing step for downstream GSEA):

    This function performs preprocessing and differential expression analysis
    on RNA-seq count data using **DESeq2**, a widely used method for identifying
    differentially expressed genes. DESeq2 models raw counts using a negative
    binomial distribution and performs statistical testing while accounting
    for differences in sequencing depth and biological variability.

    The workflow generates a tidy results table suitable for downstream analyses,
    including **Gene Set Enrichment Analysis (GSEA)**.

    Parameters:
    - raw_counts: pandas DataFrame of raw counts (genes × samples)
        Expected format:
            - Rows: genes (e.g., Ensembl IDs or gene symbols)
            - Columns: sample names
            - One column (gene_column) contains gene identifiers
        Dataset after preprocessing:
            - `filter_counts`: genes × samples matrix of integer counts,
              retaining genes with at least one sample >10 reads
    - sample_conditions: pandas DataFrame of sample metadata
        Expected format:
            - Column sample_column: sample identifiers (may require fuzzy matching)
            - Column group_column: experimental group or condition
        Dataset after processing:
            - `sample_conditions`: samples × group labels, with sample IDs aligned
              to counts columns
    - gene_column: name of the column in raw_counts with gene identifiers
    - sample_column: name of the column in sample_conditions with sample IDs
    - group_column: name of the column in sample_conditions with group labels

    Workflow steps:
    1. Load counts data and filter low-expressed genes
       - Filtered counts are stored in `filter_counts` (genes × samples, integer counts)
    2. Load metadata and align/fuzzy-match sample IDs to counts columns
       - Metadata is stored in `sample_conditions` (samples × group labels)
    3. Build DESeq2 dataset (`dds`)
       - DESeq2 object combines counts and metadata
       - Performs size factor estimation (normalization), dispersion estimation,
         and prepares for statistical testing
    4. Automatically select reference and tested levels for the group factor
    5. Run DESeq2 statistical testing
       - Generates Wald statistics and p-values for differential expression

    Returns:
    - results_df: a **pandas DataFrame** containing DESeq2 results for each gene
        - One row per gene
        - Columns include:
            - 'GeneID': the gene identifier (originally from the counts file)
            - 'log2FoldChange': estimated log2 fold change of expression between
              the tested group and reference group
            - 'lfcSE': standard error of the log2 fold change
            - 'stat': Wald test statistic
            - 'pvalue': raw p-value for differential expression
            - 'padj': adjusted p-value (Benjamini-Hochberg FDR correction)
        - Can be **directly used for ranking genes** for GSEA or other enrichment analyses
        - Index is reset (0..n-1), so 'GeneID' is a regular column, not the index

    Note:
    - DESeq2 is a robust method for RNA-seq differential expression analysis
      that accounts for library size differences and biological variability.
    - This function is primarily a **preprocessing step** for GSEA.
      The output `results_df` can be ranked by log2 fold change or other statistics
      for enrichment analysis of gene sets.
    """

    # raw_counts:
    #   Rows: genes
    #   Columns: sample names (e.g., "3_1", "5_2")
    #   Values: raw read counts
    raw_counts = raw_counts.set_index(gene_column)
    raw_counts.index.name = None
    # Filter lowly-expressed genes
    filter_counts = raw_counts[(raw_counts > 10).sum(axis=1) > 0]

    # sample_conditions:
    #   Column sample_column: sample IDs (to be matched with counts columns)
    #   Column group_column: experimental group (e.g., "DMSO", "Cisplatin")
    sample_conditions = fuzzy_rename_column(
        sample_conditions, sample_column, list(filter_counts.columns)
    )
    sample_conditions = sample_conditions[[sample_column, group_column]].set_index(
        sample_column
    )
    sample_conditions[group_column] = pd.Categorical(sample_conditions[group_column])

    # --- Build DESeq2 dataset ---
    # Align metadata rows to counts rows (samples) exactly
    counts_t = filter_counts.T
    # Ensure metadata index order matches counts_t index
    sample_conditions = sample_conditions.loc[counts_t.index]
    dds = DeseqDataSet(
        counts=counts_t,  # Transpose: DESeq expects samples as rows
        metadata=sample_conditions,  # Metadata must match rows of counts
        design_factors=group_column,
    )

    # --- Run normalization and dispersion estimation ---
    dds.deseq2()

    # --- Automatically pick reference/tested levels ---
    group_levels = dds.obs[group_column].unique()
    reference_level = group_levels[0]
    tested_level = group_levels[1]

    print(f"Reference level: {reference_level}")
    print(f"Tested level: {tested_level}")

    # --- Run DESeq2 stats ---
    contrast = [group_column, tested_level, reference_level]
    print("Using contrast:", contrast)

    stat_res = DeseqStats(dds, contrast=contrast, inference=DefaultInference(n_cpus=4))
    stat_res.summary()

    results = stat_res.results_df.copy()
    results.index.name = "GeneID"
    results = results.reset_index()
    return results


def rank_gsea(
    dataset,
    gene_sets,
    hit_col,
    corr_col,
    min_size=5,
    max_size=200,
    threshold=0.05,
    timestamp=None,
):
    if timestamp is None:
        timestamp = get_timestamp()

    # Detect ID type
    id_type = detect_id_type(dataset[hit_col].dropna().astype(str).tolist())
    print(f"Detected ID type: {id_type}")

    # Map & filter
    dataset = prepare_gene_symbols(dataset, hit_col)
    if dataset.empty:
        warnings.warn("No valid human symbols found after mapping.")
        return pd.DataFrame()

    data_sorted = dataset.sort_values(by=corr_col, ascending=False).reset_index(
        drop=True
    )
    rnk_data = data_sorted[[hit_col, corr_col]].dropna()

    # Set index to hit_col (ONLY APPLICABLE FOR GSEApy)
    # rnk_data.set_index(hit_col, inplace=True)
    results_df = None
    try:

        for gene_set in gene_sets:
            lib = blitz.enrichr.get_library(gene_set)
            print(f"Running GSEA for {gene_set}...")
            res = blitz.gsea(
                signature=rnk_data,
                library=lib,
                min_size=min_size,
                max_size=max_size,
            )
            res_df = pd.DataFrame(res)
            res_df.reset_index(inplace=True)
            res_df.rename(
                columns={
                    "Term": "Pathway",
                    "es": "ES",
                    "nes": "NES",
                    "pval": "NOS p-val",
                    "fdr": "FDR q-val",
                    "geneset_size": "Size",
                    "leading_edge": "Lead_genes",
                },
                inplace=True,
            )
            res_df["Data Base"] = gene_set
            res_df = res_df[
                [
                    "Data Base",
                    "Pathway",
                    "NES",
                    "Size",
                    "ES",
                    "NOS p-val",
                    "FDR q-val",
                    "Lead_genes",
                ]
            ]
            if results_df is None:
                results_df = res_df
            else:
                results_df = pd.concat([results_df, res_df])

    except Exception as e:
        print(f"GSEA prerank failed: {e}")
        return pd.DataFrame()

    save_with_timestamp(results_df, "gsea_results", timestamp)
    print(f"Number of matching results with p-val < {threshold}: {len(results_df)}")

    sorted_results = results_df.sort_values(by="NES", ascending=False)
    top_low_nes = sorted_results.head(10)
    top_high_nes = sorted_results.tail(10)
    top_combined = pd.concat([top_high_nes, top_low_nes]).drop_duplicates()

    plot_gsea_results(top_combined, timestamp)
    return results_df


def classic_gsea(
    dataset,
    metadata,
    gene_sets,
    hit_col,
    group_column,
    pheno_pos,
    pheno_neg,
    permutation_type="gene_set",
    permutation_num=1000,
    method="signal_to_noise",
    threads=4,
    seed=8,
    threshold=0.05,
):
    """
    Run classic GSEA (Gene Set Enrichment Analysis) using raw expression data
    (not pre-ranked). This function maps gene identifiers, sets up the expression
    matrix, and performs enrichment analysis for the specified phenotypes.

    Parameters:
    - dataset: pandas DataFrame containing gene expression data
        Must include a column with gene identifiers (hit_col) and expression values.
    - metadata: pandas DataFrame containing sample metadata
        Must include a column specifying sample groups (group_column)
    - gene_sets: gene sets for enrichment (e.g., 'MSigDB_Hallmark_2020')
        Can be a string (for a predefined database) or a list of gene sets.
    - hit_col: column name in dataset containing gene symbols/IDs
    - group_column: column name in metadata specifying sample groups
    - pheno_pos: name of the positive phenotype (e.g., "Cisplatin_IC50")
    - pheno_neg: name of the negative phenotype (e.g., "DMSO")
    - permutation_type: 'gene_set' or 'phenotype' (default='gene_set')
    - permutation_num: number of permutations to perform (default=1000)
    - method: ranking method for genes, e.g., 'signal_to_noise' (default)
    - threads: number of CPU threads to use (default=4)
    - seed: random seed for reproducibility (default=8)

    Returns:
    - results_df: pandas DataFrame containing GSEA results
        - Each row corresponds to one enriched gene set.
        - Columns typically include:
            - 'Term': gene set name
            - 'NES': normalized enrichment score
            - 'pval': nominal p-value
            - 'fdr': FDR q-value
            - 'Data Base': gene set database/source
            - 'Pathway': gene set/pathway name
        - This DataFrame can be used for downstream analysis or visualization.
    """

    # Detect ID type
    id_type = detect_id_type(dataset[hit_col].dropna().astype(str).tolist())
    print(f"Detected ID type: {id_type}")

    # Map & prepare gene symbols
    dataset = prepare_gene_symbols(dataset, hit_col)
    if dataset.empty:
        print("No valid gene symbols found after mapping.")
        return pd.DataFrame()

    # Set index to processed gene symbol
    dataset = dataset.set_index("symbol").drop(hit_col, axis=1)

    # Extract expression matrix for GSEA
    norm_counts = dataset.copy()

    # Extract classes
    classes = list(metadata[group_column])

    # Run GSEA
    gs = GSEA(
        data=norm_counts,
        gene_sets=gene_sets,
        classes=classes,
        permutation_type=permutation_type,
        permutation_num=permutation_num,
        outdir=None,
        method=method,
        threads=threads,
        seed=seed,
    )
    gs.pheno_pos = pheno_pos
    gs.pheno_neg = pheno_neg
    gs.run()

    # Convert results to DataFrame
    results_df = pd.DataFrame(gs.res2d)

    # Optional: parse term names if multiple gene sets
    if not isinstance(gene_sets, (list, tuple)):
        gene_sets = [gene_sets]

    if len(gene_sets) >= 2:
        results_df[["Data Base", "Pathway"]] = results_df["Term"].str.split(
            "__", expand=True
        )
    else:
        results_df["Pathway"] = results_df["Term"]
        results_df["Data Base"] = gene_sets[0]

    pval_columns = [
        col for col in results_df.columns if "p-val" in col or "q-val" in col
    ]
    results_df = results_df[results_df[pval_columns].lt(threshold).any(axis=1)]
    results_df = results_df.sort_values(by="NES", ascending=False)

    return results_df


def nrank_ora(dataset, gene_sets, gene_col, timestamp=None):
    """
    Perform Enrichr enrichment analysis on a gene list.
    Automatically maps Ensembl/Entrez/RefSeq/UniProt IDs to symbols.
    Invalid human gene symbols are dropped.
    """
    if timestamp is None:
        timestamp = get_timestamp()

    # Detect ID type
    id_type = detect_id_type(dataset[gene_col].dropna().astype(str).tolist())
    print(f"Detected ID type: {id_type}")

    # Map & filter
    dataset = prepare_gene_symbols(dataset, gene_col)

    genes = dataset[gene_col].dropna().astype(str).unique().tolist()

    if dataset.empty:
        warnings.warn("No valid human symbols found after mapping.")
        return pd.DataFrame()

    if len(genes) == 0:
        print("No valid genes to run ORA.")
        return pd.DataFrame()

    # --- Run Enrichr --
    try:
        enr = gp.enrichr(
            gene_list=genes, gene_sets=gene_sets, outdir=None, verbose=True
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

    filtered_results = results.head(20)
    print(f"Number of matching results: {len(filtered_results)}")

    return filtered_results
