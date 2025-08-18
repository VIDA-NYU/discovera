import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from Bio import Entrez
import pandas as pd
from collections import defaultdict
import requests
import random
import os

from IPython.display import display, Image


def search_pubmed(term: str, email: str):
    """Perform initial PubMed search and return metadata needed for batching."""
    Entrez.email = email
    with Entrez.esearch(db="pubmed", term=term, usehistory="y") as handle:
        record = Entrez.read(handle)
    total_count = int(record["Count"])
    if total_count == 0:
        print("No publications found for the term.")
        return None, None, 0

    return record["WebEnv"], record["QueryKey"], total_count


def fetch_pubmed_articles(webenv: str, query_key: str, total_count: int, batch_size: int = 500):
    """
    Fetch article publication years and their PMIDs (or PMCIDs if available), grouped by year.

    Returns:
        year_to_ids (dict): Dictionary {year: [list of pmids]}
        all_years (list[int]): Flat list of all years for plotting
    """
    year_to_ids = {}
    for start in range(0, total_count, batch_size):
        print(f"Fetching articles {start + 1} to {min(start + batch_size, total_count)}...")
        with Entrez.esummary(
            db="pubmed", query_key=query_key, WebEnv=webenv,
            retstart=start, retmax=batch_size
        ) as handle:
            records = Entrez.read(handle)

        for rec in records:
            pub_date = rec.get("PubDate", "")
            uid = rec.get("Id")  # Always present
            if pub_date and uid:
                try:
                    year = int(pub_date[:4])
                    year_to_ids.setdefault(year, []).append(uid)
                except ValueError:
                    continue

    all_years = [year for year, ids in year_to_ids.items() for _ in ids]
    return year_to_ids, all_years



def plot_publication_timeline(years: list[int], term: str, figsize: tuple = (12, 6)):
    """Plot the timeline of publication counts per year using a bar plot, with all years on X-axis."""
    if not years:
        print("No valid publication years to plot.")
        return None

    counts = Counter(years)
    sorted_years = sorted(counts)
    frequencies = [counts[y] for y in sorted_years]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(sorted_years, frequencies, width=0.8)

    ax.set_title(f"Publication Timeline for '{term}'", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_ylim(bottom=0)

    # Force integer ticks on both axes
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(sorted_years)  # Show all years
    ax.set_xticklabels(sorted_years, rotation=45, ha='right')  # Rotate for readability

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    display(fig)
    return fig


def literature_timeline(term: str, email: str, batch_size: int = 500, figsize: tuple = (12, 6)):
    """
    Main function to search, fetch, and plot publication timeline.

    Returns:
        - save_path (str): Full path to the saved PNG file of the figure.
    """
    webenv, query_key, total_count = search_pubmed(term, email)
    if total_count == 0 or not webenv or not query_key:
        return None, {}

    print(f"Total articles found: {total_count}")
    year_to_ids, pub_years = fetch_pubmed_articles(webenv, query_key, total_count, batch_size)
    year_to_ids = {
        year: [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]
        for year, pmids in year_to_ids.items()
    }
    fig = plot_publication_timeline(pub_years, term, figsize)
    # Save figure in Beaker environment
    filename = f"{term.replace(' ', '_')}_timeline.png"
    save_path = os.path.join(".", filename)  # Beaker userfiles folder
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Figure saved to Beaker environment: {save_path}")
    return year_to_ids, save_path


def search_pubmed_count(query: str, email: str = "test@example.com") -> int:
    """Return the number of PubMed articles for a given query."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 0,
        "email": email
    }
    r = requests.get(url, params=params)
    try:
        return int(r.json()["esearchresult"]["count"])
    except Exception:
        return 0


def prioritize_genes(
    gene_list: str,
    context_term: str,
    email: str = "test@example.com"
) -> str:
    """
    Prioritize genes in context of disease or phenotype.

    Args:
        gene_list (str): Comma-separated gene names
        context_term (str): e.g., "glioblastoma"
        email (str): For NCBI API compliance

    Returns:
        str: Markdown table of ranked genes
    """

    genes = [g.strip() for g in gene_list.split(",")]
    scores = defaultdict(dict)

    for gene in genes:
        query = f"{gene} {context_term}"
        pubmed_count = search_pubmed_count(query, email=email)
        scores[gene]["pubmed_mentions"] = pubmed_count

        # Simulated centrality and disease relevance scores
        scores[gene]["centrality"] = round(random.uniform(0.2, 1.0), 2)
        scores[gene]["disease_score"] = 1 if pubmed_count > 5 else 0

        # Composite: simple weighted sum
        composite = (
            0.5 * scores[gene]["centrality"] +
            0.3 * (1 if pubmed_count > 0 else 0) +
            0.2 * scores[gene]["disease_score"]
        )
        scores[gene]["composite"] = round(composite, 3)

    return pd.DataFrame(scores).T