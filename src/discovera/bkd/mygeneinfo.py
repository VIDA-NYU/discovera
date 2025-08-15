import mygene
import pandas as pd

def fetch_gene_annota(gene_list):
    """
    Fetch detailed gene annotations for a list of genes using MyGene.info.

    Parameters:
        gene_list (list): List of gene identifiers (HGNC symbols recommended)

    Returns:
        pd.DataFrame: DataFrame containing gene annotations
    """
    mg = mygene.MyGeneInfo()

    fields = [
        "symbol", "name", "summary", "alias",
        "entrezgene", "ensembl.gene", "HGNC",
        "type_of_gene",
        "genomic_pos", "cytoband",
        "go.BP", "go.MF", "go.CC",
        "uniprot.Swiss-Prot", "interpro", "pfam",
        "pathway.kegg", "pathway.reactome", "pathway.wikipathways",
        "disease", "clinvar",
        "homologene"
    ]

    results = mg.querymany(
        gene_list,
        scopes="symbol",
        fields=",".join(fields),
        species="human",
        as_dataframe=False
    )

    def extract_names(item):
        if isinstance(item, list):
            return "; ".join([x.get("name", str(x)) if isinstance(x, dict) else str(x) for x in item])
        elif isinstance(item, dict):
            if "name" in item:
                return item["name"]
            return "; ".join([f"{k}: {v}" for k, v in item.items()])
        return str(item) if item else ""

    def extract_go_terms(go_category):
        if isinstance(go_category, list):
            return "; ".join([g.get("term", "") for g in go_category])
        elif isinstance(go_category, dict):
            return go_category.get("term", "")
        return ""

    # Process results into clean list
    clean_data = []
    for r in results:
        clean_data.append({
            "symbol": r.get("symbol", ""),
            "name": r.get("name", ""),
            "summary": r.get("summary", ""),
            "aliases": "; ".join(r.get("alias", [])) if isinstance(r.get("alias"), list) else r.get("alias", ""),
            "entrezgene": r.get("entrezgene", ""),
            "ensembl_gene": r.get("ensembl", {}).get("gene", "") if isinstance(r.get("ensembl"), dict) else "",
            "HGNC": r.get("HGNC", "")
        })

    df = pd.DataFrame(clean_data)
    return df
