import requests
import pandas as pd
import logging

from typing import List, Dict, Optional, Union

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GraphQL API settings
GRAPHQL_URL = "https://rummagene.com/graphql"
DEFAULT_HEADERS = {"Content-Type": "application/json"}


def _post_graphql(
    query: str,
    variables: Dict,
    operation_name: str,
    url: str = GRAPHQL_URL,
    headers: Optional[Dict] = None,
) -> dict:
    """
    Internal helper to send a GraphQL POST request.

    Args:
        query (str): GraphQL query string.
        variables (Dict): Variables for the GraphQL query.
        operation_name (str): Operation name in the GraphQL schema.
        url (str): Target GraphQL endpoint.
        headers (Optional[Dict]): Optional custom headers.

    Returns:
        dict: Parsed data from the response.
    """
    headers = headers or DEFAULT_HEADERS
    payload = {"operationName": operation_name, "variables": variables, "query": query}

    response = requests.post(url, headers=headers, json=payload)

    if not response.ok:
        raise RuntimeError(
            f"GraphQL request failed: {response.status_code}\n{response.text}"
        )

    try:
        return response.json()["data"]
    except (KeyError, ValueError) as e:
        raise ValueError(f"Malformed response JSON: {e}")


def enrich_query(
    genes: Union[str, List[str]],
    filter_term: str = "",
    offset: int = 0,
    first: int = 30,
    url: str = GRAPHQL_URL,
    max_records: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run a gene enrichment analysis for a list of genes.

    Supports auto-pagination and returns matched gene sets, statistics,
    and associated publication data.

    Args:
        genes (Union[str, List[str]]): List of gene symbols or comma-separated string.
        filter_term (str): Optional term to filter enrichment results.
        offset (int): Offset for pagination.
        first (int): Number of entries per page.
        url (str): GraphQL API endpoint.
        max_records (Optional[int]): Maximum number of records to return.

    Returns:
        pd.DataFrame: Enrichment results.
    """
    if isinstance(genes, str):
        genes = [g.strip() for g in genes.split(",") if g.strip()]

    query = """
    query EnrichmentQuery($genes: [String]!, $filterTerm: String = "", $offset: Int = 0, $first: Int = 10) {
      currentBackground {
        enrich(genes: $genes, filterTerm: $filterTerm, offset: $offset, first: $first) {
          totalCount
          nodes {
            geneSetHash
            pvalue
            adjPvalue
            oddsRatio
            nOverlap
            geneSets {
              nodes {
                id
                term
                description
                nGeneIds
                geneSetPmcsById(first: 1) {
                  nodes {
                    pmcInfoByPmcid {
                      pmcid
                      title
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    all_rows = []
    current_offset = offset
    total_count_logged = False

    while True:
        data = _post_graphql(
            query=query,
            variables={
                "genes": genes,
                "filterTerm": filter_term,
                "offset": current_offset,
                "first": first,
            },
            operation_name="EnrichmentQuery",
            url=url,
        )

        enrich_data = data["currentBackground"]["enrich"]

        if not total_count_logged:
            total_count = enrich_data.get("totalCount", 0)
            logger.info(f"Total enrichment entries available: {total_count}")
            total_count_logged = True

        enrichments = enrich_data["nodes"]
        if not enrichments:
            break

        for entry in enrichments:
            for gs in entry["geneSets"]["nodes"]:
                pub_info = gs["geneSetPmcsById"]["nodes"]
                pub = (
                    pub_info[0]["pmcInfoByPmcid"]
                    if pub_info and pub_info[0].get("pmcInfoByPmcid")
                    else None
                )
                all_rows.append(
                    {
                        "Gene Set ID": gs["id"],
                        "Term": gs["term"],
                        "Description": gs["description"],
                        "# Genes in Set": gs["nGeneIds"],
                        "p-value": entry["pvalue"],
                        "adj. p-value": entry["adjPvalue"],
                        "Odds Ratio": entry["oddsRatio"],
                        "# Overlap": entry["nOverlap"],
                        "PubMed Title": pub.get("title") if pub else None,
                        "PMCID": pub.get("pmcid") if pub else None,
                    }
                )

        current_offset += first
        if max_records and len(all_rows) >= max_records:
            break

    return pd.DataFrame(all_rows[:max_records] if max_records else all_rows)


def genes_query(gene_set_id: str, url: str = GRAPHQL_URL) -> pd.DataFrame:
    """
    Fetch the genes associated with a specific gene set.

    Args:
        gene_set_id (str): Gene set UUID.

    Returns:
        pd.DataFrame: Genes with symbol, description, summary, etc.
    """
    query = """
    query ViewGeneSet($id: UUID!) {
      geneSet(id: $id) {
        genes {
          nodes {
            symbol
            ncbiGeneId
            description
            summary
          }
        }
      }
    }
    """

    data = _post_graphql(
        query=query,
        variables={"id": gene_set_id},
        operation_name="ViewGeneSet",
        url=url,
    )

    return pd.DataFrame(data["geneSet"]["genes"]["nodes"])


def overl_query(
    gene_set_id: str,
    genes: List[str],
    url: str = GRAPHQL_URL,
    headers: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Find overlapping genes between a user gene list and a gene set.

    Args:
        gene_set_id (str): Gene set UUID.
        genes (List[str]): List of gene symbols.

    Returns:
        pd.DataFrame: Overlapping genes with annotations.
    """
    query = """
    query OverlapQuery($id: UUID!, $genes: [String]!) {
      geneSet(id: $id) {
        overlap(genes: $genes) {
          nodes {
            symbol
            ncbiGeneId
            description
            summary
          }
        }
      }
    }
    """

    data = _post_graphql(
        query=query,
        variables={"id": gene_set_id, "genes": genes},
        operation_name="OverlapQuery",
        url=url,
        headers=headers,
    )

    return pd.DataFrame(data["geneSet"]["overlap"]["nodes"])


def table_search_query(
    terms: Union[str, List[str]],
    offset: int = 0,
    first: int = 10000,
    url: str = GRAPHQL_URL,
    headers: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Search gene sets by keywords in their term/title.

    Args:
        terms (Union[str, List[str]]): Keyword(s) to search.
        offset (int): Start index for pagination.
        first (int): Max results to return.
        url (str): GraphQL endpoint.

    Returns:
        pd.DataFrame: Matching gene sets.
    """
    query = """
    query TermSearch($terms: [String]!, $offset: Int = 0, $first: Int = 10) {
      geneSetTermSearch(terms: $terms, offset: $offset, first: $first) {
        nodes {
          id
          term
          nGeneIds
        }
        totalCount
      }
    }
    """

    term_list = [terms] if isinstance(terms, str) else terms

    data = _post_graphql(
        query=query,
        variables={"terms": term_list, "offset": offset, "first": first},
        operation_name="TermSearch",
        url=url,
        headers=headers,
    )

    return pd.DataFrame(data["geneSetTermSearch"]["nodes"])


def search_pubmed(term: str, retmax: int = 5000) -> List[str]:
    """
    Search PubMed Central (PMC) for articles related to a keyword.

    Args:
        term (str): Search term.
        retmax (int): Maximum number of results to return.

    Returns:
        List[str]: List of PMC IDs.
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pmc",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",  # options: 'relevance', 'pub+date'
    }

    r = requests.get(search_url, params=search_params)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_pmc_info(
    pmcids: List[str], url: str = GRAPHQL_URL, headers: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Get metadata for PMC articles by their PMCIDs.

    Args:
        pmcids (List[str]): List of PMC article IDs.

    Returns:
        pd.DataFrame: Metadata with title, year, DOI.
    """
    query = """
    query GetPmcInfoByIds($pmcids: [String]!) {
      getPmcInfoByIds(pmcids: $pmcids) {
        nodes {
          pmcid
          title
          yr
          doi
        }
      }
    }
    """

    data = _post_graphql(
        query=query,
        variables={"pmcids": pmcids},
        operation_name="GetPmcInfoByIds",
        url=url,
        headers=headers,
    )

    return pd.DataFrame(data["getPmcInfoByIds"]["nodes"])


def gene_sets_paper_query(
    pmcids: List[str], url: str = GRAPHQL_URL, headers: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Determine how frequently gene sets (terms) appear in PMC articles.

    Args:
        pmcids (List[str]): List of PMCIDs.

    Returns:
        pd.DataFrame: Table of term frequencies per article.
    """
    query = """
    query TermsPmcs($pmcids: [String]!) {
      termsPmcsCount(pmcids: $pmcids) {
        nodes {
          pmc
          id
          term
          count
        }
      }
    }
    """

    data = _post_graphql(
        query=query,
        variables={"pmcids": pmcids},
        operation_name="TermsPmcs",
        url=url,
        headers=headers,
    )
    data = pd.DataFrame(data["termsPmcsCount"]["nodes"])
    data = data.rename(columns={"pmc": "pmcid"})
    return data
