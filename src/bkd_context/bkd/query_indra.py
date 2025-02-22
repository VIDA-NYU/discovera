import pandas as pd
import json
import time
import requests
import json

from pandas import json_normalize
from indra.sources.indra_db_rest.api import get_statements
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def retry_on_failure(func, max_retries=3, wait_time=5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if "502 Bad Gateway" in str(e):
                print(f"502 Bad Gateway encountered. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(wait_time * (attempt + 1))  # Exponential backoff
            else:
                raise e  # Raise other exceptions immediately
    raise Exception("Max retries reached. Unable to complete the request.")


def expand_and_flatten(state_json):
    """
    Expands and flattens a statement JSON object into multiple rows
    if necessary (e.g., handling nested lists like 'evidence').

    Parameters:
    - statement_json: A single statement JSON object.

    Returns:
    - A DataFrame where nested structures are expanded into multiple rows.
    """
    flat_data = json_normalize(state_json)
    if 'evidence' in state_json:
        evidence_df = json_normalize(state_json['evidence'])
        expanded_df = flat_data.merge(evidence_df, how='cross')
    else:
        expanded_df = flat_data

    return expanded_df

def extract_relations(statements):
    """
    This function takes a list of statements (each with a .to_json() method),
    extracts the JSON data, expands them where necessary, and appends them together into one DataFrame.

    Parameters:
    - statements: A list of statement objects (with a .to_json() method).

    Returns:
    - A Pandas DataFrame containing all the extracted and expanded statement data.
    """
    all_dataframes = []
    for statement in statements:
        statement_json = statement.to_json()
        expanded_df = expand_and_flatten(statement_json)
        all_dataframes.append(expanded_df)
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
    else:
        # Print message and return an empty DataFrame with no columns if there is nothing to concatenate
        print("No results found.")
        final_df = pd.DataFrame()

    return final_df


class EdgeExtractor:
    def __init__(self, nodes):
        self.nodes = nodes
        self.statements = None
        self.edges = None

    def get_statements(self):
        p = retry_on_failure(
            get_statements,
            max_retries=3,
            wait_time=5,
            subject=None,
            object=None,
            agents=self.nodes,
            stmt_type=None,
            use_exact_type=False,
            limit=None,
            persist=True,
            timeout=None,
            strict_stop=False,
            ev_limit=None,
            sort_by="ev_count",
            tries=3,
            use_obtained_counts=False,
            api_key=None,
        )
        self.statements = p.statements
        return self.statements

    def extract_edges(self):
        if self.statements is None:
            self.get_statements()
        self.edges = extract_relations(self.statements)
        if self.edges.empty:
            print("No relationships found in the statements.")
        return self.edges

    def get_type_counts(self):
        if self.edges is None:
            self.extract_edges()
        return self.edges.type.value_counts() if not self.edges.empty else pd.Series()

    def group_by_type(self):
        if self.edges is None:
            self.extract_edges()
        if not self.edges.empty:
            return self.edges.groupby(["type", "subj.name", "obj.name"]).size().reset_index(name="count")
        else:
            print("No relationships to group by type.")
            return pd.DataFrame()

    def get_statements_raw(self):
        if self.statements is None:
            self.get_statements()
        return self.statements


def bulk_edges(nodes_lists):
    results = []
    for nodes in tqdm(nodes_lists):
        try:
            print(nodes)
            edge_extra = EdgeExtractor(nodes)
            if edge_extra.get_statements() == []:
                print("No relationships documented in INDRA")
                type_count = pd.NA
                edge_types = pd.NA
                relationships = pd.NA
                edges = pd.NA
            else:
                type_count = edge_extra.get_type_counts()
                edge_types = edge_extra.group_by_type()
                relationships = edge_extra.get_statements_raw()
                edges = edge_extra.extract_edges()
            results.append(
                {
                    "nodes": nodes,
                    "type_count": type_count,
                    "edge_types": edge_types,
                    "relationships": relationships,
                    "edges": edges,
                }
            )
        except Exception as e:
            print(f"An error occurred while processing nodes {nodes}: {e}")
            continue  # Continue to the next iteration even if there's an error

    return pd.DataFrame(results)


def process_results(results):
    results_filtered = results[results["type_count"].notna()]
    results_filtered["total_docum"] = results_filtered["type_count"].apply(
        lambda x: x.sum() if isinstance(x, pd.Series) else 0
    )
    results_filtered = results_filtered.sort_values(by="total_docum", ascending=False).reset_index(drop=True)
    return results_filtered


def bulk_edges_parallel(nodes_lists, max_workers=5):
    results = []

    def process_node(nodes):
        try:
            edge_extra = EdgeExtractor(nodes)
            if edge_extra.get_statements() == []:
                print(f"No relationships documented in INDRA for nodes: {nodes}")
                return {
                    "nodes": nodes,
                    "type_count": pd.NA,
                    "edge_types": pd.NA,
                    "relationships": pd.NA,
                    "edges": pd.NA,
                }
            else:
                return {
                    "nodes": nodes,
                    "type_count": edge_extra.get_type_counts(),
                    "edge_types": edge_extra.group_by_type(),
                    "relationships": edge_extra.get_statements_raw(),
                    "edges": edge_extra.extract_edges(),
                }
        except Exception as e:
            print(f"An error occurred while processing nodes {nodes}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_node, nodes): nodes for nodes in nodes_lists}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:  # Skip failed attempts
                results.append(result)

    return pd.DataFrame(results)


def query_indra_networ(source=None, target=None, response_format="json"):
    """
    Queries the INDRA API with specified source and target entities.
    
    Parameters:
        source (str, optional): The source entity (e.g., a gene or protein name). Defaults to None.
        target (str, optional): The target entity. Defaults to None.
        response_format (str): The format of the response. Defaults to "json".
    
    Returns:
        dict: The API response in JSON format if successful, else an error message.
    """
    url = 'https://network.indra.bio/api/query'  # INDRA API endpoint
    
    payload = {
        "source": source,
        "target": target,
        "format": response_format
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for HTTP issues
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def query_bulk_pairs(pairs):
    """
    Queries INDRA API for multiple source-target relationships.
    
    Parameters:
        pairs (list of tuples): List of (source, target) pairs.
    
    Returns:
        list: List of responses from the INDRA API.
    """
    results = []
    for source, target in pairs:
        result = query_indra_networ(source=source, target=target)
        results.append({"source": source, "target": target, "response": result})
    return results


def excerpts(results):
    results = pd.concat(results['edges'].values, axis=0, ignore_index=True)
    return results

def relationships(results):
    results = pd.concat(results['edge_types'].values, axis=0, ignore_index=True)
    return results