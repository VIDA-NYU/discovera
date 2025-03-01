import pandas as pd
import time
import requests

from itertools import combinations
from pandas import json_normalize
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from indra.sources.indra_db_rest.api import get_statements


def retry_on_failure(func, max_retries=3, wait_time=5, **kwargs):
    """
    Retry a function call upon failure, with exponential backoff.

    This function retries the execution of the given function up to the specified
    maximum number of retries. If the function fails with a "502 Bad Gateway"
    error, it will retry with an increasing wait time between attempts.

    Args:
        func (callable): The function to be executed.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        wait_time (int, optional): Initial wait time (in seconds) between retries. Defaults to 5.
        **kwargs: Additional keyword arguments passed to the function.

    Returns:
        The return value of the function on success.

    Raises:
        Exception: If the function fails after the maximum number of retries.
    """
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
    Extract and expand relations from a list of statements.

    This function takes a list of statements, each with a `.to_json()` method,
    and extracts the relevant data. It expands any nested structures and combines
    the results into one DataFrame.

    Args:
        statements (list): A list of statement objects (with a `.to_json()` method).

    Returns:
        pd.DataFrame: A DataFrame containing all the extracted and expanded statement data.
    """
    all_dataframes = []
    for statement in statements:
        statement_json = statement.to_json()
        expanded_df = expand_and_flatten(statement_json)
        all_dataframes.append(expanded_df)
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    return final_df


class EdgeExtractor:
    """
    A class to extract edges (relationships) from INDRA statements for given nodes.

    This class interacts with the INDRA API to retrieve statements involving the
    provided nodes. It then extracts and flattens relationships into a DataFrame.

    Attributes:
        nodes (list): A list of nodes for which to extract relations.
        statements (list, optional): A list of statements retrieved from the INDRA API.
        edges (pd.DataFrame, optional): A DataFrame containing extracted edges.
    """

    def __init__(self, nodes):
        """
        Initialize the EdgeExtractor with a list of nodes.

        Args:
            nodes (list): A list of nodes for edge extraction.
        """
        self.nodes = nodes
        self.statements = None
        self.edges = None

    def get_statements(self):
        """
        Retrieve statements related to the given nodes from the INDRA API.

        This function uses the `retry_on_failure` utility to handle retry logic in case
        of failure. It stores the retrieved statements in the `self.statements` attribute.

        Returns:
            list: A list of statements related to the nodes.
        """
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
        """
        Extract edges from the statements related to the given nodes.

        This function checks if statements have been retrieved; if not, it calls
        `get_statements` to fetch them. It then processes the statements to extract
        relationships and returns them as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted edges.
        """
        if self.statements is None:
            self.get_statements()
        self.edges = extract_relations(self.statements)
        if self.edges.empty:
            print("No relationships found in the statements.")
        return self.edges


def nodes_batch(nodes):
    """
    Process a batch of nodes and extract relationships.

    This function processes the provided list of nodes, extracts relationships using
    the `EdgeExtractor` class, and returns a DataFrame with relationships and the nodes.

    Args:
        nodes (list): A list of nodes for edge extraction.

    Returns:
        pd.DataFrame or None: A DataFrame containing the extracted relationships or None
        if no relationships are found for the given nodes.
    """
    try:
        # Process the nodes and extract edges
        edge_extra = EdgeExtractor(nodes)
        statements = edge_extra.extract_edges()
        
        if not statements.empty:
            # Add a new column 'nodes' to the statements DataFrame with the current nodes
            statements = statements.assign(nodes=[tuple(nodes)] * len(statements))
            return statements
        else:
            print(f"⚠️ No edges found for nodes: {nodes}")  # Print nodes with empty results
            return None
        
    except Exception as e:
        print(f"❌ An error occurred while processing nodes {nodes}: {e}")
        return None


def normalize_nodes(nodes):
    """
    Normalize the input to always return a clean list of node names.

    Args:
        nodes (str, list, or tuple): The input nodes, which can be:
                                     - A single string with nodes (with or without quotes/parentheses).
                                     - A list or tuple of nodes.
                                     - A list containing a single string with comma-separated nodes.

    Returns:
        list: A properly formatted list of node names.
    """
    if isinstance(nodes, str):
        # Remove surrounding parentheses or extra quotes
        nodes = nodes.strip("()\"'")

        # Split by commas and clean spaces/quotes
        return [node.strip(" \"'") for node in nodes.split(",")]
    
    elif isinstance(nodes, list):
        # If list contains a single string, treat it like a string case
        if len(nodes) == 1 and isinstance(nodes[0], str):
            return normalize_nodes(nodes[0])  
        return [node.strip(" \"'") for node in nodes]  # Clean quotes from each item
    
    elif isinstance(nodes, tuple):
        return [node.strip(" \"'") for node in nodes]  # Convert tuple to list & clean quotes

    else:
        raise ValueError("Invalid input format for nodes. Use a string, list, or tuple.")


def bulk_edges(nodes, size):
    """
    Process a list of node sets in parallel to extract relationships.

    This function processes multiple sets of nodes in parallel using the `ThreadPoolExecutor`
    and extracts relationship information for each node set. It then combines the results
    into one DataFrame.

    Args:
        nodes_lists (list): A list of node sets (each a list of nodes).

    Returns:
        pd.DataFrame: A DataFrame containing all the extracted relationships across the nodes.
    """

    results = []
    
    nodes = normalize_nodes(nodes)

    nodes_lists = list(combinations(nodes, r=int(size)))

    with ThreadPoolExecutor() as executor:
        # Map the nodes_lists to the executor for parallel processing
        for statements in tqdm(executor.map(nodes_batch, nodes_lists), total=len(nodes_lists), desc="Processing nodes"):
            if statements is not None:
                results.append(statements)

    if results:
        combined_df = pd.concat(results, ignore_index=True)
        # Select only the desired columns from the combined DataFrame
        selected_columns = [
            "nodes", "type", "subj.name", "obj.name", "belief", "text",
            "text_refs.PMID", "text_refs.DOI", "text_refs.PMCID",
            "text_refs.SOURCE", "text_refs.READER"
        ]
        # Ensure that the columns exist before selecting to avoid KeyErrors
        existing_columns = [col for col in selected_columns if col in combined_df.columns]
        combined_df = combined_df[existing_columns]
        combined_df['url'] = 'https://doi.org/' + combined_df['text_refs.DOI']

        return combined_df
    else:
        return pd.DataFrame()
    

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
    