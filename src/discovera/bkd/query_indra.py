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
                print(
                    f"502 Bad Gateway encountered. Retrying {attempt + 1}/{max_retries}..."
                )
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
    if "evidence" in state_json:
        evidence_df = json_normalize(state_json["evidence"])
        expanded_df = flat_data.merge(evidence_df, how="cross")
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
            print(
                f"⚠️ No edges found for nodes: {nodes}"
            )  # Print nodes with empty results
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
        return [
            node.strip(" \"'") for node in nodes
        ]  # Convert tuple to list & clean quotes

    else:
        raise ValueError(
            "Invalid input format for nodes. Use a string, list, or tuple."
        )


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
        for statements in tqdm(
            executor.map(nodes_batch, nodes_lists),
            total=len(nodes_lists),
            desc="Processing nodes",
        ):
            if statements is not None:
                results.append(statements)

    if results:
        combined_df = pd.concat(results, ignore_index=True)
        # Select only the desired columns from the combined DataFrame
        selected_columns = [
            "nodes",
            "type",
            "subj.name",
            "obj.name",
            "belief",
            "text",
            "text_refs.PMID",
            "text_refs.DOI",
            "text_refs.PMCID",
            "text_refs.SOURCE",
            "text_refs.READER",
        ]
        # Ensure that the columns exist before selecting to avoid KeyErrors
        existing_columns = [
            col for col in selected_columns if col in combined_df.columns
        ]
        combined_df = combined_df[existing_columns]
        combined_df["url"] = "https://doi.org/" + combined_df["text_refs.DOI"]

        # TODO: check if this is the best way to handle this
        # Step 1: Pick one row with highest belief per unique 'type'
        type_representatives = combined_df.sort_values(
            "belief", ascending=False
        ).drop_duplicates(subset="type", keep="first")

        # Step 2: Exclude already selected rows
        remaining_df = combined_df.drop(type_representatives.index)

        # Step 3: Track used subj and obj names
        used_subj = set(type_representatives["subj.name"])
        used_obj = set(type_representatives["obj.name"])

        # Step 4: Define mask to prioritize diverse subj/obj
        remaining_df = remaining_df.assign(
            is_new_subj=~remaining_df["subj.name"].isin(used_subj),
            is_new_obj=~remaining_df["obj.name"].isin(used_obj),
        )

        # Step 5: Sort by new subj/obj and belief
        remaining_df = remaining_df.sort_values(
            by=["is_new_subj", "is_new_obj", "belief"], ascending=[False, False, False]
        )

        # Step 6: Select additional rows to make total 20
        additional_needed = 20 - len(type_representatives)
        additional_rows = remaining_df.head(additional_needed)

        # Combine and reset index
        final_df = pd.concat([type_representatives, additional_rows]).reset_index(
            drop=True
        )

        return final_df
    else:
        return pd.DataFrame()


# import pandas as pd
# from itertools import combinations
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# import os
# import hashlib
# import pickle

# def hash_node_combination(combo):
#     """Create a unique hash for a node combination to use in filenames."""
#     return hashlib.md5("::".join(sorted(combo)).encode()).hexdigest()

# def bulk_edges(nodes, size, cache_dir="edge_cache", save_every=100):
#     """
#     Process a list of node sets in parallel to extract relationships, saving progress.

#     Args:
#         nodes (list): List of nodes.
#         size (int): Size of combinations.
#         cache_dir (str): Directory to store intermediate results.
#         save_every (int): Save after this many processed batches.

#     Returns:
#         pd.DataFrame: Combined results from all node combinations.
#     """
#     os.makedirs(cache_dir, exist_ok=True)
#     nodes = normalize_nodes(nodes)
#     nodes_lists = list(combinations(nodes, r=int(size)))

#     # Load cache index if exists
#     processed_hashes = set(os.listdir(cache_dir))
#     results = []

#     def process_and_cache(combo):
#         combo_hash = hash_node_combination(combo) + ".pkl"
#         if combo_hash in processed_hashes:
#             with open(os.path.join(cache_dir, combo_hash), "rb") as f:
#                 return pickle.load(f)
#         statements = nodes_batch(combo)
#         if statements is not None and not statements.empty:
#             with open(os.path.join(cache_dir, combo_hash), "wb") as f:
#                 pickle.dump(statements, f)
#             return statements
#         return None

#     with ThreadPoolExecutor() as executor:
#         for res in tqdm(executor.map(process_and_cache, nodes_lists), total=len(nodes_lists), desc="Processing nodes"):
#             if res is not None:
#                 results.append(res)
#                 if len(results) % save_every == 0:
#                     # Save intermediate result
#                     pd.concat(results, ignore_index=True).to_csv(
#                         os.path.join(cache_dir, "intermediate_results.csv"), index=False
#                     )

#     if results:
#         combined_df = pd.concat(results, ignore_index=True)
#         selected_columns = [
#             "nodes", "type", "subj.name", "obj.name", "belief", "text",
#             "text_refs.PMID", "text_refs.DOI", "text_refs.PMCID",
#             "text_refs.SOURCE", "text_refs.READER"
#         ]
#         existing_columns = [col for col in selected_columns if col in combined_df.columns]
#         combined_df = combined_df[existing_columns]
#         if 'text_refs.DOI' in combined_df.columns:
#             combined_df['url'] = 'https://doi.org/' + combined_df['text_refs.DOI']
#         return combined_df
#     else:
#         return pd.DataFrame()


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
    url = "https://network.indra.bio/api/query"  # INDRA API endpoint

    payload = {"source": source, "target": target, "format": response_format}

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


def explain_downstream(source, targets, id_type="hgnc.symbol"):
    """
    Sends a POST request to the Indra Discovery API to explain downstream relationships.

    Parameters:
        source (str): The source gene/protein symbol (e.g., "BRCA1").
        targets (list): A list of target gene/protein symbols (e.g., ["TP53", "PARP1"]).
        id_type (str): The identifier type to use for source and targets. Default is "hgnc.symbol".

    Returns:
        dict: The JSON response from the API containing the explanation of downstream interactions.

    Example:
        results = explain_downstream("BRCA1", ["TP53", "PARP1", "RAD51", "CHEK2"])
        print(results)
    """
    url = "https://discovery.indra.bio/api/explain_downstream"
    payload = {"source": source, "targets": targets, "id_type": id_type}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()
