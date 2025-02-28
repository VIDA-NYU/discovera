


import pandas as pd

def group_edges(edges, group_columns=None):
    """
    Groups the edges DataFrame by the specified columns and counts occurrences.
    
    This function works well for pairs of relationships between two defined nodes. 
    However, it may not perform well when the relationships are more complex, 
    such as cases where 'subj.name' and 'obj.name' are not clearly defined.

    Args:
        edges (pd.DataFrame): The DataFrame containing edge relationships.
        group_columns (list, optional): List of column names to group by.
                                        If None, defaults to ["nodes", "type", "subj.name", "obj.name"].

    Returns:
        pd.DataFrame: A grouped DataFrame with a count column, sorted by count and type.
    
    Raises:
        ValueError: If any specified group column is not present in the DataFrame.
    """
    # Set default group columns if not provided
    if group_columns is None:
        group_columns = ["nodes", "type", "subj.name", "obj.name"]

    # Check if the provided group columns exist in the DataFrame
    if not set(group_columns).issubset(edges.columns):
        raise ValueError("Some specified group columns are not in the DataFrame.")

    # Perform grouping and counting
    result = edges.groupby(group_columns).size().reset_index(name="count")
    
    # Dynamically sort by 'count' (descending) and 'type' and other columns (ascending)
    sort_columns = ["count"] + [col for col in group_columns if col != "count"]
    result = result.sort_values(by=sort_columns, ascending=[False] + [True] * (len(sort_columns) - 1))
    
    return result

