def group_edges(edges, grouping):
    """
    Groups the edges DataFrame by predefined grouping types and counts occurrences.

    Args:
        edges (pd.DataFrame): The DataFrame containing edge relationships.
        grouping (str, optional): The type of grouping to apply. Options are:
            - "summary": Groups only by 'nodes'.
            - "detailed": Groups by 'nodes', 'type', 'subj.name', and 'obj.name'. (default)
            - "view": Groups by 'nodes' and 'type'.

    Returns:
        pd.DataFrame: A grouped DataFrame with a count column, sorted by count and type.

    Raises:
        ValueError: If an invalid group_type is provided.
    """

    # Define grouping options
    group_options = {
        "summary": ["nodes"],
        "detailed": ["nodes", "type", "subj.name", "obj.name"],
        "view": ["nodes", "type"],
    }

    # Validate group_type
    if grouping not in group_options:
        raise ValueError(
            f"Invalid group_type '{grouping}'. Choose from {list(group_options.keys())}."
        )

    group_columns = group_options[grouping]

    # Check if the provided group columns exist in the DataFrame
    if not set(group_columns).issubset(edges.columns):
        raise ValueError("Some specified group columns are not in the DataFrame.")

    # Perform grouping and counting
    result = edges.groupby(group_columns).size().reset_index(name="count")

    # Sort dynamically: count descending, other columns ascending
    sort_columns = ["count"] + [col for col in group_columns if col != "count"]
    result = result.sort_values(
        by=sort_columns, ascending=[False] + [True] * (len(sort_columns) - 1)
    )

    return result
