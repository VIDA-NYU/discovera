
"""
Centralized mapping from report keyword to benchmark CSV column.
Used across multiple scripts for consistency.
"""

# Mapping JSON keywords to benchmark columns
REPORT_TO_COLUMN = {
    "discovera(gpt-4o)": "Discovera (gpt-4o)",
    "llm(gpt-4o)": "LLM (gpt-4o)",
    "groundtruth": "Ground Truth",
    "discovera(o4-mini)": "Discovera (o4-mini)",
    #"biomni": "Biomni",
    "biomni(11-05-25)": "Biomni (11-05-25)",
    "llm(o4-mini)": "LLM (o4-mini)",
    "biomni(o4-mini)": "Biomni (o4-mini)"

    # Add more mappings here if needed
}