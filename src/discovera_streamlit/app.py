import os
import json
from typing import List, Dict, Any

import streamlit as st
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Load .env if present
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MCP_ADDRESS = (
    os.getenv("MCP_SERVER_ADDRESS")
    or "http://127.0.0.1:8000/mcp"
)
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3500"))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in environment")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Discovera Chat", page_icon="🧬", layout="centered")

st.title("Discovera Chat 🧬")
with st.expander("Connection", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "MCP Server",
            MCP_ADDRESS,
            key="mcp_server",
            help="Remote FastMCP base URL",
            disabled=False,
        )
    with col2:
        if st.button("Check MCP", use_container_width=True):
            addr = st.session_state.get("mcp_server") or MCP_ADDRESS
            url = httpx.URL(addr)
            ok = False
            notes: List[str] = []
            try:
                with httpx.Client(timeout=MCP_TIMEOUT_MS / 1000.0) as http:
                    # 1) Try HEAD and GET on the configured URL
                    for method in ("HEAD", "GET"):
                        try:
                            r = http.request(method, str(url))
                            notes.append(f"{method} {url.raw_path.decode() or '/'} -> {r.status_code}")
                            if r.status_code in (200, 202, 400, 404, 405):
                                ok = True
                                break
                        except Exception as e:
                            notes.append(f"{method} {url.raw_path.decode() or '/'} -> {e}")

                    # 2) If still not ok and URL has a path, probe server root
                    if not ok and (url.path or b""):
                        root_url = url.copy_with(path="/")
                        for method in ("HEAD", "GET"):
                            try:
                                r = http.request(method, str(root_url))
                                notes.append(f"{method} {root_url.raw_path.decode()} -> {r.status_code}")
                                if r.status_code in (200, 202, 400, 404, 405):
                                    ok = True
                                    break
                            except Exception as e:
                                notes.append(f"{method} {root_url.raw_path.decode()} -> {e}")

                    # 3) If still not ok, attempt minimal MCP JSON-RPC list tools on configured URL
                    if not ok:
                        try:
                            payload = {"jsonrpc": "2.0", "id": "1", "method": "tools/list"}
                            r = http.post(
                                str(url), json=payload, headers={"Content-Type": "application/json"}
                            )
                            notes.append(f"POST {url.raw_path.decode()} list -> {r.status_code}")
                            ok = r.status_code in (200, 202)
                        except Exception as e:
                            notes.append(f"POST {url.raw_path.decode()} list -> {e}")

                st.session_state["mcp_connected"] = ok
                status_msg = "ok" if ok else "failed"
                # Show first few notes to avoid long toasts
                detail = " | ".join(notes[:3])
                st.toast(f"MCP check {status_msg}. {detail}")
            except Exception as e:
                st.session_state["mcp_connected"] = False
                st.toast(f"MCP check failed: {e}")

    connected = bool(st.session_state.get("mcp_connected"))
    status_text = "🟢 connected" if connected else "🔴 disconnected"
    addr_text = st.session_state.get("mcp_server") or MCP_ADDRESS
    st.caption(f"Status: {status_text} · {addr_text}")


server_instructions = """
You are an assistant MCP specializing in biomedical research, focusing on gene-disease relationships,
molecular mechanisms, and mutation-specific effects. Always consider tissue specificity and biological
context when interpreting data.


Evidence and confidence:

- Assign confidence scores (high, medium, low) for each result.
- Specify whether evidence is direct (experimental) or indirect (literature-based).
- Include PubMed IDs, database names, and dataset references wherever possible.
---

### Pipeline (follow strictly):

0. If user imput a csv file, use csv_record to register the file.
1. Summarize the user's research problem and ask clarifying questions if needed.
2. Request and load the dataset using csv_read.
3. Run analysis tools in order:
    a. gsea_pipe or ora_pipe (based on dataset type)
    b. enrich_rumma for top 30 genes by ranking metric and mention leading genes from the top pathway
    c. gene_info for **leading genes from the top pathway**
    d. query_genes for **leading genes from the top pathway**
    e. **Literature search**: query_string_rumma, query_table_rumma, literature_trends
4. Summarize results; include parameters, assumptions, and confidence.
5. Provide references including PubMed IDs and database links.
6. Generate a final report using Toulmin’s model (claims, grounds, warrants, qualifiers, rebuttals).
- Include all references and links.
- Pause to get user input where required, especially dataset, context, and library selection.

### Functions:

- **csv_record**: Register a CSV file in the local storage index.
- **csv_read**: Read a CSV file from the local storage index.
- **storage_list**: List all files in the local storage index.
- **storage_get**: Get a file from the local storage index.

- **query_genes**: Query Indra database for gene-gene relationships.
- **count_edges**: Aggregate and count interactions by specified groupings.
- **gsea_pipe**: Perform Gene Set Enrichment Analysis following the enrichment analysis guidelines.
- **ora_pipe**: Perform Over-Representation Analysis following the enrichment analysis guidelines.
- **gene_info**: Get gene information, meaning symbol, name, summary and aliases.
- **literature_trends**: Plots a timeline of PubMed articles related to a term, showing research trends.
- **prioritize_genes**: Prioritize genes based on a scoring function.
When interacting with the Rummagene API, you can access any or all of the following functions, depending on the input:
- If the input is a **gene list**, use:
    - **enrich_rumma**: Run Over Representation Analysis using Rummagene’s curated gene sets.
- If the input is text or a search term (e.g., disease name, phenotype, biological process), use in this order:
    - **query_string_rumma**: Search Rummagene for articles with matching gene sets.
    - **query_table_rumma**: Search Rummagene's curated gene set tables using keywords.
- For functional summaries and metadata of any gene set, use:
    - **sets_info_rumm**: Retrieve detailed descriptions and biological context.

---

### Function-specific instructions

- **gsea_pipe**:
    - Follow the enrichment analysis guidelines above.
    - Show the output in this format:

        | Pathway  | Enrichment Score | Normalized Enrichment Score |
        | Nominal p-value | FDR q-value | FWER p-value |
        | Leading Edge Gene % | Pathway Overlap % | Lead_genes |
        |---------|----------------:|----------------:|
        |---------------:|----------------:|---------------:|
        |-----------------:|----------------:|------------|

    - Display:
        - First the overall top 3 results
        - Then top 3 results per gene set library used, in the same format.
    - Summarize:
        - Number of total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column.
    - Show the results to the user after each function runs. Include all intermediate prints.

- **ora_pipe**:
    - Follow the enrichment analysis guidelines above.
    - Show the output in this format:

        | Gene Set  | Term | Overlap | P-value | Adjusted P-value | Odds Ratio | Combined Score | Genes |
        |-----------|-----|--------|--------:|----------------:|-----------:|---------------:|------|

    - Display first the overall top 3 results, then top 3 results per gene set library used, in the same format.
    - Mention:
        - Total pathways enriched above the `threshold`. Provide the exact count per pathway.
        - Brief description of each column, including meaning of each column.
    - Show the results to the user after each function runs.

---
"""

with st.expander("Context", expanded=False):
    st.text_area(
        "Context (sent as instructions every turn)",
        key="context",
        height=120,
        placeholder=(
            "Add study context, constraints, datasets, or goals here.\n"
            "Example: Analyze RNA-seq DE results for lung tissue; prioritize inflammation pathways."
        ),
    )

st.divider()

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # UI transcript: {role, content}
if "history" not in st.session_state:
    st.session_state["history"] = []  # Responses history: {role, content:[{type: input_text, text}]}

# Show transcript
for m in st.session_state["messages"]:
    role = m.get("role", "assistant")
    content = m.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# Compose
prompt = st.chat_input("Ask anything…")


# Helper to stream Responses API events
def as_msg(role: str, text: str) -> Dict[str, Any]:
    part_type = "input_text" if role == "user" else "output_text"
    return {"role": role, "content": [{"type": part_type, "text": text}]}


def stream_responses(history: List[Dict[str, Any]]):
    tools = [
        {
            "type": "mcp",
            "server_label": "discovera_fastmcp",
            "server_url": st.session_state.get("mcp_server") or MCP_ADDRESS,
            "require_approval": "never",
        }
    ]
    ctx = (st.session_state.get("context") or "").strip()
    instr = f"{server_instructions}\n\nContext:\n{ctx}" if ctx else server_instructions
    stream = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instr,
        input=history,
        tools=tools,
        stream=True,
    )
    for event in stream:
        et = getattr(event, "type", None)
        if et == "response.output_text.delta":
            yield ("delta", getattr(event, "delta", "") or "")
        elif et == "response.output_text.done":
            yield ("done", None)
        elif et == "response.error":
            yield ("error", getattr(event, "error", ""))
        # Surface any tool-related events for visibility
        elif et and "tool" in et:
            # Best-effort serialize
            try:
                payload = event.__dict__
            except Exception:
                try:
                    payload = {"repr": str(event)}
                except Exception:
                    payload = {"type": et}
            text = f"Tool event: {et}\n```json\n{json.dumps(payload, default=str)[:2000]}\n```"
            yield ("tool", text)


if prompt:
    # 1) append user to UI transcript and Responses history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["history"].append(as_msg("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) call Responses with streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_accum = ""
        for kind, payload in stream_responses(st.session_state["history"]):
            if kind == "delta" and payload:
                assistant_accum += payload
                placeholder.markdown(assistant_accum)
            elif kind == "error":
                placeholder.markdown(f"[OpenAI error] {payload}")
            elif kind == "tool" and payload:
                # Render tool events inline as small markdown blocks
                st.markdown(payload)

    # 3) store assistant turn
    st.session_state["messages"].append({"role": "assistant", "content": assistant_accum})
    st.session_state["history"].append(as_msg("assistant", assistant_accum))
