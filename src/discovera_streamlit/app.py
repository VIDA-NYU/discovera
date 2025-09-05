import os
import json
import asyncio
import threading
import queue
from typing import List, Dict, Any

import streamlit as st
import httpx
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Load .env if present
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MCP_ADDRESS = os.getenv("MCP_SERVER_ADDRESS") or "http://127.0.0.1:8000/mcp"
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3500"))
LANGGRAPH_CONFIG = RunnableConfig(
    {"configurable": {"thread_id": "discovera_streamlit"}}
)

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in environment")
    st.stop()

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
                            notes.append(
                                f"{method} {url.raw_path.decode() or '/'} -> {r.status_code}"
                            )
                            if r.status_code in (200, 202, 400, 404, 405):
                                ok = True
                                break
                        except Exception as e:
                            notes.append(
                                f"{method} {url.raw_path.decode() or '/'} -> {e}"
                            )

                    # 2) If still not ok and URL has a path, probe server root
                    if not ok and (url.path or b""):
                        root_url = url.copy_with(path="/")
                        for method in ("HEAD", "GET"):
                            try:
                                r = http.request(method, str(root_url))
                                notes.append(
                                    f"{method} {root_url.raw_path.decode()} -> {r.status_code}"
                                )
                                if r.status_code in (200, 202, 400, 404, 405):
                                    ok = True
                                    break
                            except Exception as e:
                                notes.append(
                                    f"{method} {root_url.raw_path.decode()} -> {e}"
                                )

                    # 3) If still not ok, attempt minimal MCP JSON-RPC list tools on configured URL
                    if not ok:
                        try:
                            payload = {
                                "jsonrpc": "2.0",
                                "id": "1",
                                "method": "tools/list",
                            }
                            r = http.post(
                                str(url),
                                json=payload,
                                headers={"Content-Type": "application/json"},
                            )
                            notes.append(
                                f"POST {url.raw_path.decode()} list -> {r.status_code}"
                            )
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
### Tool use policy (critical)

- At each step, call at most one tool.
- Do not emit multiple tool calls in a single assistant message.
- Always wait for the previous tool's result before deciding the next action.
- If no tool is needed, respond directly to the user.
- Prefer short, incremental actions; avoid speculative parallel calls.
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
if "checkpointer" not in st.session_state:
    print("Creating checkpointer...")
    st.session_state["checkpointer"] = InMemorySaver()

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


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _load_mcp_tools(server_url: str):
    client = MultiServerMCPClient(
        {
            "discovera": {
                "transport": "sse",
                "url": _ensure_trailing_slash(server_url),
            }
        }
    )
    return asyncio.run(client.get_tools())


def _system_message() -> SystemMessage:
    return SystemMessage(content=server_instructions)


def _human_message(prompt: str) -> HumanMessage:
    ctx = (st.session_state.get("context") or "").strip()

    return HumanMessage(
        content=f"""
    Context:
    {ctx}

    User Prompt:
    {prompt}
    """
    )


def stream_langgraph(prompt: str):
    server_url = st.session_state.get("mcp_server") or MCP_ADDRESS

    # Build tools and agent
    try:
        tools = _load_mcp_tools(server_url)
    except Exception as e:
        yield ("error", f"[MCP tools error] {e}")
        return

    try:
        model_id = f"openai:{OPENAI_MODEL}"
        agent = create_react_agent(
            model_id,
            tools=tools,
            prompt=prompt,
            version="v2",
            checkpointer=st.session_state["checkpointer"],
        )
    except Exception as e:
        yield ("error", f"[Agent init error] {e}")
        return

    inputs = {"messages": [_system_message(), _human_message(prompt)]}

    # Stream updates from the graph (agent thoughts + tool calls/results)
    def _stream_agent_updates():
        q: "queue.Queue[Any]" = queue.Queue()

        async def _produce():
            try:
                async for update in agent.astream(
                    inputs, config=LANGGRAPH_CONFIG, stream_mode="updates"
                ):
                    q.put(update)
            except Exception as exc:  # surface errors back to main thread
                q.put(("__error__", exc))
            finally:
                q.put(("__done__", None))

        def _runner():
            asyncio.run(_produce())

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        while True:
            item = q.get()
            if isinstance(item, tuple) and item[0] == "__error__":
                raise item[1]
            if isinstance(item, tuple) and item[0] == "__done__":
                break
            yield item

    try:
        for update in _stream_agent_updates():
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                messages_update = (
                    node_update.get("messages")
                    if isinstance(node_update, dict)
                    else None
                )
                if not messages_update:
                    continue
                for msg in messages_update:
                    # Agent messages ("thoughts"/content)
                    if isinstance(msg, AIMessage):
                        content_text = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        if content_text:
                            yield ("delta", content_text)
                        # Surface tool call intents
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if tool_calls:
                            calls_render = []
                            for call in tool_calls:
                                name = call.get("name")
                                args = call.get("args")
                                calls_render.append({"name": name, "args": args})
                            tool_calls_json = json.dumps(
                                calls_render, default=str, indent=2
                            )
                            preview_json = (
                                tool_calls_json
                                if len(tool_calls_json) <= 2000
                                else tool_calls_json[:2000] + "…"
                            )
                            yield (
                                "tool",
                                "Tool calls:\n```json\n" + preview_json + "\n```",
                            )
                    # Tool results
                    elif isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, "name", "tool") or "tool"
                        tool_text = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        preview = (
                            tool_text
                            if len(tool_text) <= 1000
                            else (tool_text[:1000] + "…")
                        )
                        yield (
                            "tool",
                            f"{tool_name} result:\n```\n{preview}\n```",
                        )
        # Signal completion
        yield ("done", None)
    except Exception as e:
        yield ("error", f"[Run error] {e}")


if prompt:
    # 1) append user to UI transcript and show it
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) stream from LangGraph agent
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_accum = ""
        for kind, payload in stream_langgraph(prompt):
            if kind == "delta" and payload:
                assistant_accum += payload
                placeholder.markdown(assistant_accum)
            elif kind == "error" and payload:
                st.markdown(str(payload))
            elif kind == "tool" and payload:
                st.markdown(str(payload))

    # 3) store assistant turn
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_accum}
    )
