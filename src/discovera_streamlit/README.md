## Discovera Streamlit Chat UI

A lightweight Streamlit chat client for Discovera’s FastMCP agent. It lets you connect to a remote MCP server, add study context (and optional CSV/TXT uploads), and interact with a LangGraph-powered agent that performs enrichment, literature search, and gene-centric analyses.


### Key Features

- **MCP connection panel**: Enter the SSE endpoint and verify connectivity.
- **Context composer**: Add free-form study context; optionally upload CSV/TXT files stored in S3 with presigned download links.
- **Streaming responses**: Agent thoughts and tool events render in real time.
- **Tool event panels**: Foldable sections show tool call intents and results.
- **Copy view**: Toggle assistant messages between rich text and a copy-friendly view.


## Prerequisites

- Python 3.8+
- An OpenAI API key
- A reachable MCP server SSE endpoint (e.g., FastMCP)
- AWS S3 bucket credentials (required only for file uploads)


## Installation

From the repository root:

```bash
pip install -r src/discovera_streamlit/requirements.txt
```

Or inside this folder:

```bash
cd src/discovera_streamlit
pip install -r requirements.txt
```


## Configuration

Create a `.env` file alongside `app.py` (or export these variables in your shell):

```
# OpenAI
OPENAI_API_KEY=sk-proj-...
# Optional: defaults to gpt-4o-mini in code
OPENAI_MODEL=gpt-5-nano

# MCP Server (SSE endpoint)
MCP_SERVER_ADDRESS=https://discovera-fastmcp.users.hsrn.nyu.edu/sse
# For local/dev you can use an ngrok tunnel
# MCP_SERVER_ADDRESS=https://<your-ngrok-subdomain>.ngrok-free.app/sse

# Timeouts (optional)
MCP_TIMEOUT_MS=3500

# AWS S3 (required for file uploads via the sidebar)
AWS_REGION=us-east-2
AWS_ACCESS_KEY=AKIA...
AWS_SECRET_ACCESS_KEY=...
UPLOADS_S3_BUCKET=discovera
UPLOADS_S3_PREFIX=uploads
# Presigned URL TTL in seconds (default 604800 = 7 days)
S3_PRESIGN_TTL_SECONDS=604800
```

Notes:
- The app uses `AWS_ACCESS_KEY` and `AWS_SECRET_ACCESS_KEY` variable names (intentionally different from the standard `_ID` naming). These are passed to `boto3` under the hood.
- File uploads are optional. Without valid S3 credentials and bucket, the chat works but uploads are disabled.


## Run

From the repository root:

```bash
streamlit run src/discovera_streamlit/app.py
```

Or inside this folder:

```bash
cd src/discovera_streamlit
streamlit run app.py
```


## How to Use

1. **Check MCP connection**
   - Open the sidebar → Connection → verify or edit the MCP Server URL → click “Check MCP”.
   - You should see a toast and a status indicator (🟢/🔴).

2. **Add study context**
   - In the sidebar → Context, enter background, goals, datasets, and constraints.

3. **(Optional) Upload CSV/TXT files**
   - Drop CSV/TXT files in the uploader to include them in context.
   - Files are uploaded to S3 and shown with presigned download links; the agent receives those links in the prompt and can register/read them via csv tools.

4. **Chat with the agent**
   - Use the chat input at the bottom. The agent streams intermediate thoughts, tool calls, and results.
   - Tool calls and outputs appear as foldable panels. Click to expand for details.
   - Use “Copy view” on assistant messages if you need a plain-text snapshot.

5. **Iterate**
   - Refine your question, add more context, or upload additional files. The agent keeps an in-memory thread for the session.


## Tips and Troubleshooting

- **OPENAI_API_KEY missing**: The app will show an error and stop. Set it in `.env` or your shell and restart.
- **MCP connection fails**: Double-check the full SSE endpoint (often ends with `/sse`). Ensure the server is reachable and not blocked by firewalls.
- **Uploads not working**: Verify `AWS_REGION`, `UPLOADS_S3_BUCKET`, and credentials. Your IAM principal needs `s3:PutObject` and `s3:GetObject` for the bucket/prefix.
- **Large files**: Prefer CSV. The app creates short previews and presigned links; default expiry is 7 days, configurable via `S3_PRESIGN_TTL_SECONDS`.


## Advanced Configuration

- The system instructions sent to the agent are defined in `server_instructions` within `app.py`.
- The app uses LangGraph’s `create_react_agent` and an in-memory checkpointer. To change thread identifiers or agent behavior, edit `LANGGRAPH_CONFIG` and agent initialization in `app.py`.

