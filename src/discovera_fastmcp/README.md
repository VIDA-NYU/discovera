## Discovera MCP Server

FastMCP server that exposes tools for biomedical knowledge discovery (e.g., querying INDRA, simple GSEA). It serves over SSE on port 8000 and can be connected to ChatGPT via the Model Context Protocol (MCP).

### Requirements
- Python 3.10+
- Environment variable: `OPENAI_API_KEY`
- Project dependencies: `pip install -r requirements.txt`

Optional:
- `.env` file at project root or alongside the server containing `OPENAI_API_KEY=...` (loaded via `python-dotenv`).

### Run locally
From the project root:
```bash
export OPENAI_API_KEY=your_key
python -m src.discovera_fastmcp.server
```

If you see `ModuleNotFoundError: No module named 'src'`, launch with an explicit `PYTHONPATH`:
```bash
PYTHONPATH="$PWD" python src/discovera_fastmcp/server.py
```

If you see `ModuleNotFoundError: No module named 'fastmcp'`:
```bash
pip install -r requirements.txt
```

### Docker
Build an image (Python 3.11-slim base, runs on port 8000):
```bash
docker build -f src/discovera_fastmcp/Dockerfile.fastmcp -t yourrepo/discovera:fastmcp-amd64 .
```

Push:
```bash
docker push yourrepo/discovera:fastmcp-amd64
```

### Kubernetes
Apply manifests from project root:
```bash
kubectl apply -f .kube/deployment.yaml # Update your docker path here
kubectl apply -f .kube/service.yaml
kubectl apply -f .kube/ingress.yaml
```

To roll out a new image without changing manifests, delete the current pod and let the Deployment recreate it:
```bash
kubectl get pods
kubectl delete pod <discovera-fastmcp-pod-name>
```

Note: Supply `OPENAI_API_KEY` via a Kubernetes `Secret` and reference it in the Deployment `env` section for production.

### Connect in ChatGPT
Follow OpenAI’s MCP guide to add a custom server:
[OpenAI MCP: Connect in ChatGPT](https://platform.openai.com/docs/mcp#connect-in-chatgpt)

Use the server URL `http://<host>:8000/sse` and the name `discovera_fastmcp`.

### Troubleshooting
- Missing API key
  - Error: `OpenAI API key not found. Please set OPENAI_API_KEY...`
  - Fix: Set `OPENAI_API_KEY` in your environment or `.env` file.

- `ModuleNotFoundError: No module named 'src'`
  - Run with module mode from project root: `python -m src.discovera_fastmcp.server`
  - Or: `PYTHONPATH="$PWD" python src/discovera_fastmcp/server.py`

- `ModuleNotFoundError: No module named 'fastmcp'`
  - Install deps: `pip install -r requirements.txt`

- Pydantic schema errors for DataFrame types
  - FastMCP/Pydantic prefer JSON-serializable types. If needed, convert DataFrames to `list[dict]` when returning, and accept lists of dicts as input.
