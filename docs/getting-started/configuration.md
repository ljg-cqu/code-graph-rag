---
description: "Configure Code-Graph-RAG with provider settings, environment variables, and model options."
---

# Configuration

Configuration is managed through environment variables in the `.env` file. The provider-explicit configuration supports mixing different providers for orchestrator and cypher models.

## Provider Examples

### All Ollama (Local Models)

```bash
ORCHESTRATOR_PROVIDER=ollama
ORCHESTRATOR_MODEL=llama3.2
ORCHESTRATOR_ENDPOINT=http://localhost:11434/v1

CYPHER_PROVIDER=ollama
CYPHER_MODEL=codellama
CYPHER_ENDPOINT=http://localhost:11434/v1
```

### All OpenAI Models

```bash
ORCHESTRATOR_PROVIDER=openai
ORCHESTRATOR_MODEL=gpt-4o
ORCHESTRATOR_API_KEY=sk-your-openai-key

CYPHER_PROVIDER=openai
CYPHER_MODEL=gpt-4o-mini
CYPHER_API_KEY=sk-your-openai-key
```

### All Google Models

```bash
ORCHESTRATOR_PROVIDER=google
ORCHESTRATOR_MODEL=gemini-2.5-pro
ORCHESTRATOR_API_KEY=your-google-api-key

CYPHER_PROVIDER=google
CYPHER_MODEL=gemini-2.5-flash
CYPHER_API_KEY=your-google-api-key
```

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### All Anthropic Models

```bash
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=claude-3-5-sonnet-latest
ORCHESTRATOR_API_KEY=sk-ant-api03-your-anthropic-key

CYPHER_PROVIDER=anthropic
CYPHER_MODEL=claude-3-haiku-latest
CYPHER_API_KEY=sk-ant-api03-your-anthropic-key
```

Get your Anthropic API key from [Anthropic Console](https://console.anthropic.com/settings/keys).

### Anthropic-Compatible Endpoints

For custom Anthropic-compatible endpoints (e.g., Baidu Qianfan, vLLM, or other proxies):

```bash
ORCHESTRATOR_PROVIDER=anthropic
ORCHESTRATOR_MODEL=GLM-5
ORCHESTRATOR_API_KEY=your-custom-api-key
ORCHESTRATOR_ENDPOINT=https://qianfan.baidubce.com/anthropic/coding
```

### Mixed Providers

```bash
ORCHESTRATOR_PROVIDER=google
ORCHESTRATOR_MODEL=gemini-2.5-pro
ORCHESTRATOR_API_KEY=your-google-api-key

CYPHER_PROVIDER=ollama
CYPHER_MODEL=codellama
CYPHER_ENDPOINT=http://localhost:11434/v1
```

## Orchestrator Model Settings

| Variable | Description |
|----------|-------------|
| `ORCHESTRATOR_PROVIDER` | Provider name (`google`, `openai`, `ollama`, `anthropic`) |
| `ORCHESTRATOR_MODEL` | Model ID (e.g., `gemini-2.5-pro`, `gpt-4o`, `llama3.2`) |
| `ORCHESTRATOR_API_KEY` | API key for the provider (if required) |
| `ORCHESTRATOR_ENDPOINT` | Custom endpoint URL (if required) |
| `ORCHESTRATOR_PROJECT_ID` | Google Cloud project ID (for Vertex AI) |
| `ORCHESTRATOR_REGION` | Google Cloud region (default: `us-central1`) |
| `ORCHESTRATOR_PROVIDER_TYPE` | Google provider type (`gla` or `vertex`) |
| `ORCHESTRATOR_THINKING_BUDGET` | Thinking budget for reasoning models |
| `ORCHESTRATOR_SERVICE_ACCOUNT_FILE` | Path to service account file (for Vertex AI) |

## Cypher Model Settings

| Variable | Description |
|----------|-------------|
| `CYPHER_PROVIDER` | Provider name (`google`, `openai`, `ollama`, `anthropic`) |
| `CYPHER_MODEL` | Model ID (e.g., `gemini-2.5-flash`, `gpt-4o-mini`, `codellama`) |
| `CYPHER_API_KEY` | API key for the provider (if required) |
| `CYPHER_ENDPOINT` | Custom endpoint URL (if required) |
| `CYPHER_PROJECT_ID` | Google Cloud project ID (for Vertex AI) |
| `CYPHER_REGION` | Google Cloud region (default: `us-central1`) |
| `CYPHER_PROVIDER_TYPE` | Google provider type (`gla` or `vertex`) |
| `CYPHER_THINKING_BUDGET` | Thinking budget for reasoning models |
| `CYPHER_SERVICE_ACCOUNT_FILE` | Path to service account file (for Vertex AI) |

## System Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMGRAPH_HOST` | `localhost` | Memgraph hostname |
| `MEMGRAPH_PORT` | `7687` | Memgraph port |
| `MEMGRAPH_HTTP_PORT` | `7444` | Memgraph HTTP port |
| `LAB_PORT` | `3000` | Memgraph Lab port |
| `MEMGRAPH_BATCH_SIZE` | `1000` | Batch size for Memgraph operations |
| `TARGET_REPO_PATH` | `.` | Default repository path |
| `LOCAL_MODEL_ENDPOINT` | `http://localhost:11434/v1` | Fallback endpoint for Ollama |
| `DEFAULT_CONTEXT_WINDOW` | `256000` | Global default context window (fallback when model-specific not detected) |
| `PYTHON_INSPECT_TIMEOUT` | `10` | Timeout for Python object inspection operations (seconds) |
| `SHELL_COMMAND_TIMEOUT` | `30` | Timeout for shell command execution (seconds) |

## Model Catalog Management

Configure dynamic model discovery and external catalog files:

| Variable | Default | Description |
|----------|---------|-------------|
| `CGR_MODEL_CATALOG_PATH` | `None` | Path to external JSON/YAML model catalog file (overrides static catalog) |
| `CGR_DISABLE_MODEL_DISCOVERY` | `False` | Disable dynamic model discovery from `.env` configuration |

**Note**: Role-configured models (ORCHESTRATOR_*, CYPHER_*, CGR_WORKER_LLMS) automatically appear in `/models` command output with configuration status indicators.

## Document GraphRAG Settings

Configure the separate document graph for indexing and querying documentation:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOC_MEMGRAPH_HOST` | `localhost` | Document Memgraph hostname |
| `DOC_MEMGRAPH_PORT` | `7688` | Document Memgraph port |
| `DOC_MEMGRAPH_VECTOR_INDEX_NAME` | `doc_embeddings` | Vector index name for document embeddings |
| `DOC_MEMGRAPH_VECTOR_CAPACITY` | `100000` | Vector index capacity |
| `DOC_SUPPORTED_EXTENSIONS` | `.md,.rst,.txt,.pdf,.docx` | Supported document extensions |
| `DOC_MAX_FILE_SIZE_MB` | `50` | Maximum document file size |
| `DOC_ENABLED` | `True` | Enable/disable document indexing |

## Setting Up Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh

ollama pull llama3.2
# Or try other models:
# ollama pull llama3
# ollama pull mistral
# ollama pull codellama
```

Ollama automatically starts serving on `localhost:11434`.

!!! note
    Local models provide privacy and no API costs, but may have lower accuracy compared to cloud models like Gemini or GPT-4o.

## Programmatic Configuration

You can also configure providers programmatically via the Python SDK:

```python
from cgr import settings

settings.set_orchestrator("openai", "gpt-4o", api_key="sk-...")
settings.set_cypher("google", "gemini-2.5-flash", api_key="your-key")
```
