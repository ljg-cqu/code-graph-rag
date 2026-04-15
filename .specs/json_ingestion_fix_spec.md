# JSON Ingestion Bug Fix Specification

## Problem Statement
When running `uv run cgr start --repo-path <path> --ingest-json --json-path <file.json>`, the JSON data is not ingested into Memgraph, and the UI reports "Memgraph is empty" when accessing `http://localhost:3002/lab/query`.

## Root Causes
### 1. Ingestion Execution Block
JSON ingestion logic in `cli.py` `_handle_indexing` function is **nested inside the `if effective_index_docs:` block**, so it only runs if the user passes `--index-docs` or `--index-all` flags, even if `--ingest-json` is explicitly enabled.

### 2. Separate JSON Memgraph Instance
JSON ingestion writes to a **dedicated, isolated Memgraph instance** (using `JSON_MEMGRAPH_HOST`/`JSON_MEMGRAPH_PORT` config) that is separate from the main code and document Memgraph instances. The frontend UI only queries the main code Memgraph instance, so it never sees JSON data.

### 3. Missing Docker Configuration (Potential)
The default `docker-compose.yaml` does not include a service definition for the dedicated JSON Memgraph instance, so it may not be running at all, leading to silent ingestion failures.

## Fix Requirements
### Fix 1: Decouple JSON Ingestion from Document Indexing
Modify the `_handle_indexing` function in `cli.py` to run JSON ingestion **independently** of document indexing when `--ingest-json` flag is passed:
- Add check for `ingest_json` flag at the top level of `_handle_indexing`
- Execute JSON ingestion even if `effective_index_docs` is False
- Auto-initialize JSON Memgraph connection and schema/indices when ingestion runs

### Fix 2: Add JSON Instance Support to Frontend & Query Router
Update the query router and UI to support querying the JSON Memgraph instance:
- Add new `query_mode` option: `JSON_ONLY`
- Add new `--json` flag to query commands to target JSON instance
- Update UI to allow selecting JSON data source and displaying JSON entities/relationships

### Fix 3: Add JSON Memgraph Service to Docker Compose
Update `docker-compose.yaml` to include the JSON Memgraph service:
```yaml
  json-memgraph:
    image: memgraph/memgraph-mage:latest
    container_name: code-graph-rag-json-memgraph
    ports:
      - "7689:7687"
      - "7445:7444"
    volumes:
      - json-memgraph-data:/var/lib/memgraph
    environment:
      - MEMGRAPH_AUTH_ENABLED=false
      - MEMGRAPH_QUERY_LOG_LEVEL=WARNING
    networks:
      - code-graph-rag-network
```
- Add corresponding volume definition: `json-memgraph-data:`
- Update `.env.example` with default JSON Memgraph config:
  ```
  JSON_MEMGRAPH_HOST=localhost
  JSON_MEMGRAPH_PORT=7689
  JSON_MEMGRAPH_USERNAME=
  JSON_MEMGRAPH_PASSWORD=
  JSON_MEMGRAPH_BATCH_SIZE=100
  JSON_MEMGRAPH_VECTOR_INDEX_NAME=json_entity_embeddings
  JSON_MEMGRAPH_VECTOR_CAPACITY=1000000
  JSON_PARALLEL_WORKERS=10
  ```

### Fix 4: Add Ingestion Result Output
Ensure that when `cgr start` runs with `--ingest-json`, it prints clear output showing:
- Number of JSON files processed
- Number of entities/relationships ingested
- Any errors or failures
- Confirmation that data is written to JSON Memgraph instance

## Immediate Workaround for Users
To get ingestion working immediately with current code:
1. Run the command with `--index-docs` flag to trigger the ingestion block:
   ```bash
   uv run cgr start --repo-path <path> --ingest-json --json-path <file.json> --index-docs
   ```
2. To query JSON data, use the dedicated `cgr ingest-json` command and query the JSON Memgraph instance directly on port 7689.

## Validation Steps
1. Run `uv run cgr start --repo-path . --ingest-json --json-path sample_json_ingest.json` without `--index-docs`
2. Verify JSON ingestion runs and outputs results
3. Verify data exists in JSON Memgraph instance
4. Verify UI can query and display JSON data when JSON source is selected
5. Verify docker-compose starts all 3 Memgraph instances (code, document, json) correctly
