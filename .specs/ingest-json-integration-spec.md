# JSON Ingestion Integration Design Spec for `cgr start` Command
## Version: 1.1 | Status: Revised for Parallel Worker Support | Minimal Changes Only

### Objective
Add optional automatic JSON ingestion to the `cgr start` command when running document indexing (`--index-docs` or `--index-all` flags), with zero breaking changes, native 10-parallel-worker support, and full schema compliance.

### Requirements Met
1. **Optional Execution**: JSON ingestion is disabled by default, explicitly enabled via new CLI flag
2. **Schema Validation**: All JSON files are validated against existing `/ingestion_schema.json` automatically, invalid files are skipped with warning
3. **Flexible Input**: Users can specify a custom JSON file/directory path, or default to scanning the repo root for `*.json` files
4. **Parallel Safe**: Native support for 10 parallel workers with concurrency controls, no duplicate ingestion or database overload
5. **Minimal Changes**: No new files required, only modifications to existing `cli.py` and minimal reuse of existing `ingest_json_data` function

---

## Issues Found & Fixed During Parallel Worker Review
| Valid Issue | Fix Applied |
|-------------|-------------|
| No concurrency control for 10 parallel workers → risk of duplicate entries/DB connection exhaustion | Added connection pooling + per-file locking + worker count limit |
| No schema caching → 10x redundant reads of ingestion_schema.json during parallel validation | Added one-time schema caching before worker spawn |
| No workspace association → ingested JSON not linked to active `--doc-workspace`, breaks multi-workspace isolation | Added workspace ID propagation to all ingested JSON entities |
| No aggregated error reporting → errors from parallel workers are scattered | Added consolidated error aggregation across all workers |
| No file-level locking → parallel workers can process same JSON file multiple times | Added advisory file locks during JSON processing |

---

## Implementation Plan

### 1. New CLI Flags for `start` Command (Add to existing `start` function parameters)
```python
# Add these 4 new flags to the start() command parameters, no other changes to existing flags
ingest_json: bool = typer.Option(
    False,
    "--ingest-json",
    help="Enable automatic JSON ingestion during document indexing (validates against ingestion_schema.json)",
),
json_path: str | None = typer.Option(
    None,
    "--json-path",
    help="Path to specific JSON file or directory to ingest (defaults to repo root scanning for *.json if not provided)",
),
json_skip_invalid: bool = typer.Option(
    True,
    "--json-skip-invalid/--json-fail-on-invalid",
    help="Skip invalid JSON files (default) or fail ingestion if any JSON is invalid",
),
json_parallel_workers: int = typer.Option(
    10,
    "--json-workers",
    min=1,
    max=32,
    help="Number of parallel workers for JSON ingestion (default: 10)",
),
```

### 2. Integration Point in `_handle_indexing` Function
Add JSON ingestion logic **immediately after successful document indexing** (inside the `if effective_index_docs:` block, right after displaying document indexing stats):
```python
# === JSON Ingestion (Optional, Parallel Worker Support) ===
if ingest_json:
    from codebase_rag.json_ingestion import ingest_json_data
    _info(style(f"Running optional JSON ingestion with {json_parallel_workers} parallel workers...", cs.Color.CYAN))
    
    # Resolve JSON input path
    target_json_path = json_path or str(repo_path)
    
    try:
        ingest_result = ingest_json_data(
            input_path=target_json_path,
            skip_existing=True,
            batch_size=batch_size,
            incremental=True,
            dry_run=False,
            parallel_workers=json_parallel_workers,
            # Pass active document workspace ID for isolation
            metadata_override={"workspace": doc_workspace}
        )
        
        # Show consolidated summary across all workers
        table = Table(
            title=style("JSON Ingestion Results", cs.Color.GREEN),
            show_header=True,
            header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
        )
        table.add_column("Metric", style=cs.Color.CYAN)
        table.add_column("Count", style=cs.Color.YELLOW, justify="right")
        table.add_row("Valid JSON files processed", str(ingest_result.files_processed))
        table.add_row("Invalid JSON files skipped", str(ingest_result.files_skipped))
        table.add_row("Entities ingested", str(ingest_result.entities_ingested))
        table.add_row("Relationships ingested", str(ingest_result.relationships_ingested))
        table.add_row("Parallel workers used", str(json_parallel_workers))
        app_context.console.print(table)
        
        # Show aggregated errors if any
        if ingest_result.errors:
            _info(style(f"Ingestion completed with {len(ingest_result.errors)} errors:", cs.Color.YELLOW))
            for error in ingest_result.errors[:15]:
                _info(style(f"  - {error}", cs.Color.RED))
            if len(ingest_result.errors) > 15:
                _info(style(f"  ... and {len(ingest_result.errors) - 15} more errors", cs.Color.RED))
            
            if not json_skip_invalid:
                raise ValueError(f"JSON ingestion failed (--json-fail-on-invalid enabled)")
            
    except Exception as e:
        _info(style(f"JSON ingestion failed: {e}", cs.Color.RED))
        if not json_skip_invalid:
            raise typer.Exit(1) from e
```

### 3. Minor Modification to Existing `ingest_json_data` Function (Only 2 changes, no rewrite)
1. Add return fields `files_processed` and `files_skipped` to the ingestion result to track how many JSON files were validated/skipped against the schema
2. Add 2 new optional parameters: `parallel_workers` (default 1) and `metadata_override` (default None) to support parallel execution and workspace association
3. Add one-time schema load/caching before spawning workers, add per-file advisory locking to prevent duplicate processing

---

## Usage Examples
### 1. Default: Auto-scan repo for valid JSON when indexing docs with 10 parallel workers
```bash
cgr start --index-all --ingest-json
```
- Scans repo root for all *.json files
- Validates each against `/ingestion_schema.json`
- Skips invalid files with warning
- Uses 10 parallel workers for fast ingestion
- Links all ingested data to active document workspace

### 2. Specify custom JSON file to ingest with reduced worker count
```bash
cgr start --index-docs --ingest-json --json-path ./custom_data/my_entities.json --json-workers 2
```
- Only ingests the specified JSON file
- Uses 2 parallel workers for small datasets
- Fails immediately if file is invalid if `--json-fail-on-invalid` is added

### 3. Fail on invalid JSON
```bash
cgr start --index-all --ingest-json --json-fail-on-invalid
```
- Exits with error code if any JSON file fails schema validation

---

## Validation Rules (Reuse Existing Logic, No New Code)
1. All JSON files are validated against `/ingestion_schema.json` using existing jsonschema validation in `json_ingestion.py`
2. Invalid JSON files are skipped by default, warning logged to console
3. JSON files without required `entities` field are automatically skipped
4. Deduplication and conflict resolution uses existing default `last-write-wins` strategy, same as standalone `ingest-json` command
5. Workspace ID is automatically added to all ingested entities/relationships for multi-workspace isolation

---

## Backward Compatibility
1. No breaking changes to existing `start` command functionality, JSON ingestion is disabled by default
2. All existing flags and behavior remain unchanged
3. No new dependencies required, uses existing `ingest_json_data` function and schema validation
4. No additional files created, only modifications to existing `cli.py` and minor update to ingestion result object
5. Parallel worker support is optional, defaults to 10 but can be adjusted for smaller systems
