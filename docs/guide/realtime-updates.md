---
description: "Keep your Code-Graph-RAG knowledge graph synchronized with code changes using the real-time file watcher."
---

# Real-Time Graph Updates

For active development, keep your knowledge graph automatically synchronized with code changes using the real-time updater.

## What It Does

- Watches your repository for file changes (create, modify, delete)
- Automatically updates the knowledge graph in real-time
- Maintains consistency by recalculating all function call relationships
- Filters out irrelevant files (`.git`, `node_modules`, etc.)

## Usage

Run the real-time updater in a separate terminal:

```bash
python realtime_updater.py /path/to/your/repo
```

Or using the Makefile:

```bash
make watch REPO_PATH=/path/to/your/repo
```

### With Custom Memgraph Settings

```bash
python realtime_updater.py /path/to/your/repo \
  --host localhost --port 7687 --batch-size 1000
```

```bash
make watch REPO_PATH=/path/to/your/repo HOST=localhost PORT=7687 BATCH_SIZE=1000
```

## Multi-Terminal Workflow

```bash
# Terminal 1: Start the real-time updater
python realtime_updater.py ~/my-project

# Terminal 2: Run the AI assistant
cgr start --repo-path ~/my-project
```

## Environment Configuration

Configure default behavior via environment variables:

```bash
# Enable realtime updates by default
CGR_REALTIME_UPDATER=true

# Set debounce delay (default: 5.0 seconds)
CGR_REALTIME_DEBOUNCE=10.0

# Set max wait time (default: 30.0 seconds)
CGR_REALTIME_MAX_WAIT=60.0

# Enable/disable specific file types
CGR_REALTIME_CODE=true      # Watch code files (default: true)
CGR_REALTIME_DOCS=false     # Watch document files (default: false)
CGR_REALTIME_JSON=false     # Watch JSON files (default: false)
```

CLI flags override environment settings:

```bash
# Override env config for this session
cgr start --realtime-debounce 2.0 --realtime-docs
```

## CLI Arguments

Defaults can be configured via environment variables (see above).

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `repo_path` | Yes | | Path to repository to watch |
| `--host` | No | `localhost` | Memgraph host |
| `--port` | No | `7687` | Memgraph port |
| `--batch-size` | No | `CGR_REALTIME_BATCH_SIZE` (default: 100) | Number of buffered nodes/relationships before flushing to Memgraph |

## Performance Note

The updater currently recalculates all CALLS relationships on every file change to ensure consistency. This prevents "island" problems where changes in one file aren't reflected in relationships from other files, but may impact performance on very large codebases with frequent changes. Optimization of this behavior is a work in progress.
