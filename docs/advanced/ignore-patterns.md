---
description: "Configure .cgrignore to exclude files or directories from Code-Graph-RAG analysis."
---

# Ignore Patterns

You can specify additional files or directories to exclude from analysis by creating a `.cgrignore` file in your repository root.

## Format

```
# Comments start with #
vendor
.custom_cache
my_build_output
docs/tree-sitter.txt
*.txt
*.pdf
*.egg-info/
```

## Rules

- One pattern per line
- Lines starting with `#` are comments
- Blank lines are ignored
- Patterns support exact paths, directory prefixes, and glob-style matches such as `*.txt` and `*.egg-info/`
- Prefixing a pattern with `!` unignores matching paths
- Patterns from `.cgrignore` are merged with `--exclude` flags and auto-detected directories
- The same ignore rules are applied consistently to code indexing and document indexing

## Default Exclusions

Code-Graph-RAG automatically excludes common non-source directories such as `.git`, `node_modules`, `__pycache__`, `dist`, `build`, and similar.
