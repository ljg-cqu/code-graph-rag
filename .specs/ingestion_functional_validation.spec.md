# Functional Ingestion Validation & Fix Specification

## Core Validation Requirements (per User Request)
This specification covers validation of all functional ingestion components against the official `ingestion_schema.json` standard.

---

## 1. JSON Ingestion Validation & Fixes
### Current State
Official JSON ingestion schema exists and is enforced as single source of truth, with no transformations applied during ingestion.
### Validation Checks
| Check Item | Expected Result | Root Cause if Failed | Fix Approach |
|------------|-----------------|----------------------|--------------|
| JSON schema validation | All ingested JSON files conform to `ingestion_schema.json` requirements | Missing required fields (dataset_id in metadata, name in entities, source/target/relationship in relationships) | Add pre-ingestion JSON schema validation step that rejects invalid batches with clear error messages before writing to graph |
| Metadata propagation | Top-level metadata (dataset_id, source, default labels) is correctly applied to all entities/relationships in batch | Metadata propagation logic missing or incorrect | Add metadata merging logic that correctly combines batch-level and entity/relationship-level metadata |
| Nested JSON structure parsing | Nested JSON properties are correctly stored as node/relationship properties | Flattening logic corrupting nested properties | Remove all property transformation logic, store JSON properties exactly as provided per schema requirements |
| Idempotency | Re-ingesting the same batch does not create duplicate entities | Deduplication logic missing (based on dataset_id + entity ID/name) | Add deduplication step that checks for existing entities with same dataset_id and ID/name before creation |

---

## 2. Document Parsing & Ingestion Validation & Fixes
### Current State
Document ingestion is failing at embedding generation step due to SSL/API connectivity issues, as confirmed in earlier checks.
### Validation Checks
| Check Item | Expected Result | Root Cause if Failed | Fix Approach |
|------------|-----------------|----------------------|--------------|
| Format support | All supported document types (Markdown, PDF, DOCX, TXT) are correctly parsed | Unsupported file type handlers missing | Add validation for file types before parsing, return clear error for unsupported types |
| Section splitting | Documents are split into logical sections/chunks with correct hierarchical relationships (Document → Section → Chunk) | Splitter logic incorrectly splits mid-paragraph/section, loses hierarchy | Update splitter logic to respect document structure (Markdown headers, PDF outline, DOCX headings) when generating chunks |
| Source attribution | All chunks/sections have correct source path, page number (for PDF), and position offsets | Source metadata missing from chunk properties | Add mandatory source metadata to all document-derived nodes: source_file, start_offset, end_offset, page_number (where applicable) |
| Content completeness | No content is lost during parsing/splitting | Truncated content from large documents or non-standard formatting | Add content length validation that compares total chunk content length to original document content length (adjusted for formatting removal) |
| Embedding correlation | Every chunk node has a corresponding embedding vector in the vector store that matches the chunk content | Embedding generation failed, vector store entries not linked to graph nodes | Add post-ingestion validation that checks for embedding existence for every chunk node, and runs a semantic similarity check between chunk content and embedding to confirm correlation |

---

## 3. Tree-Sitter Code Parsing & Ingestion Validation & Fixes
### Validation Checks
| Check Item | Expected Result | Root Cause if Failed | Fix Approach |
|------------|-----------------|----------------------|--------------|
| Entity capture | All code entities (modules, files, classes, functions, methods, interfaces) are correctly captured with correct qualified names | Tree-sitter grammar rules missing for specific language constructs, qualified name generation logic incorrect | Add test suite for each supported language that verifies all common entity types are captured correctly |
| Call graph correctness | All function/method calls are correctly captured as `CALLS` relationships between caller and callee nodes | Reference resolution logic fails to resolve cross-file/cross-module calls | Improve import resolution logic to support relative imports, aliased imports, and dynamic references |
| AST mapping | Code nodes have correct start/end line numbers, source file path, and code snippet properties | AST node position extraction logic incorrect | Add validation that code snippet property matches the actual file content at the specified line numbers |
| Relationship correctness | Appropriate parent-child relationships exist (Module → File → Class → Method, File → Function) | Hierarchy relationship generation logic missing | Add mandatory hierarchy relationships for all code entities to ensure no disconnected production code nodes |
| Duplicate prevention | No duplicate code nodes exist for the same entity | Deduplication logic missing for code entities | Add deduplication based on qualified name + file path for code nodes |

---

## 4. Disconnected Node Validation
### Validation Rule
- **Expected disconnected nodes**: Only test-related entities (test files, test classes, test methods) are allowed to be disconnected
- **Abnormal disconnected nodes**: Any production code entity (module, file, class, function, method) with zero incoming and zero outgoing relationships is an ingestion error
### Fix Approach for Abnormal Disconnected Nodes
1. First check if entity is correctly captured: verify qualified name, file path, and existence in source code
2. Check if reference resolution logic failed to find calls/references to/from the entity
3. Check if hierarchy relationship generation logic failed to attach the entity to its parent (e.g. function not attached to parent file)
4. Fix the specific ingestion component that failed, re-ingest the affected files

---

## Validation Script Implementation
A runnable validation script `validate_ingestion.py` will be provided in the `.specs/` directory that implements all above checks, outputs a detailed validation report, and flags all issues with root cause and fix recommendations.
