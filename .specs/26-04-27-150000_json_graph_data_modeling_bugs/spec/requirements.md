# JSON Graph Data Modeling Issues - Requirements

**Spec ID:** 26-04-27-150000_json_graph_data_modeling_bugs
**Date:** 2026-04-27
**Status:** Approved
**Complexity:** Medium (Level 2)
**Layer:** Dynamics (Data Ingestion)

---

## Context

During JSON ingestion into the code-graph-rag system (port 7689), 29 entities and 37 relationships from the Critical Thinking Knowledge Base were processed. Analysis of the `json_ingestion.py` code reveals data modeling issues that can cause incorrect entity categorization, relationship edge label failures, ID collisions, and silent data corruption.

---

## Functional Requirements

| ID | Requirement | Priority | Rationale |
|----|-------------|:--------:|-----------|
| FR-1 | System MUST normalize relationship categories consistently using the 9 canonical categories (8 MECE + RELATED_TO fallback) | High | Ensures uniform relationship categorization across all JSON inputs |
| FR-2 | System MUST sanitize relationship type strings before using as Cypher edge labels | High | Prevents Cypher query failures from special characters/spaces |
| FR-3 | System MUST generate globally unique entity IDs that prevent collisions within a dataset | High | Ensures referential integrity when auto-generating IDs from names |
| FR-4 | System MUST warn when falling back to verb inference for missing category | Medium | Prevents silent mis-categorization |
| FR-5 | System MUST handle all Unicode emoji variants including ZWJ sequences | Low | Ensures emoji prefix extraction works for diverse inputs |
| FR-6 | System MUST deduplicate entity labels case-insensitively | Low | Prevents label pollution |

---

## Non-Functional Requirements

| ID | Requirement | Priority | Rationale |
|----|-------------|:--------:|-----------|
| NFR-1 | Fixes MUST be backward compatible with existing valid JSON inputs | High | Avoid breaking existing workflows |
| NFR-2 | Fixes MUST maintain O(n) ingestion performance for n entities | High | No quadratic complexity introduced |
| NFR-3 | Warnings MUST be actionable with specific remediation guidance | Medium | Users can fix issues without guessing |
| NFR-4 | Schema validation MUST preserve extensible properties (additionalProperties: true) | Low | Existing behavior: preserves extensibility for domain-specific fields; must not be changed |

---

## Issues Under Analysis

### Issue 1: Inconsistent Category Normalization
- **File:** `codebase_rag/json_ingestion.py` — `_normalize_relationship_category()`
- **Problem:** `_normalize_relationship_category()` has only 2 aliases mapped (`ATTRIBUTE`→`ATTRIBUTIVE`, `INFERRED`→`RELATED_TO`) but many common aliases are unmapped
- **Risk:** Input with `category: "CAUSES"` or `"PART_OF"` fails to normalize to canonical form

### Issue 2: Relationship Type Not Sanitized
- **File:** `codebase_rag/json_ingestion.py` — `ingest_relationships()` (relationship loop)
- **Problem:** `rel_type = str(relationship["relationship"])` used directly as edge label with only backtick escaping
- **Risk:** Spaces, hyphens, dots in relationship types may cause Cypher parse failures

### Issue 3: Entity ID Collisions
- **File:** `codebase_rag/json_ingestion.py` — `_canonical_entity_id()` and `_normalize_entities()`
- **Problem:** `_canonical_entity_id()` uses slug normalization that can collide; no collision avoidance for auto-generated IDs
- **Risk:** "Python 3.9" and "Python 3_9" generate same slug "python_3_9" when auto-generating IDs

### Issue 4: Silent Category Fallback
- **File:** `codebase_rag/json_ingestion.py` — `_relationship_properties()`
- **Problem:** Falls back to verb string when category missing, masking missing field
- **Risk:** Incorrect relationship categorization without user awareness

### Issue 5: Unicode Emoji Handling
- **File:** `codebase_rag/json_ingestion.py` — `_extract_emoji_prefix()`
- **Problem:** `_extract_emoji_prefix()` uses limited Unicode ranges; no ZWJ or skin-tone modifier support
- **Risk:** Extended emoji sequences like "👨‍🦲" leave partial emoji in names

### Issue 6: Duplicate Label Deduplication
- **File:** `codebase_rag/json_ingestion.py` — `_entity_labels()`
- **Problem:** `_entity_labels()` uses `dict.fromkeys()` which deduplicates exact matches only. Case variants (e.g., `"JobPosting"` and `"JOBPOSTING"`) are both retained
- **Risk:** Label explosion and query issues

---

## Issues Refracted or Skipped

| Issue | Reason |
|-------|--------|
| Issue 7 (MAGE PageRank) | Already handled gracefully in current code; returns 0 and logs informative message when MAGE unavailable |
| Issue 8 (Relationship Insertion Order) | Current code already validates references explicitly via `_validate_relationship_references` and `_lookup_entity_reference`; failures are logged, not silent |
| Issue 9 (Lenient Schema) | `additionalProperties: true` is intentional per schema design ("maximum flexibility"); only `name` is required by design, consistent with NFR-4 |

---

## Acceptance Criteria

| ID | Criterion | Verification Method |
|----|-----------|---------------------|
| AC-1 | JSON with `category: "ATTRIBUTE"` normalizes to `ATTRIBUTIVE` | Unit test |
| AC-2 | JSON with `category: "CAUSES"` normalizes to `CAUSAL` | Unit test |
| AC-3 | Relationship with type "has-property" creates valid edge label | Unit test sanitization |
| AC-4 | Two entities with names differing only in slug chars get unique IDs | Unit test collision check |
| AC-5 | Missing category field produces warning with verb fallback | Log analysis |
| AC-6 | ZWJ emoji "👨‍🦲" extracts correctly leaving clean name | Unit test |
| AC-7 | Labels ["JobPosting", "JOBPOSTING"] deduplicate to single label | Unit test case-insensitive dedup |

---

## Constraints

| ID | Constraint |
|----|------------|
| C-1 | Must not break existing valid JSON ingestion workflows |
| C-2 | Changes must be deployable without database migration |
| C-3 | Fixes must maintain current performance characteristics |
| C-4 | All existing unit tests must continue to pass |
