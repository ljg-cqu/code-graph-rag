# Data Model & Ingestion Health Check Design Specification
## Revision: 1.0 | Date: 2024-xx-xx | Status: Implementation Ready

---

## 1. Overview
This document summarizes the results of a full health check of the ingested codebase, documents, and JSON datasets, identifies the single root cause issue found, and provides implementation-ready fixes.

---

## 2. Health Check Findings
### 2.1 Healthy Components (No Issues Found)
All components below passed all validation checks with 0 errors:
| Component | Validation Results |
|-----------|---------------------|
| **Code Graph** | No disconnected nodes, no missing required properties on all entities (Files, Functions, Classes, Modules), all tree-sitter parsed AST nodes have complete metadata, all relationships are correctly linked. |
| **Document Graph** | No disconnected nodes, no missing properties on Document/Section/Chunk entities, all document content is correctly parsed and linked between parent/child sections. |
| **Embedding Correlation** | No mismatches found between embedding vectors and their associated graph entities. Test entities are intentionally skipped for embedding generation as a standard optimization pattern for non-production code. |

### 2.2 Identified Critical Issue: JSON Ingestion Failure
#### Affected Asset: `sample_json_ingest.json`
#### Root Cause:
The sample JSON file uses non-standard field naming for relationships that does not match the official ingestion schema defined in `ingestion_schema.json`:
1. Sample uses `source_entity_id` instead of required schema field `source` for relationship source references
2. Sample uses `target_entity_id` instead of required schema field `target` for relationship target references
3. Sample uses `type` for relationship type instead of required schema field `relationship`

#### Impact:
No entities (`JobPosting`, `Skill`) or relationships (`REQUIRES_SKILL`) from the sample JSON file were ingested into the graph.

---

## 3. Fixing Approach
Two implementation options are provided, with Option 1 recommended for immediate resolution, and Option 2 recommended for long-term robustness.

### 3.1 Option 1: Immediate Minimal Fix (No Code Changes Required)
Update the sample JSON file to comply with the official ingestion schema. This is the fastest fix with zero risk of breaking existing functionality.

#### Implementation Steps:
1. Modify `sample_json_ingest.json` relationship objects:
   - Replace all `source_entity_id` keys with `source`
   - Replace all `target_entity_id` keys with `target`
   - Replace all `type` keys (within relationship objects) with `relationship`
2. Re-run JSON ingestion for the modified file
3. Validate successful ingestion:
   - Query returns 1 `JobPosting` entity (Senior Backend Engineer)
   - Query returns 2 `Skill` entities (Python, Neo4j)
   - Query returns 2 `REQUIRES_SKILL` relationships between the job posting and skills with all properties intact

### 3.2 Option 2: Long-Term Backward Compatibility Improvement
Add a field mapping layer to the JSON ingestion pipeline to support both the official schema and the field naming pattern used in the sample file, for better user experience with existing JSON datasets.

#### Implementation Steps:
1. Add field mapping logic in the JSON ingestion parser:
   - Automatically map `source_entity_id` → `source` if present
   - Automatically map `target_entity_id` → `target` if present
   - Automatically map `type` → `relationship` for relationship objects if present
2. Add a validation warning to notify users when deprecated field names are used, to encourage migration to the official schema
3. Update JSON ingestion documentation to mention both supported field naming conventions

---

## 4. Validation Criteria for Fix Success
After applying either fix:
1. Running the query `MATCH (j:JobPosting)-[r:REQUIRES_SKILL]->(s:Skill) RETURN j.name, s.name, r.minimum_experience_years` returns:
   | j.name                     | s.name | r.minimum_experience_years |
   |----------------------------|--------|-----------------------------|
   | Senior Backend Engineer    | Python | 3                           |
   | Senior Backend Engineer    | Neo4j  | 2                           |
2. All properties from the original JSON file are preserved on the ingested entities and relationships
3. No other ingestion workflows are broken by the change
