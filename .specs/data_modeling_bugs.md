# Data Modeling Bugs & Issues — code-graph-rag JSON Graph Ingestion

## Executive Summary

The JSON ingestion from `spec.json` (critical-thinking-knowledge-base-v3.1.0) completed successfully (29 entities, 37 relationships) but contained **entity category mapping errors** that misclassify cognitive concepts. The root cause was the `ENTITY_SUBTYPE_REGISTRY` in `concept_extraction.py` using incorrect MECE category assignments for pedagogical/cognitive entity types.

**Status**: Issues 1, 3, and 4 have been resolved. Issue 2 is a medium-term recommendation.

---

## Issue 1: ENTITY_SUBTYPE_REGISTRY Has Wrong Category Mappings ✅ FIXED

**File:** `codebase_rag/document/concept_extraction.py` (line ~1256)

### Original Incorrect Mappings

| Entity Type | Original Category | Corrected Category | Rationale |
|-------------|-----------------|-------------------|-----------|
| `Cognitive Mechanism` | `SYSTEM_STRUCTURE` | `EVENT_PROCESS` | Cognitive mechanisms unfold over time (e.g., System 1, System 2) |
| `Conceptual Framework` | `SYSTEM_STRUCTURE` | `ABSTRACT_CONCEPT` | Frameworks are abstract ideas, not concrete structures |
| `Decision Framework` | `SYSTEM_STRUCTURE` | `ABSTRACT_CONCEPT` | Mental models are abstract conceptual tools |
| `Educational Framework` | `SYSTEM_STRUCTURE` | `ABSTRACT_CONCEPT` | Pedagogical frameworks are abstract schemas |
| `Research Framework` | `SYSTEM_STRUCTURE` | `ABSTRACT_CONCEPT` | Research models are abstract representations |
| `Framework Component` | `SYSTEM_STRUCTURE` | `INFORMATION_EXPRESSION` | Components are expressive parts (elements, standards, virtues) |
| `Technology Tool` | `CONCRETE_ENTITY` | `INFORMATION_EXPRESSION` | Generative AI in this context is an information modality |

### Root Cause Analysis

The `resolve_entity_category()` function (line 1526) follows a 3-step resolution:
1. Use `declared_category` if valid
2. Look up `declared_subtype` in `ENTITY_SUBTYPE_REGISTRY`
3. Fall back to `ABSTRACT_CONCEPT`

Since spec.json entities lacked an `entity_category` field, they relied on the registry. The registry mappings violated MECE principles:

- **SYSTEM_STRUCTURE** is for organized collections with structural/architectural properties (networks, hierarchies, pipelines)
- Frameworks and cognitive mechanisms are **ABSTRACT_CONCEPT** or **EVENT_PROCESS**

### Fix Applied

The `ENTITY_SUBTYPE_REGISTRY` was reorganized into category-specific sections:

```python
# 💡 Abstract Concept — Frameworks (4)
"Conceptual Framework": "ABSTRACT_CONCEPT",
"Decision Framework": "ABSTRACT_CONCEPT",
"Educational Framework": "ABSTRACT_CONCEPT",
"Research Framework": "ABSTRACT_CONCEPT",

# ⏱️ Event/Process — Cognitive Mechanisms (1)
"Cognitive Mechanism": "EVENT_PROCESS",

# 📨 Information/Expression — Framework Components & Tools (2)
"Framework Component": "INFORMATION_EXPRESSION",
"Technology Tool": "INFORMATION_EXPRESSION",
```

**Note on `FrameworkComponent` (no space):** This is the JSON camelCase variant of `"Framework Component"`, not a duplicate. Both keys coexist to handle the LLM concept extractor (spaced) and JSON ingestion (camelCase) data sources. Both were corrected to `INFORMATION_EXPRESSION`.

---

## Issue 2: spec.json Missing entity_category Field

**File:** External dataset (`spec.json`)

### Problem

Entities in spec.json rely solely on the `type`→`category` mapping via `ENTITY_SUBTYPE_REGISTRY`. The `entity_category` field is not present, so:
- No override is possible
- All categorization depends entirely on the registry
- Entity types not in the registry fall back to `ABSTRACT_CONCEPT`

### Fix Applied

`_entity_properties()` in `json_ingestion.py` now reads `entity_category` from the JSON entity when present and passes it to `resolve_entity_category()` as the declared category. This makes the ingestion pipeline forward-compatible with explicit category overrides.

### Recommendation

Add `entity_category` field to each entity in spec.json for explicit control:

```json
{
  "id": "entity-002",
  "name": "System 1",
  "type": "Cognitive Mechanism",
  "entity_category": "EVENT_PROCESS",
  "properties": { ... }
}
```

This makes categorization explicit and independent of registry changes.

---

## Issue 3: Relationship Verb→Category Inference Incomplete ✅ FIXED

**File:** `codebase_rag/json_ingestion.py` (line ~263)

### Actual Behavior (Before Fix)

The `_relationship_properties()` function uses the `relationship` verb as a fallback when `category` is missing:

```python
raw_category = str(
    relationship.get("category")
    or original_verb  # ← verb IS used as fallback
    or "RELATED_TO"
)
normalized_category = _normalize_relationship_category(raw_category)
```

However, the `_normalize_relationship_category()` mapping was incomplete — many verbs like `comprises`, `similar-to`, `contrasts-with`, `precedes`, etc. were not in the mapping and fell through to `RELATED_TO`.

### Fix Applied

The `_normalize_relationship_category()` mapping was expanded to include all common relationship verbs:

```python
mapping = {
    # Causal
    "CAUSES": "CAUSAL", "PRODUCES": "CAUSAL", "PREVENTS": "CAUSAL",
    "ENABLES": "CAUSAL", "INHIBITS": "CAUSAL", "RESULTS-IN": "CAUSAL",
    "REDUCES": "CAUSAL", "DEPENDS-ON": "CAUSAL", "BLOCKS": "CAUSAL",
    "DRIVES": "CAUSAL", "FACILITATES": "CAUSAL", "PROMOTES": "CAUSAL",
    "CORRELATES-WITH": "CAUSAL", "LEADS-TO": "CAUSAL", "MITIGATES": "CAUSAL",
    "TRIGGERS": "CAUSAL", "INFLUENCES": "CAUSAL",
    # Compositional
    "COMPRISES": "COMPOSITIONAL", "CONTAINS": "COMPOSITIONAL",
    "PART_OF": "COMPOSITIONAL", "PARTOF": "COMPOSITIONAL",
    # Hierarchical
    "IS_A": "HIERARCHICAL", "ISA": "HIERARCHICAL",
    "TYPE_OF": "HIERARCHICAL", "KIND_OF": "HIERARCHICAL",
    # Sequential
    "PRECEDES": "SEQUENTIAL", "FOLLOWS": "SEQUENTIAL",
    "PRECEDED-BY": "SEQUENTIAL", "FOLLOWED-BY": "SEQUENTIAL",
    # Comparative
    "SIMILAR-TO": "COMPARATIVE", "CONTRASTS-WITH": "COMPARATIVE",
    "RELATES-TO": "COMPARATIVE", "SIMILAR": "COMPARATIVE",
    # ... and more
}
```

This aligns the normalization mapping with the `VERB_REGISTRY` in `concept_extraction.py`.

---

## Issue 4: Cypher GROUP BY Query Syntax Error

**File:** code_graph_query agent testing

### Problem

This query fails in Memgraph:
```cypher
MATCH ()-[r]->() RETURN type(r), r.category, count(*) as cnt GROUP BY type(r), r.category
```

Memgraph doesn't support `GROUP BY` in this position. Use `WITH` + `RETURN`:

```cypher
MATCH ()-[r]->()
WITH type(r) as rel_type, r.category as cat, count(*) as cnt
RETURN rel_type, cat, cnt
ORDER BY cnt DESC
```

This is a query issue, not a code bug.

---

## Fix Sequence (Completed)

1. **✅ Done**: Fix `ENTITY_SUBTYPE_REGISTRY` category mappings — 7 type entries corrected (plus JSON camelCase variant)
2. **⏳ Medium-term**: Add `entity_category` field to spec.json for explicit control
3. **✅ Done**: Expand `_normalize_relationship_category()` verb mapping
4. **ℹ️ Informational**: Cypher query syntax — not a code issue

---

## Verification Query

After fixes, verify with:

```cypher
MATCH (n:JsonEntity)
RETURN n.entity_id, n.name, n.type, n.entity_category
ORDER BY n.entity_category, n.name
```

### How Categorization Works

Resolution is **type-based**, not name-based. The `type` field in spec.json is looked up in `ENTITY_SUBTYPE_REGISTRY`. Entity names are shown below for reference only.

**Corrected registry mappings for the critical-thinking dataset:**

| `type` (registry key) | Resolved Category | Example Entity Names |
|-----------------------|-------------------|---------------------|
| `Cognitive Mechanism` | `EVENT_PROCESS` | System 1, System 2 |
| `Conceptual Framework` | `ABSTRACT_CONCEPT` | Paul-Elder Framework |
| `Decision Framework` | `ABSTRACT_CONCEPT` | Munger Mental Models |
| `Educational Framework` | `ABSTRACT_CONCEPT` | Bloom's Taxonomy |
| `Research Framework` | `ABSTRACT_CONCEPT` | Facione Delphi Study |
| `Framework Component` | `INFORMATION_EXPRESSION` | Elements of Thought, Intellectual Standards, Intellectual Virtues |
| `Technology Tool` | `INFORMATION_EXPRESSION` | Generative AI |

**Note:** Entity names not listed above (e.g., "Critical Thinking", "Cognitive Bias", "Working Memory") resolve based on their own `type` field. If their `type` is absent from `ENTITY_SUBTYPE_REGISTRY` and no `entity_category` override is present, they fall back to `ABSTRACT_CONCEPT`.

---

## Files Modified

- `codebase_rag/document/concept_extraction.py` — `ENTITY_SUBTYPE_REGISTRY` corrections + docstring consistency (Working Memory, Educational Framework)
- `codebase_rag/json_ingestion.py` — `_normalize_relationship_category()` expanded verb mapping + `_entity_properties()` now reads `entity_category` override from JSON
- `codebase_rag/tests/test_entity_taxonomy.py` — `test_registry_has_expected_size` (200 sub-types), 8 new tests for corrected registry mappings
- `codebase_rag/tests/test_json_ingestion_enhancements.py` — 29 new verb mapping tests + 2 tests for `entity_category` override
