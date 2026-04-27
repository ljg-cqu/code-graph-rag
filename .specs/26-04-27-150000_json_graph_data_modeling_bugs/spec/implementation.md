# JSON Graph Data Modeling Issues - Implementation

**Spec ID:** 26-04-27-150000_json_graph_data_modeling_bugs
**Date:** 2026-04-27
**Status:** Approved
**Layer:** Dynamics (Data Ingestion)

---

## Implementation Details

### Issue 1: Category Normalization Fix

**Location:** `codebase_rag/json_ingestion.py` - `_normalize_relationship_category()`

**Fix:** Expand normalization alias map to handle common synonyms. Use `cs.DOC_CONCEPT_CATEGORIES` for validation instead of hardcoded set. Log warning via `logger` (loguru) for unknown categories.

```python
def _normalize_relationship_category(category: str) -> str:
    if not category or not category.strip():
        return "RELATED_TO"

    category = category.upper().strip()
    mapping = {
        "ATTRIBUTE": "ATTRIBUTIVE",
        "ATTRIBUTES": "ATTRIBUTIVE",
        "INFERRED": "RELATED_TO",
        "INFERENCE": "RELATED_TO",
        "CAUSES": "CAUSAL",
        "CAUSE": "CAUSAL",
        "PART_OF": "COMPOSITIONAL",
        "PARTOF": "COMPOSITIONAL",
        "IS_A": "HIERARCHICAL",
        "ISA": "HIERARCHICAL",
        "INSTANCEOF": "HIERARCHICAL",
        "SIMILAR": "COMPARATIVE",
        "COMPARABLE": "COMPARATIVE",
        "PRECEDES": "SEQUENTIAL",
        "FOLLOWS": "SEQUENTIAL",
        "ENABLES": "CAUSAL",
        "TRIGGERS": "CAUSAL",
        "INFLUENCES": "CAUSAL",
    }
    result = mapping.get(category, category)

    if result not in cs.DOC_CONCEPT_CATEGORIES:
        logger.warning(
            ls.JSON_UNKNOWN_CATEGORY.format(
                category=category,
                expected=", ".join(sorted(cs.DOC_CONCEPT_CATEGORIES)),
            )
        )
        return "RELATED_TO"

    return result
```

---

### Issue 2: Relationship Type Sanitization

**Location:** `codebase_rag/json_ingestion.py` - `ingest_relationships()` (relationship loop)

**Fix:** Add `sanitize_edge_label()` function and apply before Cypher construction. Preserve original case. Use sanitized label consistently for both Cypher queries and existing-relationship key matching.

```python
def sanitize_edge_label(label: str) -> str:
    if not label:
        raise ValueError("Empty relationship type cannot be sanitized")

    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", label.strip())
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)

    if sanitized and sanitized[0].isdigit():
        sanitized = f"R_{sanitized}"

    if not sanitized:
        raise ValueError(
            f"Relationship type '{label}' reduces to empty after sanitization"
        )

    return sanitized
```

**Updated ingestion loop:**
```python
rel_type = sanitize_edge_label(str(relationship["relationship"]))
```

---

### Issue 3: Entity ID Collision Prevention

**Location:** `codebase_rag/json_ingestion.py` - `_canonical_entity_id()` and `_normalize_entities()`

**Fix:** Keep existing `_canonical_entity_id()` signature and slug format to preserve backward compatibility. Add a collision-avoidance wrapper used only during auto-ID generation in `_normalize_entities`. The `seen_ids` set already exists in `_normalize_entities`; reuse it.

```python
def _canonical_entity_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or "entity"


def _canonical_entity_id_with_collision_avoidance(
    name: str, seen_ids: set[str]
) -> str:
    base_id = _canonical_entity_id(name)
    if base_id not in seen_ids:
        return base_id
    name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    candidate = f"{base_id}_{name_hash}"
    suffix = 1
    while candidate in seen_ids:
        candidate = f"{base_id}_{name_hash}_{suffix}"
        suffix += 1
    return candidate
```

**Updated `_normalize_entities`:**
```python
if not entity.get("id") and entity.get("name"):
    entity["id"] = _canonical_entity_id_with_collision_avoidance(
        str(entity["name"]), seen_ids
    )
```

---

### Issue 4: Warn on Missing Category Field

**Location:** `codebase_rag/json_ingestion.py` - `_relationship_properties()`

**Fix:** Add explicit `logger.warning` (loguru) before the fallback. Log message template added to `logs.py`.

```python
if not relationship.get("category") and original_verb:
    logger.warning(
        ls.JSON_MISSING_CATEGORY.format(
            relationship=original_verb,
            source=relationship.get("source"),
            target=relationship.get("target"),
        )
    )
```

---

### Issue 5: Unicode Emoji Handling

**Location:** `codebase_rag/json_ingestion.py` - `_extract_emoji_prefix()`

**Fix:** Update to handle ZWJ sequences (U+200D), skin tone modifiers (U+1F3FB-U+1F3FF), and variation selectors (U+FE0F). Keep existing emoji range checks.

```python
def _extract_emoji_prefix(name: str) -> tuple[str, str | None]:
    if not name:
        return name, None

    idx = 0
    chars = list(name)

    def _is_emoji_char(cp: int) -> bool:
        return (
            (0x2300 <= cp <= 0x23FF)
            or (0x2600 <= cp <= 0x26FF)
            or (0x2700 <= cp <= 0x27BF)
            or (0x1F1E0 <= cp <= 0x1F1FF)
            or (0x1F300 <= cp <= 0x1F9FF)
            or (0x1FA00 <= cp <= 0x1FA6F)
            or (0x1FA70 <= cp <= 0x1FAFF)
        )

    while idx < len(chars):
        cp = ord(chars[idx])
        if not _is_emoji_char(cp):
            break

        idx += 1

        if idx < len(chars) and ord(chars[idx]) == 0xFE0F:
            idx += 1

        if idx < len(chars) and 0x1F3FB <= ord(chars[idx]) <= 0x1F3FF:
            idx += 1

        if idx < len(chars) and ord(chars[idx]) == 0x200D:
            peek = idx + 1
            if peek < len(chars) and _is_emoji_char(ord(chars[peek])):
                idx += 1
            else:
                break

    if idx == 0:
        return name, None

    emoji = name[:idx]
    clean_name = name[idx:].strip()
    return clean_name, emoji
```

---

### Issue 6: Case-Insensitive Label Deduplication

**Location:** `codebase_rag/json_ingestion.py` - `_entity_labels()`

**Fix:** Use a `seen_lower` set for case-insensitive deduplication instead of `dict.fromkeys()`.

```python
def _entity_labels(entity: dict[str, Any]) -> list[str]:
    entity_type = str(entity.get("type") or "Entity")
    normalized_type = _normalize_label(entity_type)

    labels = [JSON_ENTITY_LABEL, normalized_type]
    seen_lower: set[str] = {label.lower() for label in labels}

    for label in entity.get("labels") or []:
        normalized = _normalize_label(str(label))
        if normalized and normalized.lower() not in seen_lower:
            labels.append(normalized)
            seen_lower.add(normalized.lower())

    return [label for label in labels if label]
```

---

## File Changes Summary

| File | Change Type | Lines Affected |
|------|-------------|---------------|
| `codebase_rag/json_ingestion.py` | Modify | `_canonical_entity_id()`, `_extract_emoji_prefix()`, `_normalize_relationship_category()`, `_normalize_entities()`, `_entity_labels()`, `_relationship_properties()`, `ingest_relationships()` |
| `codebase_rag/logs.py` | Add | 2 new log templates |
| `codebase_rag/tests/test_json_ingestion.py` | Add | 1 new test |
| `codebase_rag/tests/test_json_ingestion_enhancements.py` | Modify | Expanded tests |

---

## Testing Requirements

### Unit Tests

1. `test_category_normalization()` - Verify all alias mappings and unknown-category warning
2. `test_edge_label_sanitization()` - Verify special char handling, case preservation, leading digit
3. `test_entity_id_collision()` - Verify unique ID generation with hash suffix in `_normalize_entities`
4. `test_emoji_extraction()` - Verify ZWJ sequence, skin tone, family emoji handling
5. `test_label_deduplication()` - Verify case-insensitive dedup
6. `test_missing_category_warning()` - Verify warning logged with verb fallback
