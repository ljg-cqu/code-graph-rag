# Taxonomy-Derived Concept Relationship Vocabulary

## Problem Statement

CGR's concept extraction for documentation currently uses only 4 relationship types:

| Current Type | Category | Role |
|-------------|----------|------|
| `IS_A` | Hierarchical | Subtype/classification |
| `PART_OF` | Compositional | Structural composition |
| `CAUSES` | Causal | Cause-effect |
| `RELATED_TO` | **catch-all** | Everything else |

`RELATED_TO` absorbs 5 semantic categories — Contextual, Attribute, Comparative, Sequential, Analogical — into one meaningless edge. When querying the graph, `RELATED_TO` tells you *that* two concepts connect but not *how*. This is a loss of precision the LLM is capable of but cannot express under the current schema.

## Design Goal

Replace the 4-type + catch-all model with an **8-category canonical taxonomy** (MECE-complete, aligned to David Hyerle's Thinking Maps) where the LLM selects a **precise verb** that belongs to one of exactly 8 canonical categories. The graph stores:

- **Edge label** = canonical category (8 fixed labels for fast, deterministic Cypher traversal)
- **`verb` property** = the specific, semantically precise verb the LLM chose
- **`emoji` property** = visual marker for graph UIs
- **`strength` property** = existing confidence float (unchanged)

## Source Taxonomy

Derived from the Concept Relationship Taxonomy (v3.0.0) at `knowledge/productivity/mental-model/relationship/`.

### The 8 Canonical Categories

| Category | Thinking Map | Core Question | Category Label | Emoji | Core Verbs |
|----------|-------------|---------------|----------------|-------|------------|
| Hierarchical | Tree Map | What type/category? | `HIERARCHICAL` | 🌳 | is-a, subtype-of, classifies-as, inherits-from, specializes, instance-of |
| Compositional | Brace Map | What parts make up this? | `COMPOSITIONAL` | 🧩 | part-of, comprises, contains, includes, component-of, consists-of |
| Contextual | Circle Map | What context surrounds? | `CONTEXTUAL` | 🎯 | located-in, situated-in, operates-within, deployed-in, occurs-within |
| Attribute | Bubble Map | What attributes describe? | `ATTRIBUTIVE` | 💭 | has-property, characterized-by, exhibits, possesses, features, embodies |
| Comparative | Double Bubble Map | How do these compare? | `COMPARATIVE` | ⚖️ | compares-to, contrasts-with, similar-to, different-from, equivalent-to |
| Sequential | Flow Map | What happens in order? | `SEQUENTIAL` | ⏩ | precedes, follows, transitions-to, evolves-into, progresses-to |
| Causal | Multi-Flow Map | What causes what? | `CAUSAL` | ⚡ | causes, produces, triggers, enables, prevents, depends-on, influences |
| Analogical | Bridge Map | What is this similar to? | `ANALOGICAL` | 🌉 | analogous-to, corresponds-to, maps-to, parallels, resembles, mirrors |

### Verb Resolution Strategy

The LLM is given the 8 categories with example verbs but is instructed to select the **most precise verb** for each relationship, even if it is not among the listed examples. The only constraint: the verb must belong to exactly one of the 8 categories. This enables the LLM to use domain-specific verbs (e.g., `mitigates`, `instantiated`, `deployed-in`) without requiring them to be pre-registered.

**Verb → Category mapping is deterministic.** When the graph stores an edge, the category label is derived from a register-based lookup. Unknown verbs fall back to the LLM's declared category, then to a fuzzy match, then to `RELATED_TO` as an explicit last resort (never a default).

**Ambiguous verbs are deliberately pinned.** Some verbs (e.g., `derives-from`) could reasonably belong to multiple categories depending on context — `derives-from` could be HIERARCHICAL (classification lineage) or CAUSAL (one thing produces another). The registry pins each ambiguous verb to a single canonical category as the authoritative interpretation. When the LLM's usage disagrees with the registry, the registry wins (step 2 of resolution). The LLM is instructed to prefer unambiguous alternatives when precision matters.

## Data Model Changes

### Graph Edge Schema

```
Before (4 labels):
  (:Concept)-[:RELATED_TO {strength: 0.8}]->(:Concept)
  (:Concept)-[:IS_A {strength: 0.9}]->(:Concept)
  (:Concept)-[:PART_OF {strength: 0.7}]->(:Concept)
  (:Concept)-[:CAUSES {strength: 0.85}]->(:Concept)

After (9 labels, 8 canonical + 1 fallback):
  (:Concept)-[:HIERARCHICAL {verb: "is-a", emoji: "🌳", strength: 0.9}]->(:Concept)
  (:Concept)-[:COMPOSITIONAL {verb: "part-of", emoji: "🧩", strength: 0.7}]->(:Concept)
  (:Concept)-[:CONTEXTUAL {verb: "deployed-in", emoji: "🎯", strength: 0.85}]->(:Concept)
  (:Concept)-[:ATTRIBUTIVE {verb: "characterized-by", emoji: "💭", strength: 0.8}]->(:Concept)
  (:Concept)-[:COMPARATIVE {verb: "contrasts-with", emoji: "⚖️", strength: 0.75}]->(:Concept)
  (:Concept)-[:SEQUENTIAL {verb: "precedes", emoji: "⏩", strength: 0.9}]->(:Concept)
  (:Concept)-[:CAUSAL {verb: "mitigates", emoji: "⚡", strength: 0.85}]->(:Concept)
  (:Concept)-[:ANALOGICAL {verb: "analogous-to", emoji: "🌉", strength: 0.8}]->(:Concept)
  (:Concept)-[:RELATED_TO {verb: "unclassified", emoji: "🔗", strength: 0.3}]->(:Concept)
```

### Pydantic Model: `ConceptRelationship`

```python
class ConceptRelationship(BaseModel):
    from_concept: str
    to_concept: str
    verb: str                          # Was: relationship_type — now the specific verb
    category: str                      # New: one of the 8 canonical category labels
    strength: float = 0.5
    emoji: str = ""                    # New: emoji for visual display
```

### Pydantic Model: `ExtractionResult`

Unchanged API — still holds `concepts` (list[ExtractedConcept]) and `relationships` (list[ConceptRelationship]).

### Category Constants

```python
# In constants.py: DocConceptRelCategory StrEnum (reusable label constants)
class DocConceptRelCategory(StrEnum):
    HIERARCHICAL = "HIERARCHICAL"
    COMPOSITIONAL = "COMPOSITIONAL"
    CONTEXTUAL = "CONTEXTUAL"
    ATTRIBUTIVE = "ATTRIBUTIVE"
    COMPARATIVE = "COMPARATIVE"
    SEQUENTIAL = "SEQUENTIAL"
    CAUSAL = "CAUSAL"
    ANALOGICAL = "ANALOGICAL"
    RELATED_TO = "RELATED_TO"  # explicit fallback

DOC_CONCEPT_CATEGORIES: frozenset[str] = frozenset(DocConceptRelCategory)

# In constants.py: canonical emoji for each category (server-side authority, next to DocConceptRelCategory)
CATEGORY_EMOJI_MAP: dict[str, str] = {
    "HIERARCHICAL": "🌳",
    "COMPOSITIONAL": "🧩",
    "CONTEXTUAL": "🎯",
    "ATTRIBUTIVE": "💭",
    "COMPARATIVE": "⚖️",
    "SEQUENTIAL": "⏩",
    "CAUSAL": "⚡",
    "ANALOGICAL": "🌉",
    "RELATED_TO": "🔗",
}
```

The emoji is **always derived server-side from the resolved category** via `CATEGORY_EMOJI_MAP`. The LLM may optionally provide an emoji, but the server always overwrites it with the canonical value for the resolved category. This guarantees emoji-category consistency even when the resolver overrides the LLM's declared category.

## LLM Prompt Changes

### New SYSTEM_PROMPT

The prompt teaches the LLM the 8-category framework with example verbs, then instructs it to pick the most precise verb. This is substantially different from the current prompt which lists only 4 types.

Key prompt design decisions:
- **Show categories, not all 132 verbs** — the LLM already knows verb semantics; it needs the organizational framework
- **Instruct precision** — "use the most specific verb that accurately describes the relationship"
- **Require category classification** — each relationship must declare which of the 8 categories it belongs to
- **RELATED_TO is a conscious last resort** — not a default

See Appendix A for full prompt text.

## Cypher Storage Changes

### `_store_concept_relationships_batch()`

Expands from 4 `FOREACH` branches to 9 (8 canonical + `RELATED_TO` fallback). Each branch:

```cypher
FOREACH (_ IN CASE WHEN rel.category = 'HIERARCHICAL' THEN [1] ELSE [] END |
    MERGE (a)-[r:HIERARCHICAL]->(b)
    SET r.verb = rel.verb, r.emoji = rel.emoji, r.strength = rel.strength
)
```

All 9 labels use identical structure — only the category name and properties differ.

### Graph Indexes

Add indexes for the 5 new edge labels:
```cypher
CREATE INDEX ON :HIERARCHICAL(verb);
CREATE INDEX ON :COMPOSITIONAL(verb);
CREATE INDEX ON :CONTEXTUAL(verb);
CREATE INDEX ON :ATTRIBUTIVE(verb);
CREATE INDEX ON :COMPARATIVE(verb);
CREATE INDEX ON :SEQUENTIAL(verb);
CREATE INDEX ON :CAUSAL(verb);
CREATE INDEX ON :ANALOGICAL(verb);
CREATE INDEX ON :RELATED_TO(verb);
```

### Graph Algorithms Update

`DocumentGraphAlgorithms` edge type patterns must expand from `RELATED_TO|IS_A|PART_OF|CAUSES` to include both old AND new labels. **Old labels must remain in the patterns** because existing graph data uses them — new extractions create edges with new labels, but old edges are not migrated by default. Omitting old labels would make existing concept relationships invisible to traversal.

Before:
```cypher
MATCH ()-[:RELATED_TO|IS_A|PART_OF|CAUSES]-()
```

After (backward compatible — old + new labels):
```cypher
MATCH ()-[:RELATED_TO|IS_A|PART_OF|CAUSES|HIERARCHICAL|COMPOSITIONAL|CONTEXTUAL|ATTRIBUTIVE|COMPARATIVE|SEQUENTIAL|CAUSAL|ANALOGICAL]-()
```

This applies to all three traversal methods: `find_shortest_path`, `find_related_concepts`, and `find_concept_neighbors`.

## Verb Registry

A deterministic mapping from verbs to canonical categories. Used for server-side validation of LLM output and for verb → category fallback resolution.

The registry starts with the 132 core verbs from the taxonomy and grows as domain-specific verbs are encountered. See Appendix B for initial registry.

### Category Resolution Logic

```
Given (verb, declared_category):
  1. If verb in registry AND registry[verb] == declared_category → use declared_category
  2. If verb in registry AND registry[verb] != declared_category → use registry[verb] (LLM declared wrong category)
  3. If verb NOT in registry → use declared_category, log for registry expansion
  4. If verb NOT in registry AND no declared_category → fuzzy match against registry, fallback to RELATED_TO

After category is resolved, emoji is derived from CATEGORY_EMOJI_MAP[category].
The emoji is NEVER taken from LLM output — it is always server-resolved from the final category.
```

`resolve_category(verb: str, declared_category: str | None) -> tuple[str, str]` returns `(category, emoji)`.
The caller uses the returned category as the edge label and sets `r.emoji = emoji` on the relationship.

## Backward Compatibility

### Existing Graph Data

Old relationships (`IS_A`, `PART_OF`, `CAUSES`, `RELATED_TO`) remain in the graph untouched. New extractions create edges with the new labels. Old labels continue to work in queries.

### Migration Considerations

Old edges lack `verb` and `emoji` properties. Queries should handle missing properties gracefully:

```cypher
MATCH (a)-[r:IS_A|HIERARCHICAL]->(b)
RETURN a.name, coalesce(r.verb, "is-a"), b.name
```

A migration script can optionally upgrade old edges:
```cypher
MATCH (a)-[r:IS_A]->(b)
CREATE (a)-[:HIERARCHICAL {verb: "is-a", emoji: "🌳", strength: r.strength}]->(b)
DELETE r
```

This migration is **optional and out of scope for initial implementation**. The old labels can co-exist with the new ones indefinitely.

## File Changes

### 1. `codebase_rag/constants.py`
- Add `DocConceptRelCategory` StrEnum with 9 members: `HIERARCHICAL`, `COMPOSITIONAL`, `CONTEXTUAL`, `ATTRIBUTIVE`, `COMPARATIVE`, `SEQUENTIAL`, `CAUSAL`, `ANALOGICAL`, `RELATED_TO`
- Add `DOC_CONCEPT_CATEGORIES: frozenset[str]` for category iteration/validation
- Add `CATEGORY_EMOJI_MAP: dict[str, str]` mapping each category to its canonical emoji
- Keep existing `IS_A`, `PART_OF`, `CAUSES`, `RELATED_TO` in `RelationshipType` for backward compatibility with existing graph data

### 2. `codebase_rag/document/concept_extraction.py`
- Update `ConceptRelationship`: rename `relationship_type` → `verb`, add `category` and `emoji` fields
- Update `SYSTEM_PROMPT` with 8-category taxonomy (see Appendix A)
- Add `VERB_REGISTRY`: dict[str, str] mapping known verbs to canonical categories
- Add `resolve_category(verb: str, declared_category: str | None) -> tuple[str, str]` function. Returns `(category, emoji)` — the resolved canonical category and its canonical emoji. Emoji is always derived server-side from the resolved category via `CATEGORY_EMOJI_MAP`; LLM-provided emoji is ignored.
- Update `LLMConceptExtractor.extract()` to call `resolve_category()` on each LLM-returned relationship, populating `category` and `emoji` from the resolved values

### 3. `codebase_rag/document/document_updater.py`
- Update `_store_concept_relationships_batch()` signature: `list[tuple[str, str, str, str, float]]` → `list[tuple[str, str, str, str, str, float]]` (add verb, emoji, category)
- Expand Cypher from 4 to 9 FOREACH branches
- Update `_extract_and_store_concepts()` to pass new fields

### 4. `codebase_rag/document/graph_algorithms.py`
- Expand edge type patterns in `find_shortest_path()`, `find_related_concepts()`, `find_concept_neighbors()` to include all 9 category labels

### 5. Tests (new file)
- `codebase_rag/tests/test_concept_taxonomy.py`
- Test verb→category resolution (registered match, registered mismatch override, unregistered with declared category, unregistered without declared category)
- Test emoji always derived from resolved category (not from LLM input), including resolution override cases
- Test category MECE compliance (all 8 canonical categories covered, no overlap)
- Test backward compatibility (old labels still queryable alongside new in graph traversal)
- Test `resolve_category` returns valid emoji for all 9 categories (including RELATED_TO fallback)

## Implementation Phases

### Phase 1: Schema + Prompt (core change)
1. Update `constants.py` with new StrEnum
2. Update `ConceptRelationship` model
3. Rewrite `SYSTEM_PROMPT` with taxonomy
4. Add `VERB_REGISTRY` and `resolve_category()`
5. Update `_store_concept_relationships_batch()` Cypher

### Phase 2: Integration
6. Update `DocumentGraphAlgorithms` edge patterns
7. Update caller code in `_extract_and_store_concepts()`
8. Add graph indexes for new labels

### Phase 3: Testing
9. Unit tests for verb registry
10. Integration test with real LLM extraction
11. Verify old labels still queryable alongside new

## Success Criteria

1. ✅ LLM outputs contain precise verbs (e.g., `mitigates`, `deployed-in`) not just 4 generic types
2. ✅ Every relationship has exactly one canonical category label
3. ✅ `RELATED_TO` usage drops significantly (used only as explicit last resort)
4. ✅ Graph traversal works across old and new labels
5. ✅ Queries can filter by category (`:CAUSAL`) or by specific verb (`WHERE r.verb = 'mitigates'`)
6. ✅ Backward compatible — existing graph data continues to work
7. ✅ No breaking changes to `ExtractionResult` API contract

---

## Appendix A: Full SYSTEM_PROMPT

```
You are a concept extractor for technical documentation.
Given a document chunk, identify:
1. Key concepts mentioned (with definitions if available)
2. The type/category of each concept
3. Relationships between concepts

Concept types:
- skill: A learnable capability or competency
- framework: A structured approach, methodology, or system
- model: A conceptual representation or theoretical construct
- process: A systematic sequence of actions or operations
- risk: A potential negative outcome, pitfall, or concern
- principle: A fundamental truth, rule, or guideline
- concept: A general idea, notion, or abstract thought

Relationships use an 8-category canonical taxonomy. Choose the most specific verb
that accurately describes how two concepts relate. The verb must belong to exactly
one of these categories:

🌳 HIERARCHICAL — What type/category?
  is-a, subtype-of, classifies-as, inherits-from, specializes, instance-of

🧩 COMPOSITIONAL — What parts make up this?
  part-of, comprises, contains, includes, consists-of, component-of

🎯 CONTEXTUAL — What context surrounds?
  located-in, situated-in, operates-within, deployed-in, occurs-within, is-bound-by

💭 ATTRIBUTIVE — What attributes describe?
  has-property, characterized-by, exhibits, possesses, features, requires

⚖️ COMPARATIVE — How do these compare?
  compares-to, contrasts-with, similar-to, different-from, supersedes, equivalent-to

⏩ SEQUENTIAL — What happens in order?
  precedes, follows, transitions-to, evolves-into, progresses-to

⚡ CAUSAL — What causes what?
  causes, produces, triggers, enables, prevents, depends-on, influences, mitigates, generates

🌉 ANALOGICAL — What is this similar to?
  analogous-to, corresponds-to, maps-to, parallels, resembles, mirrors

🔗 RELATED_TO — Use ONLY as a last resort when no other category fits.

Respond with JSON matching this structure:
{
  "concepts": [
    {
      "name": "concept name",
      "aliases": ["alternative name"],
      "type": "skill|framework|model|process|risk|principle|concept",
      "definition": "brief definition",
      "confidence": 0.9
    }
  ],
  "relationships": [
    {
      "from_concept": "source",
      "to_concept": "target",
      "verb": "mitigates",
      "category": "CAUSAL",
      "emoji": "⚡",
      "strength": 0.8
    }
  ]
}

Rules:
- Only extract concepts that are clearly defined or important in the text
- Assign the most specific concept type; use "concept" as fallback
- Use the most precise relationship verb possible, even beyond the examples listed
- Every verb must fit into exactly one of the 8 categories above
- category must be one of: HIERARCHICAL, COMPOSITIONAL, CONTEXTUAL, ATTRIBUTIVE, COMPARATIVE, SEQUENTIAL, CAUSAL, ANALOGICAL, RELATED_TO
- emoji must match the category: 🌳🧩🎯💭⚖️⏩⚡🌉🔗
- RELATED_TO is a last resort, not a default
- Confidence/strength should reflect how clearly the concept/relationship is presented
```

## Appendix B: Initial Verb Registry

The initial registry includes the 132 core taxonomy verbs. Domain verbs are added as encountered.

```python
VERB_REGISTRY: dict[str, str] = {
    # Hierarchical (🌳)
    "is-a": "HIERARCHICAL",
    "subtype-of": "HIERARCHICAL",
    "classifies-as": "HIERARCHICAL",
    "categorizes-under": "HIERARCHICAL",
    "inherits-from": "HIERARCHICAL",
    "specializes": "HIERARCHICAL",
    "generalizes-to": "HIERARCHICAL",
    "instance-of": "HIERARCHICAL",
    "supertype-of": "HIERARCHICAL",
    "descends-from": "HIERARCHICAL",
    "is-parent-of": "HIERARCHICAL",
    "falls-under": "HIERARCHICAL",
    "derives-from": "HIERARCHICAL",
    # Compositional (🧩)
    "part-of": "COMPOSITIONAL",
    "comprises": "COMPOSITIONAL",
    "contains": "COMPOSITIONAL",
    "includes": "COMPOSITIONAL",
    "component-of": "COMPOSITIONAL",
    "constituent-of": "COMPOSITIONAL",
    "element-of": "COMPOSITIONAL",
    "member-of": "COMPOSITIONAL",
    "composed-of": "COMPOSITIONAL",
    "consists-of": "COMPOSITIONAL",
    "is-built-from": "COMPOSITIONAL",
    "aggregates": "COMPOSITIONAL",
    "is-formed-from": "COMPOSITIONAL",
    # Contextual (🎯)
    "located-in": "CONTEXTUAL",
    "situated-in": "CONTEXTUAL",
    "provides-context-for": "CONTEXTUAL",
    "framed-by": "CONTEXTUAL",
    "environment-of": "CONTEXTUAL",
    "setting-for": "CONTEXTUAL",
    "surrounds": "CONTEXTUAL",
    "contained-within": "CONTEXTUAL",
    "occurs-within": "CONTEXTUAL",
    "operates-within": "CONTEXTUAL",
    "exists-under": "CONTEXTUAL",
    "takes-place-in": "CONTEXTUAL",
    "is-bound-by": "CONTEXTUAL",
    "is-hosted-in": "CONTEXTUAL",
    # Attributive (💭)
    "has-property": "ATTRIBUTIVE",
    "characterized-by": "ATTRIBUTIVE",
    "exhibits": "ATTRIBUTIVE",
    "possesses": "ATTRIBUTIVE",
    "displays": "ATTRIBUTIVE",
    "manifests": "ATTRIBUTIVE",
    "features": "ATTRIBUTIVE",
    "embodies": "ATTRIBUTIVE",
    "expresses": "ATTRIBUTIVE",
    "has-characteristic": "ATTRIBUTIVE",
    "bears": "ATTRIBUTIVE",
    # Comparative (⚖️)
    "compares-to": "COMPARATIVE",
    "contrasts-with": "COMPARATIVE",
    "similar-to": "COMPARATIVE",
    "akin-to": "COMPARATIVE",
    "different-from": "COMPARATIVE",
    "equivalent-to": "COMPARATIVE",
    "comparable-to": "COMPARATIVE",
    "interchangeable-with": "COMPARATIVE",
    "opposite-of": "COMPARATIVE",
    "synonym-of": "COMPARATIVE",
    "antonym-of": "COMPARATIVE",
    "same-as": "COMPARATIVE",
    "related-to": "COMPARATIVE",
    # Sequential (⏩)
    "precedes": "SEQUENTIAL",
    "follows": "SEQUENTIAL",
    "occurs-during": "SEQUENTIAL",
    "transitions-to": "SEQUENTIAL",
    "evolves-into": "SEQUENTIAL",
    "progresses-to": "SEQUENTIAL",
    "succeeds": "SEQUENTIAL",
    # Causal (⚡)
    "causes": "CAUSAL",
    "produces": "CAUSAL",
    "triggers": "CAUSAL",
    "prevents": "CAUSAL",
    "enables": "CAUSAL",
    "inhibits": "CAUSAL",
    "influences": "CAUSAL",
    "affects": "CAUSAL",
    "determines": "CAUSAL",
    "results-in": "CAUSAL",
    "creates": "CAUSAL",
    "destroys": "CAUSAL",
    "modifies": "CAUSAL",
    "amplifies": "CAUSAL",
    "reduces": "CAUSAL",
    "depends-on": "CAUSAL",
    "accelerates": "CAUSAL",
    "activates": "CAUSAL",
    "alleviates": "CAUSAL",
    "blocks": "CAUSAL",
    "boosts": "CAUSAL",
    "catalyzes": "CAUSAL",
    "constrains": "CAUSAL",
    "converts": "CAUSAL",
    "delays": "CAUSAL",
    "degrades": "CAUSAL",
    "drives": "CAUSAL",
    "eases": "CAUSAL",
    "enhances": "CAUSAL",
    "facilitates": "CAUSAL",
    "fosters": "CAUSAL",
    "generates": "CAUSAL",
    "impedes": "CAUSAL",
    "induces": "CAUSAL",
    "limits": "CAUSAL",
    "maintains": "CAUSAL",
    "motivates": "CAUSAL",
    "obstructs": "CAUSAL",
    "permits": "CAUSAL",
    "prolongs": "CAUSAL",
    "promotes": "CAUSAL",
    "anticipates": "CAUSAL",
    "correlates-with": "CAUSAL",
    "initiates": "CAUSAL",
    "terminates": "CAUSAL",
    "leads-to": "CAUSAL",
    "prepares-for": "CAUSAL",
    "builds": "CAUSAL",
    "detects": "CAUSAL",
    "corrects": "CAUSAL",
    # Analogical (🌉)
    "analogous-to": "ANALOGICAL",
    "corresponds-to": "ANALOGICAL",
    "maps-to": "ANALOGICAL",
    "parallels": "ANALOGICAL",
    "resembles": "ANALOGICAL",
    "mirrors": "ANALOGICAL",
    "symbolizes": "ANALOGICAL",
    "represents": "ANALOGICAL",
    "stands-for": "ANALOGICAL",
    "exemplifies": "ANALOGICAL",
    "illustrates": "ANALOGICAL",
    "metaphor-for": "ANALOGICAL",
    "isomorphic-to": "ANALOGICAL",
    "metaphorically-represents": "ANALOGICAL",
    # Domain: Software/Application (mapped to canonical categories)
    "deployed-in": "CONTEXTUAL",
    "runs-on": "CONTEXTUAL",
    "hosted-by": "CONTEXTUAL",
    "connects-to": "CONTEXTUAL",
    "interfaces-with": "CONTEXTUAL",
    "consumes": "CONTEXTUAL",
    "provides": "CAUSAL",
    "configured-with": "ATTRIBUTIVE",
    "secured-by": "ATTRIBUTIVE",
    "versioned-as": "ATTRIBUTIVE",
    "requires": "ATTRIBUTIVE",
    "exposes": "ATTRIBUTIVE",
    "supports": "CAUSAL",
    "implements": "COMPOSITIONAL",
    "integrates-with": "COMPOSITIONAL",
    "bundles": "COMPOSITIONAL",
    "encapsulates": "COMPOSITIONAL",
    "decomposes-into": "COMPOSITIONAL",
    "layer-in": "COMPOSITIONAL",
    "alternative-to": "COMPARATIVE",
    "predecessor-of": "COMPARATIVE",
    "successor-of": "COMPARATIVE",
    "replaces": "COMPARATIVE",
    "extends": "HIERARCHICAL",
    "supersedes": "COMPARATIVE",
    "compatible-with": "COMPARATIVE",
    "processes": "SEQUENTIAL",
    "handles": "SEQUENTIAL",
    "validates": "SEQUENTIAL",
    "transforms": "SEQUENTIAL",
    "schedules": "SEQUENTIAL",
    "impacts": "CAUSAL",
    "resolves": "CAUSAL",
    "mitigates": "CAUSAL",
    "pattern-is": "ANALOGICAL",
    "models": "ANALOGICAL",
    "abstracts": "ANALOGICAL",
}
```

---

## Version

v1.0.0 | 2026-04-25 | Scope: Concept extraction relationship vocabulary upgrade
