# Taxonomy-Derived Entity Category Classification

## Problem Statement

CGR's concept extraction currently classifies concepts with an ad-hoc, unstructured `type` field:

```python
type: str | None = Field(
    default=None,
    description="Concept type: skill, framework, model, process, risk, principle, concept",
)
```

Seven informal labels with no boundary rules, no disambiguation protocol, no sub-type hierarchy, and no MECE guarantee. This creates three problems:

1. **Ambiguous classification** — Is "Kubernetes" a framework, a model, or a concept? Without boundary rules, the LLM picks arbitrarily.
2. **No sub-type precision** — "Docker Image" and "Data Center" are both vague "framework/model/concept" under the current system. They're fundamentally different kinds of entities (Container Image vs. Facility).
3. **Asymmetric with relationships** — CGR's relationship taxonomy (8 MECE categories, 132 verbs, boundary rules, server-side resolution) gives edges rich semantics. Nodes have none of this structure.

The relationship taxonomy (integrated in v1.0.0) classifies *how concepts connect*. The entity taxonomy classifies *what concepts are*. Together they give the knowledge graph typed nodes and typed edges — a complete semantic model.

## Design Goal

Replace the ad-hoc 7-label `type` field with a **7-category MECE entity taxonomy** where every extracted concept is classified into exactly one canonical entity category, optionally with a domain-specific sub-type. The Concept node stores:

- **`entity_category` property** = one of 7 canonical categories (for deterministic Cypher filtering)
- **`entity_subtype` property** = optional domain-specific sub-type (for precision)
- **`emoji` property** = visual marker (server-resolved, same pattern as relationship emoji)

The entity taxonomy is the **node counterpart** to the 8-category relationship taxonomy. Both share the same design language: 2-step selection, boundary disambiguation matrix, edge case resolution protocol, and server-side authority for emoji.

## Source Taxonomy

Derived from the Entity Category Taxonomy (v1.0.0) at `knowledge/productivity/mental-model/entity/`.

### The 7 Canonical Entity Categories

Ordered on a concrete → abstract ontological gradient:

| # | Category | Core Question | Label | Emoji | Core Sub-Types |
|---|----------|---------------|-------|-------|----------------|
| 1 | Concrete Entity | Does it occupy physical space? | `CONCRETE_ENTITY` | 🧱 | Natural Object, Artifact, Substance, Organism, Body Part, Food/Consumable, Geographic Feature, Celestial Body |
| 2 | Event/Process | Does it unfold over time? | `EVENT_PROCESS` | ⏱️ | Natural Event, Human Action, Process, Incident, Activity, State Change, Project/Initiative, Ritual/Routine |
| 3 | Information/Expression | Is it a representation? | `INFORMATION_EXPRESSION` | 📨 | Data, Signal, Symbol, Narrative, Code/Formula, Record/Document, Media |
| 4 | Property/Attribute | Is it a characteristic of something else? | `PROPERTY_ATTRIBUTE` | 📏 | Physical Quality, Quantitative Measure, Mental State, Capability/Skill, Disposition, Relational Property, Evaluative Property |
| 5 | System/Structure | Is it an organized collection? | `SYSTEM_STRUCTURE` | 🏗️ | Natural System, Social System, Technological System, Network, Hierarchy, Framework, Market/Platform |
| 6 | Agent/Role | Does it exercise intention? | `AGENT_ROLE` | 🎭 | Individual, Collective, Institutional Agent, Non-Human Agent, Role/Position, Persona |
| 7 | Abstract Concept | Is it a pure idea? | `ABSTRACT_CONCEPT` | 💡 | Domain/Discipline, Theory/Model, Principle/Rule, Value/Ideal, Category/Class, Relation/Connection |

**Abstract Concept is the catch-all last resort** — mirroring `RELATED_TO` in the relationship taxonomy. Before classifying an entity here, the other 6 categories must be ruled out.

### Domain Extensions (Phase 2+)

The 7 core categories are MECE-complete. Domain extensions provide **specialized sub-types within** existing categories — never new top-level categories:

| Domain | File | Categories Extended | Sub-Types |
|--------|------|---------------------|:---------:|
| Software | `domain-extensions/software.md` | All 7 | 31 |
| Business | `domain-extensions/business.md` | Concrete, System, Agent, Abstract | 22 |
| Scientific | `domain-extensions/scientific.md` | Concrete, Event, Property, System | 20 |

Software domain sub-types (most relevant to CGR): Virtual Machine, Container Image, Data Center, Deployment, Build, Test Run, Incident, API, Protocol, Config File, Log Stream, Source File, SLI, Quality Attribute, Distributed System, CI/CD Pipeline, Monorepo, Service Mesh, CI Bot, Service Account, End User, Design Pattern, Algorithm, Protocol Spec, Paradigm, SLA.

### Boundary Disambiguation Matrix

When an entity is ambiguous between two categories, a named test rule resolves it:

| Category Pair | Boundary Rule |
|---------------|---------------|
| Concrete vs Event | Substance test: has material composition → Concrete; temporal occurrence → Event |
| Concrete vs Information | Material test: value in physical substance → Concrete; value in encoded meaning → Information |
| Concrete vs Property | Independence test: exists independently → Concrete; exists only as characteristic → Property |
| Concrete vs System | Touch test: single physical instance → Concrete; multi-part with emergence → System |
| Concrete vs Agent | Intent test: acts with purpose → Agent; inert physical thing → Concrete |
| Concrete vs Abstract | Tangibility test: perceivable by senses → Concrete; exists only as idea → Abstract |
| Event vs Information | Occurrence test: the event itself → Event; record/representation → Information |
| Event vs Property | Duration test: unfolds over time → Event; static characteristic → Property |
| Event vs System | Temporal test: defined by time span → Event; defined by persistent organization → System |
| Event vs Agent | Actor test: is the action → Event; is the entity that acts → Agent |
| Event vs Abstract | Instantiation test: specific occurrence → Event; general pattern/idea → Abstract |
| Information vs Property | Representation test: encodes meaning → Information; inherent characteristic → Property |
| Information vs System | Symbol test: message/encoding → Information; organized structure → System |
| Information vs Agent | Source test: content/message → Information; entity that created it → Agent |
| Information vs Abstract | Encoding test: requires physical medium/symbol → Information; purely mental → Abstract |
| Property vs System | Emergence test: single characteristic → Property; organized whole with multiple properties → System |
| Property vs Agent | Ascription test: characteristic OF an agent → Property; the agent itself → Agent |
| Property vs Abstract | Inherence test: always property OF something → Property; stands alone → Abstract |
| System vs Agent | Agency test: has goals/intentions → Agent; passive organization → System |
| System vs Abstract | Instance test: has concrete instances → System; exists only as idea → Abstract |
| Agent vs Abstract | Embodiment test: entity that acts → Agent; purely conceptual idea → Abstract |
| Organism vs Agent | Behavior test: biological behavior → Concrete; intentional/moral agency or artificial/supernatural → Agent |

### Edge Case Resolution Protocol

```
Given an ambiguous entity:
  1. Apply core question — run the Step 1 selection question for each candidate category
  2. Apply boundary rule — consult the disambiguation rule for that specific pair
  3. Ontological gradient tiebreaker — prefer the more concrete category
     (Concrete → Event → Information → Property → System → Agent → Abstract)
  4. "Not elsewhere" test — verify the entity doesn't fit more naturally elsewhere;
     if Abstract Concept is the candidate, confirm all 6 other categories ruled out
  5. Document — record the edge case and resolution for future reference
```

## Data Model Changes

### Graph Node Schema

```
Before (no entity category on Concept nodes):
  (:Concept {
    name: "Docker Image",
    qualified_name: "workspace:Docker Image",
    definition: "...",
    confidence: 0.9,
    source_chunk_qn: "..."
  })

After (entity category + subtype + emoji on Concept nodes):
  (:Concept {
    name: "Docker Image",
    qualified_name: "workspace:Docker Image",
    definition: "...",
    confidence: 0.9,
    source_chunk_qn: "...",
    entity_category: "CONCRETE_ENTITY",
    entity_subtype: "Container Image",
    entity_emoji: "🧱"
  })
```

Note: entity classification adds properties to the existing `Concept` node. No new node labels — the 7 categories are property values, not labels. This avoids Cypher label proliferation while still enabling property-based filtering.

### Pydantic Model: `ExtractedConcept`

```python
class ExtractedConcept(BaseModel):
    name: str
    aliases: list[str] = []
    type: str | None = None              # DEPRECATED — kept for backward compat, maps to entity_category
    definition: str
    confidence: float
    source_chunk_qn: str
    context: str = ""
    entity_category: str | None = None   # New: one of 7 canonical category labels
    entity_subtype: str | None = None    # New: domain-specific sub-type (optional, Phase 2)
    entity_emoji: str = ""               # New: server-resolved emoji for the category
```

**Migration strategy for `type`**: The existing `type` field is kept but deprecated. During extraction, `type` is set equal to `entity_category` for backward compatibility with any code reading the old field. A future major version can remove it.

### Category Constants

```python
# In constants.py: DocConceptEntityCategory StrEnum
class DocConceptEntityCategory(StrEnum):
    CONCRETE_ENTITY = "CONCRETE_ENTITY"
    EVENT_PROCESS = "EVENT_PROCESS"
    INFORMATION_EXPRESSION = "INFORMATION_EXPRESSION"
    PROPERTY_ATTRIBUTE = "PROPERTY_ATTRIBUTE"
    SYSTEM_STRUCTURE = "SYSTEM_STRUCTURE"
    AGENT_ROLE = "AGENT_ROLE"
    ABSTRACT_CONCEPT = "ABSTRACT_CONCEPT"

# Frozen set for iteration/validation. Note: distinct from DOC_CONCEPT_CATEGORIES,
# which holds the 8 relationship categories. DOC_ENTITY_CATEGORIES = 7 entity categories.
DOC_ENTITY_CATEGORIES: frozenset[str] = frozenset(DocConceptEntityCategory)

ENTITY_CATEGORY_EMOJI_MAP: dict[str, str] = {
    "CONCRETE_ENTITY": "🧱",
    "EVENT_PROCESS": "⏱️",
    "INFORMATION_EXPRESSION": "📨",
    "PROPERTY_ATTRIBUTE": "📏",
    "SYSTEM_STRUCTURE": "🏗️",
    "AGENT_ROLE": "🎭",
    "ABSTRACT_CONCEPT": "💡",
}
```

The emoji is **always derived server-side** from the resolved category via `ENTITY_CATEGORY_EMOJI_MAP`. Same pattern as `CATEGORY_EMOJI_MAP` for relationships. The Concept node property is named `entity_emoji` (not plain `emoji`) to distinguish it from relationship edge emoji — a Concept node's emoji represents its entity classification, while a relationship edge's emoji represents its connection type.

### Entity Sub-Type Registry (Phase 2)

A deterministic mapping from sub-type names to canonical categories, serving the same role as `VERB_REGISTRY` for relationships:

```python
ENTITY_SUBTYPE_REGISTRY: dict[str, str] = {
    # Core sub-types (49 from entity.md)
    "Natural Object": "CONCRETE_ENTITY",
    "Artifact": "CONCRETE_ENTITY",
    "Substance": "CONCRETE_ENTITY",
    "Organism": "CONCRETE_ENTITY",
    # ... (full 49 core + domain extensions)
}
```

## Entity Category Resolution Logic

```
Given (entity_name, definition, declared_category, declared_subtype):
  1. If declared_category is a valid DocConceptEntityCategory → use it
  2. If declared_category is invalid/None AND declared_subtype is in ENTITY_SUBTYPE_REGISTRY
     → use registry[declared_subtype] (sub-type lookup provides authoritative fallback)
  3. If declared_category is invalid/None AND no subtype registry match
     → fallback to ABSTRACT_CONCEPT, log a warning (it's the last resort)
  4. If ABSTRACT_CONCEPT is the result, log a debug message (routine classification)

After category is resolved, emoji is derived from ENTITY_CATEGORY_EMOJI_MAP[category].
The emoji is always derived server-side from the resolved category. The LLM may optionally
provide an entity_emoji, but the server always overwrites it with the canonical value for the
resolved category. This guarantees emoji-category consistency even when the resolver overrides
the LLM's declared category.

If entity_subtype from the LLM is not in ENTITY_SUBTYPE_REGISTRY, it is kept as-is (the
registry is non-exhaustive; the LLM may produce valid but unregistered sub-types).
```

`resolve_entity_category(declared_category: str | None, declared_subtype: str | None, definition: str = "") -> tuple[str, str | None, str]` returns `(category, subtype, emoji)`.

Key difference from relationship resolution: entity classification doesn't have a strong verb → category signal (like `VERB_REGISTRY`). The LLM's declared category carries more weight. The sub-type registry provides a secondary validation path where the relationship taxonomy relied on fuzzy verb matching — sub-type lookup is deterministic and simpler.

## LLM Prompt Changes

### Updated SYSTEM_PROMPT

The concept type section of the prompt changes from the ad-hoc 7-label list to the 7-category MECE taxonomy. The relationship section is unchanged.

Current concept type instruction:
```
Concept types:
- skill: A learnable capability or competency
- framework: A structured approach, methodology, or system
- model: A conceptual representation or theoretical construct
- process: A systematic sequence of actions or operations
- risk: A potential negative outcome, pitfall, or concern
- principle: A fundamental truth, rule, or guideline
- concept: A general idea, notion, or abstract thought
```

New concept type instruction:
```
Entity categories (7 MECE categories — every concept fits exactly one):

🧱 CONCRETE_ENTITY — Does it occupy physical space?
  Natural Object, Artifact, Substance, Organism, Body Part, Food/Consumable,
  Geographic Feature, Celestial Body
  Software-specific: Virtual Machine, Container Image, Data Center, Mobile Device, Peripheral

⏱️ EVENT_PROCESS — Does it unfold over time?
  Natural Event, Human Action, Process, Incident, Activity, State Change,
  Project/Initiative, Ritual/Routine
  Software-specific: Deployment, Build, Test Run, Incident, Migration

📨 INFORMATION_EXPRESSION — Is it a representation?
  Data, Signal, Symbol, Narrative, Code/Formula, Record/Document, Media
  Software-specific: API, Protocol, Config File, Log Stream, Source File

📏 PROPERTY_ATTRIBUTE — Is it a characteristic of something else?
  Physical Quality, Quantitative Measure, Mental State, Capability/Skill,
  Disposition, Relational Property, Evaluative Property
  Software-specific: SLI, Quality Attribute, Capacity Metric

🏗️ SYSTEM_STRUCTURE — Is it an organized collection?
  Natural System, Social System, Technological System, Network, Hierarchy,
  Framework, Market/Platform
  Software-specific: Distributed System, CI/CD Pipeline, Monorepo, Service Mesh

🎭 AGENT_ROLE — Does it exercise intention or fulfill a role?
  Individual, Collective, Institutional Agent, Non-Human Agent, Role/Position, Persona
  Software-specific: CI Bot, Service Account, End User, On-Call Engineer

💡 ABSTRACT_CONCEPT — Is it a pure idea with no physical form?
  Domain/Discipline, Theory/Model, Principle/Rule, Value/Ideal, Category/Class, Relation/Connection
  Software-specific: Design Pattern, Algorithm, Protocol Spec, Paradigm, SLA

💡 ABSTRACT_CONCEPT is the LAST RESORT — confirm the entity doesn't fit any of the
first 6 categories before using it. This mirrors `related-to` in the relationship taxonomy.
```

The JSON response format adds `entity_category`, `entity_subtype`, and `entity_emoji` to each concept:
```json
{
  "concepts": [
    {
      "name": "Docker Image",
      "aliases": ["Container Image"],
      "type": "CONCRETE_ENTITY",
      "entity_category": "CONCRETE_ENTITY",
      "entity_subtype": "Container Image",
      "entity_emoji": "🧱",
      "definition": "A packaged runtime environment...",
      "confidence": 0.9
    }
  ],
  "relationships": [...]
}
```

Rules additions:
- Every concept must have an `entity_category` from the 7 canonical categories
- `entity_subtype` is optional but recommended for software-domain concepts
- `entity_emoji` must match the category: 🧱⏱️📨📏🏗️🎭💡
- ABSTRACT_CONCEPT is a last resort, not a default
- The old `type` field should equal `entity_category` for backward compatibility

## Cypher Storage Changes

### `_merge_concept_nodes_batch()`

Add `entity_category`, `entity_subtype`, and `entity_emoji` properties to Concept node creation. The Concept node label does not change — entity categories are properties, not labels.

Also update the `concept_nodes` dict built in `_extract_and_store_concepts()` (line ~2647) to include the three new fields: `entity_category`, `entity_subtype`, `entity_emoji`.

Current (simplified):
```cypher
MERGE (c:Concept {qualified_name: $qualified_name})
SET c.name = $name,
    c.definition = $definition,
    c.confidence = $confidence
```

New:
```cypher
MERGE (c:Concept {qualified_name: $qualified_name})
SET c.name = $name,
    c.definition = $definition,
    c.confidence = $confidence,
    c.entity_category = $entity_category,
    c.entity_subtype = $entity_subtype,
    c.entity_emoji = $entity_emoji
```

### Graph Indexes

Add index on entity_category for filtered concept queries:
```cypher
CREATE INDEX ON :Concept(entity_category);
CREATE INDEX ON :Concept(entity_subtype);
```

This enables efficient queries like:
```cypher
MATCH (c:Concept {entity_category: "AGENT_ROLE"})-[r:CAUSAL]->(target)
RETURN c.name, target.name, r.verb
```

### Graph Algorithms

`DocumentGraphAlgorithms` methods do NOT require changes. Entity classification is a node property, not an edge label. Traversal patterns remain unchanged. However, future enhancements could add entity-category-aware traversal (e.g., "find paths that only pass through SYSTEM_STRUCTURE concepts").

## File Changes

### 1. `codebase_rag/constants.py`
- Add `DocConceptEntityCategory` StrEnum with 7 members
- Add `DOC_ENTITY_CATEGORIES: frozenset[str]`
- Add `ENTITY_CATEGORY_EMOJI_MAP: dict[str, str]`

### 2. `codebase_rag/document/concept_extraction.py`
- Update `ExtractedConcept`: add `entity_category`, `entity_subtype`, `entity_emoji` fields; deprecate `type`
- Add `ENTITY_SUBTYPE_REGISTRY: dict[str, str]` (Phase 2 — core 49 sub-types; software 32 can be added incrementally)
- Add `resolve_entity_category(declared_category, declared_subtype, definition) -> tuple[str, str, str]`
- Update `SYSTEM_PROMPT` entity classification section
- Update `LLMConceptExtractor.extract()` to call `resolve_entity_category()` on each concept

### 3. `codebase_rag/document/document_updater.py`
- Update `_merge_concept_nodes_batch()` Cypher SET clause to store entity_category, entity_subtype, entity_emoji
- Update the `concept_nodes` dict in `_extract_and_store_concepts()` to include the three new fields
- Update `_ensure_concept_indexes()` to create indexes on `:Concept(entity_category)` and `:Concept(entity_subtype)`

### 4. `codebase_rag/document/graph_algorithms.py`
- No changes required (entity categories are node properties, not edge labels)
- Optional: add helper method `get_concepts_by_category(category: str) -> list[str]`

### 5. Tests (new file)
- `codebase_rag/tests/test_entity_taxonomy.py`
- Test entity category resolution (valid declared, invalid declared + subtype match, fallback to ABSTRACT_CONCEPT)
- Test emoji always derived from resolved category (not from LLM input)
- Test MECE compliance (7 categories, no overlap)
- Test boundary disambiguation rules (22 rules)
- Test backward compatibility (old `type` field still populated)
- Test sub-type registry integrity (all sub-types map to valid categories)

## Backward Compatibility

### Existing Graph Data

Existing Concept nodes lack `entity_category`, `entity_subtype`, and `entity_emoji` properties. Queries must handle missing properties gracefully:

```cypher
MATCH (c:Concept)
WHERE coalesce(c.entity_category, "UNKNOWN") = "AGENT_ROLE"
RETURN c.name
```

### Old `type` Field

The existing `type` field on `ExtractedConcept` is deprecated but preserved. During extraction, `type` is set equal to `entity_category`. Code reading `concept.type` continues to work. New code should read `concept.entity_category`.

Old values like `"skill"`, `"framework"` are NOT automatically migrated — they remain as historical data on existing nodes. New extractions always use the 7 canonical category labels.

### Relationship Taxonomy

No changes. Entity and relationship taxonomies are independent — a concept's entity category has no bearing on what relationship categories connect it to others. A CONCRETE_ENTITY can have CAUSAL relationships; an ABSTRACT_CONCEPT can have COMPOSITIONAL relationships.

## Implementation Phases

### Phase 1: Core Schema (node classification with 7 categories)
1. Add `DocConceptEntityCategory` StrEnum, `DOC_ENTITY_CATEGORIES`, `ENTITY_CATEGORY_EMOJI_MAP` to `constants.py`
2. Add `entity_category`, `entity_subtype`, `entity_emoji` fields to `ExtractedConcept`; deprecate `type`
3. Add `resolve_entity_category()` function (steps 1-4, without sub-type registry)
4. Update `SYSTEM_PROMPT` entity classification section
5. Update `_store_concepts_batch()` Cypher to store new properties
6. Add graph indexes for `entity_category` and `entity_subtype`

### Phase 2: Sub-Type Precision
7. Add `ENTITY_SUBTYPE_REGISTRY` with 49 core sub-types
8. Add software-domain sub-types (32 entries)
9. Wire sub-type registry into `resolve_entity_category()` step 2
10. Add sub-type hinting in SYSTEM_PROMPT for software domain

### Phase 3: Testing
11. Unit tests for entity category resolution (7 categories × declared/fallback paths)
12. Boundary disambiguation tests (22 rules)
13. Sub-type registry integrity tests
14. Integration test with real LLM extraction
15. Verify old `type` field backward compatibility

## Success Criteria

1. ✅ Every extracted concept has exactly one canonical entity category (7 labels)
2. ✅ ABSTRACT_CONCEPT usage is low (used only as explicit last resort)
3. ✅ Software-domain concepts get precise sub-types (e.g., "Container Image" not just "Artifact")
4. ✅ Queries can filter by entity category: `MATCH (c:Concept {entity_category: "AGENT_ROLE"})`
5. ✅ Old `type` field continues to work (populated from entity_category)
6. ✅ Existing Concept nodes without entity_category do not break queries (coalesce fallback)
7. ✅ Relationship taxonomy is unaffected — entity and relationship classification are independent
8. ✅ Emoji always server-resolved from `ENTITY_CATEGORY_EMOJI_MAP`, never from LLM

---

## Appendix A: Full Entity Sub-Type Registry (Phase 2)

```python
ENTITY_SUBTYPE_REGISTRY: dict[str, str] = {
    # 🧱 Concrete Entity — Core (8)
    "Natural Object": "CONCRETE_ENTITY",
    "Artifact": "CONCRETE_ENTITY",
    "Substance": "CONCRETE_ENTITY",
    "Organism": "CONCRETE_ENTITY",
    "Body Part": "CONCRETE_ENTITY",
    "Food/Consumable": "CONCRETE_ENTITY",
    "Geographic Feature": "CONCRETE_ENTITY",
    "Celestial Body": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Software (5)
    "Virtual Machine": "CONCRETE_ENTITY",
    "Container Image": "CONCRETE_ENTITY",
    "Data Center": "CONCRETE_ENTITY",
    "Mobile Device": "CONCRETE_ENTITY",
    "Peripheral": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Business (5)
    "Facility": "CONCRETE_ENTITY",
    "Inventory": "CONCRETE_ENTITY",
    "Equipment": "CONCRETE_ENTITY",
    "Product": "CONCRETE_ENTITY",
    "Prototype": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Scientific (6)
    "Particle": "CONCRETE_ENTITY",
    "Molecule": "CONCRETE_ENTITY",
    "Mineral": "CONCRETE_ENTITY",
    "Fossil": "CONCRETE_ENTITY",
    "Specimen": "CONCRETE_ENTITY",
    "Isotope": "CONCRETE_ENTITY",

    # ⏱️ Event/Process — Core (8)
    "Natural Event": "EVENT_PROCESS",
    "Human Action": "EVENT_PROCESS",
    "Process": "EVENT_PROCESS",
    "Incident": "EVENT_PROCESS",
    "Activity": "EVENT_PROCESS",
    "State Change": "EVENT_PROCESS",
    "Project/Initiative": "EVENT_PROCESS",
    "Ritual/Routine": "EVENT_PROCESS",
    # ⏱️ Event/Process — Software (4; "Incident" already in core above)
    "Deployment": "EVENT_PROCESS",
    "Build": "EVENT_PROCESS",
    "Test Run": "EVENT_PROCESS",
    "Migration": "EVENT_PROCESS",
    # ⏱️ Event/Process — Scientific (6)
    "Chemical Reaction": "EVENT_PROCESS",
    "Biological Process": "EVENT_PROCESS",
    "Mutation": "EVENT_PROCESS",
    "Observation": "EVENT_PROCESS",
    "Geological Event": "EVENT_PROCESS",
    "Astronomical Event": "EVENT_PROCESS",

    # 📨 Information/Expression — Core (7)
    "Data": "INFORMATION_EXPRESSION",
    "Signal": "INFORMATION_EXPRESSION",
    "Symbol": "INFORMATION_EXPRESSION",
    "Narrative": "INFORMATION_EXPRESSION",
    "Code/Formula": "INFORMATION_EXPRESSION",
    "Record/Document": "INFORMATION_EXPRESSION",
    "Media": "INFORMATION_EXPRESSION",
    # 📨 Information/Expression — Software (5)
    "API": "INFORMATION_EXPRESSION",
    "Protocol": "INFORMATION_EXPRESSION",
    "Config File": "INFORMATION_EXPRESSION",
    "Log Stream": "INFORMATION_EXPRESSION",
    "Source File": "INFORMATION_EXPRESSION",

    # 📏 Property/Attribute — Core (7)
    "Physical Quality": "PROPERTY_ATTRIBUTE",
    "Quantitative Measure": "PROPERTY_ATTRIBUTE",
    "Mental State": "PROPERTY_ATTRIBUTE",
    "Capability/Skill": "PROPERTY_ATTRIBUTE",
    "Disposition": "PROPERTY_ATTRIBUTE",
    "Relational Property": "PROPERTY_ATTRIBUTE",
    "Evaluative Property": "PROPERTY_ATTRIBUTE",
    # 📏 Property/Attribute — Software (3)
    "SLI": "PROPERTY_ATTRIBUTE",
    "Quality Attribute": "PROPERTY_ATTRIBUTE",
    "Capacity Metric": "PROPERTY_ATTRIBUTE",
    # 📏 Property/Attribute — Scientific (4)
    "Chemical Property": "PROPERTY_ATTRIBUTE",
    "Biological Trait": "PROPERTY_ATTRIBUTE",
    "Quantum State": "PROPERTY_ATTRIBUTE",
    "Ecological Indicator": "PROPERTY_ATTRIBUTE",

    # 🏗️ System/Structure — Core (7)
    "Natural System": "SYSTEM_STRUCTURE",
    "Social System": "SYSTEM_STRUCTURE",
    "Technological System": "SYSTEM_STRUCTURE",
    "Network": "SYSTEM_STRUCTURE",
    "Hierarchy": "SYSTEM_STRUCTURE",
    "Framework": "SYSTEM_STRUCTURE",
    "Market/Platform": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Software (5)
    "Distributed System": "SYSTEM_STRUCTURE",
    "CI/CD Pipeline": "SYSTEM_STRUCTURE",
    "Monorepo": "SYSTEM_STRUCTURE",
    "Service Mesh": "SYSTEM_STRUCTURE",
    "Feature Flag System": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Business (6)
    "Org Chart": "SYSTEM_STRUCTURE",
    "Holding Company": "SYSTEM_STRUCTURE",
    "Joint Venture": "SYSTEM_STRUCTURE",
    "Franchise": "SYSTEM_STRUCTURE",
    "Cooperative": "SYSTEM_STRUCTURE",
    "Supply Chain": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Scientific (4)
    "Biome": "SYSTEM_STRUCTURE",
    "Watershed": "SYSTEM_STRUCTURE",
    "Geological Formation": "SYSTEM_STRUCTURE",
    "Star System": "SYSTEM_STRUCTURE",

    # 🎭 Agent/Role — Core (6)
    "Individual": "AGENT_ROLE",
    "Collective": "AGENT_ROLE",
    "Institutional Agent": "AGENT_ROLE",
    "Non-Human Agent": "AGENT_ROLE",
    "Role/Position": "AGENT_ROLE",
    "Persona": "AGENT_ROLE",
    # 🎭 Agent/Role — Software (4)
    "CI Bot": "AGENT_ROLE",
    "Service Account": "AGENT_ROLE",
    "End User": "AGENT_ROLE",
    "On-Call Engineer": "AGENT_ROLE",
    # 🎭 Agent/Role — Business (6)
    "Stakeholder": "AGENT_ROLE",
    "Vendor/Supplier": "AGENT_ROLE",
    "Regulator": "AGENT_ROLE",
    "Board": "AGENT_ROLE",
    "Founder": "AGENT_ROLE",
    "Customer/Client": "AGENT_ROLE",

    # 💡 Abstract Concept — Core (6)
    "Domain/Discipline": "ABSTRACT_CONCEPT",
    "Theory/Model": "ABSTRACT_CONCEPT",
    "Principle/Rule": "ABSTRACT_CONCEPT",
    "Value/Ideal": "ABSTRACT_CONCEPT",
    "Category/Class": "ABSTRACT_CONCEPT",
    "Relation/Connection": "ABSTRACT_CONCEPT",
    # 💡 Abstract Concept — Software (5)
    "Design Pattern": "ABSTRACT_CONCEPT",
    "Algorithm": "ABSTRACT_CONCEPT",
    "Protocol Spec": "ABSTRACT_CONCEPT",
    "Paradigm": "ABSTRACT_CONCEPT",
    "SLA": "ABSTRACT_CONCEPT",
    # 💡 Abstract Concept — Business (5)
    "Business Model": "ABSTRACT_CONCEPT",
    "Strategy": "ABSTRACT_CONCEPT",
    "KPI": "ABSTRACT_CONCEPT",
    "Brand": "ABSTRACT_CONCEPT",
    "Moat": "ABSTRACT_CONCEPT",
}
```

Total: 122 sub-types (49 core + 73 domain extensions) across 7 canonical categories.

---

## Version

v1.0.0 | 2026-04-25 | Scope: Entity category classification for concept extraction nodes
