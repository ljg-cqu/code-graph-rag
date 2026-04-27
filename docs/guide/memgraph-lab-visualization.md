---
description: "Customize Memgraph Lab visualization for Code-Graph-RAG graphs."
---

# Memgraph Lab Visualization

Code-Graph-RAG stores rich metadata on nodes and relationships that can be visualized in Memgraph Lab using Graph Style Script (GSS).

## Installing the Custom Style

1. Open Memgraph Lab in your browser (default: `http://localhost:3000`)

2. Navigate to **Graph Style** in the left sidebar

3. Click **Import** and select the GSS file:
   ```
   utils/memgraph_lab_style.gss
   ```

4. The style will be applied automatically to all query results

## What the Style Shows

### Nodes

| Property | Display |
|----------|---------|
| Entity name | Shows `entity_emoji + name` (e.g., "🎭 CTO Role") |
| Category colors | 7 distinct colors by entity category |

### Entity Category Colors

| Category | Color | Example |
|----------|-------|---------|
| CONCRETE_ENTITY | 🟠 Orange | Tools, Products |
| EVENT_PROCESS | 🟣 Purple | Deployments, Migrations |
| INFORMATION_EXPRESSION | 🔵 Blue | APIs, Protocols |
| PROPERTY_ATTRIBUTE | 🩵 Teal | Metrics, SLIs |
| SYSTEM_STRUCTURE | 🟢 Green | Frameworks, Layers |
| AGENT_ROLE | 🔴 Red | Roles, Stakeholders |
| ABSTRACT_CONCEPT | ⚪ Gray | Concepts, Patterns |

### Edges

| Property | Display |
|----------|---------|
| Relationship verb | Shows `relationship_emoji + verb` (e.g., "⚡ enables", "🌳 contains") |
| Category colors | 8 distinct colors by relationship category |

### Relationship Category Colors

| Category | Color | Meaning |
|----------|-------|---------|
| CAUSAL | 🔴 Red | Influences, causes |
| HIERARCHICAL | 🔵 Blue | Is-a, subtype |
| COMPOSITIONAL | 🟢 Green | Contains, part-of |
| CONTEXTUAL | 🟣 Purple | Defines context |
| ATTRIBUTIVE | 🩵 Teal | Has property |
| SEQUENTIAL | 🟠 Orange | Precedes |
| COMPARATIVE | 🟤 Brown | Compares to |
| ANALOGICAL | 🩶 Teal | Similar to |

## Manual Style Application

If you prefer to apply styles manually in Lab, use this minimal GSS:

```javascript
// Show emoji + name on JsonEntity nodes
@NodeStyle And(HasLabel(node, "JsonEntity"), HasProperty(node, "entity_emoji")) {
  label: AsText(Property(node, "entity_emoji"), " ", Property(node, "name"))
}

// Show emoji + verb on edges
@EdgeStyle And(HasProperty(edge, "relationship_emoji"), HasProperty(edge, "verb")) {
  label: AsText(Property(edge, "relationship_emoji"), " ", Property(edge, "verb"))
}
```

## Querying for Visualization

For best results, query nodes with their properties:

```cypher
// View all JSON entities with categories
MATCH (n:JsonEntity)
RETURN n
LIMIT 50

// View specific relationships by category property
MATCH (a:JsonEntity)-[r]->(b:JsonEntity)
WHERE r.category = 'CAUSAL'
RETURN a, r, b

// Explore entity hierarchy
MATCH path = (root:JsonEntity)-[*1..3]->(descendant:JsonEntity)
WHERE ALL(r IN relationships(path) WHERE r.category IN ['HIERARCHICAL', 'COMPOSITIONAL'])
RETURN path
```

## Troubleshooting

### Emojis not showing on existing data

If you ingested data before the emoji properties were added, re-run ingestion:

```bash
# Re-ingest to add emoji properties
cgr ingest-json --input-path /path/to/data.json
```

The ingestion uses MERGE (upsert), so existing entities and relationships will be updated with the new `entity_emoji`, `relationship_category`, and `relationship_emoji` properties without creating duplicates.

### Emojis not showing

1. Ensure entities have `entity_emoji` property:
   ```cypher
   MATCH (n:JsonEntity)
   RETURN n.name, n.entity_emoji
   LIMIT 5
   ```

2. If missing, re-run ingestion with the latest version to populate the property

### Edge verbs/emojis not showing

1. Ensure relationships have `verb` and `relationship_emoji` properties:
   ```cypher
   MATCH ()-[r]->()
   WHERE r.verb IS NOT NULL
   RETURN type(r), r.relationship_emoji, r.verb
   LIMIT 5
   ```

2. If missing, re-run ingestion with the latest version to add these properties

### Colors not applying

1. Verify `entity_category` property exists:
   ```cypher
   MATCH (n:JsonEntity)
   RETURN DISTINCT n.entity_category
   ```

2. Valid categories: `CONCRETE_ENTITY`, `EVENT_PROCESS`, `INFORMATION_EXPRESSION`, `PROPERTY_ATTRIBUTE`, `SYSTEM_STRUCTURE`, `AGENT_ROLE`, `ABSTRACT_CONCEPT`
