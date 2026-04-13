from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import get_config
from .cypher_queries import build_merge_node_query, build_merge_relationship_query
from .embedder import EmbeddingCache, get_embedding_provider_instance
from .logs import get_logger
from .schemas import IngestionResult, JSONInputSchema, UpdateResult
from .services.graph_service import GraphService
from .vector_store import get_vector_store_instance

__all__ = [
    "ingest_json_data",
    "delete_dataset",
    "handle_json_update_event",
    "validate_json_input",
    "load_json_files",
    "IngestionResult",
    "UpdateResult",
]

logger = get_logger(__name__)
config = get_config()

embedding_provider = get_embedding_provider_instance()
embedding_cache = EmbeddingCache()
vector_store = get_vector_store_instance()
graph_service = GraphService()


def validate_json_input(
    data: dict[str, Any],
) -> tuple[bool, JSONInputSchema | None, list[str]]:
    """Validate JSON input against the schema."""
    errors = []
    try:
        validated = JSONInputSchema(**data)
        return True, validated, []
    except Exception as e:
        errors.append(f"Schema validation failed: {str(e)}")
        return False, None, errors


def load_json_files(input_path: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load JSON files from a path (single file or directory)."""
    path = Path(input_path)
    json_files = []

    if path.is_file() and path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            json_files.append((path, json.load(f)))
    elif path.is_dir():
        for file_path in path.rglob("*.json"):
            with open(file_path, encoding="utf-8") as f:
                json_files.append((file_path, json.load(f)))
    else:
        raise ValueError(
            f"Invalid input path: {input_path} (must be .json file or directory containing JSON files)"
        )

    return json_files


def generate_embeddings_for_entities(
    entities: list[Any],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for entity descriptions."""
    embeddings = {}
    errors = []

    texts = []
    entity_ids = []
    for entity in entities:
        text = entity.properties["description"]
        texts.append(text)
        entity_ids.append(entity.id)

    # Check cache first
    cached_embeddings = embedding_cache.get_batch(texts)
    to_generate = []
    to_generate_ids = []

    for i, (text, eid) in enumerate(zip(texts, entity_ids)):
        if cached_embeddings[i] is not None:
            embeddings[eid] = cached_embeddings[i]
        else:
            to_generate.append(text)
            to_generate_ids.append(eid)

    # Generate missing embeddings
    if to_generate:
        try:
            generated = embedding_provider.encode_batch(to_generate)
            for eid, text, emb in zip(to_generate_ids, to_generate, generated):
                embeddings[eid] = emb
                embedding_cache.set(text, emb)
        except Exception as e:
            errors.append(f"Embedding generation failed: {str(e)}")

    return embeddings, errors


def generate_embeddings_for_relationships(
    relationships: list[Any],
) -> tuple[dict[str, list[float]], list[str]]:
    """Generate embeddings for relationship descriptions (if present)."""
    embeddings = {}
    errors = []

    texts = []
    rel_ids = []
    for rel in relationships:
        if "description" in rel.properties:
            text = rel.properties["description"]
            texts.append(text)
            rel_ids.append(
                rel.id
                if rel.id
                else f"{rel.source_entity_id}_{rel.target_entity_id}_{rel.type}"
            )

    # Check cache first
    cached_embeddings = embedding_cache.get_batch(texts)
    to_generate = []
    to_generate_ids = []

    for i, (text, rid) in enumerate(zip(texts, rel_ids)):
        if cached_embeddings[i] is not None:
            embeddings[rid] = cached_embeddings[i]
        else:
            to_generate.append(text)
            to_generate_ids.append(rid)

    # Generate missing embeddings
    if to_generate:
        try:
            generated = embedding_provider.encode_batch(to_generate)
            for rid, text, emb in zip(to_generate_ids, to_generate, generated):
                embeddings[rid] = emb
                embedding_cache.set(text, emb)
        except Exception as e:
            errors.append(f"Relationship embedding generation failed: {str(e)}")

    return embeddings, errors


def ingest_entities(
    dataset_id: str,
    entities: list[Any],
    entity_embeddings: dict[str, list[float]],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    conflict_resolution: str = "last-write-wins",
    last_updated_threshold: str | None = None,
) -> tuple[int, int, int, int, list[str]]:
    """Ingest entities into graph and vector store."""
    ingested = 0
    updated = 0
    skipped = 0
    failed = 0
    errors = []

    vector_entries = []

    for entity in tqdm(entities, desc="Processing entities", disable=dry_run):
        try:
            # Prepare node properties
            props = entity.properties.copy()
            props["source_dataset"] = dataset_id
            props["entity_id"] = entity.id
            unique_id = f"{dataset_id}_{entity.id}"

            # Check existing entity for incremental/skip_existing
            existing_node = None
            if not dry_run and (skip_existing or incremental):
                check_query = """
                MATCH (n {unique_id: $unique_id})
                RETURN n.updated_at as updated_at
                """
                check_result = graph_service.run_query(
                    check_query, {"unique_id": unique_id}
                )
                if check_result and check_result[0][0] is not None:
                    existing_node = check_result[0][0]

                    # Skip if skip_existing is enabled
                    if skip_existing:
                        skipped += 1
                        continue

                    # Skip if incremental and existing entry is newer/equal
                    if incremental and last_updated_threshold:
                        if existing_node >= last_updated_threshold:
                            skipped += 1
                            continue

            # Merge node into graph
            if not dry_run:
                labels = ":".join(entity.labels)
                query = build_merge_node_query(labels, "unique_id = $unique_id")
                params = {"unique_id": unique_id, "properties": props}
                result = graph_service.run_query(query, params)
                # Check if node was created or updated
                node = result[0][0] if result else None
                if node:
                    node_id = node.id
                    if result[0][1]:  # created flag
                        ingested += 1
                    else:
                        updated += 1
                else:
                    failed += 1
                    errors.append(f"Failed to ingest entity {entity.id}")
                    continue
            else:
                # Dry run: just count
                ingested += 1
                node_id = 0  # Dummy ID for dry run

            # Prepare vector entry
            if entity.id in entity_embeddings:
                vector_meta = {
                    "node_id": node_id,
                    "dataset_id": dataset_id,
                    "entity_id": entity.id,
                    "labels": entity.labels,
                    "name": entity.properties["name"],
                    **entity.properties,
                }
                vector_entries.append(
                    {"embedding": entity_embeddings[entity.id], "metadata": vector_meta}
                )

        except Exception as e:
            failed += 1
            errors.append(f"Error processing entity {entity.id}: {str(e)}")

    # Store vectors in batch
    if vector_entries and not dry_run:
        try:
            vector_store.store_embedding_batch(
                [ve["embedding"] for ve in vector_entries],
                [ve["metadata"] for ve in vector_entries],
            )
        except Exception as e:
            failed += len(vector_entries)
            errors.append(f"Failed to store entity embeddings: {str(e)}")

    return ingested, updated, skipped, failed, errors


def ingest_relationships(
    dataset_id: str,
    relationships: list[Any],
    rel_embeddings: dict[str, list[float]],
    skip_existing: bool = False,
    dry_run: bool = False,
    incremental: bool = False,
    conflict_resolution: str = "last-write-wins",
    last_updated_threshold: str | None = None,
) -> tuple[int, int, int, int, list[str]]:
    """Ingest relationships into graph and vector store."""
    ingested = 0
    updated = 0
    skipped = 0
    failed = 0
    errors = []

    vector_entries = []

    for rel in tqdm(relationships, desc="Processing relationships", disable=dry_run):
        try:
            # Prepare relationship properties
            props = rel.properties.copy()
            props["source_dataset"] = dataset_id
            if rel.id:
                props["relationship_id"] = rel.id

            source_unique_id = f"{dataset_id}_{rel.source_entity_id}"
            target_unique_id = f"{dataset_id}_{rel.target_entity_id}"

            # Check existing relationship for incremental/skip_existing
            existing_rel = None
            if not dry_run and (skip_existing or incremental):
                check_query = """
                MATCH (s {{unique_id: $source_id}})-[r:{rel_type}]->(t {{unique_id: $target_id}})
                RETURN r.updated_at as updated_at
                """.format(rel_type=rel.type)
                check_result = graph_service.run_query(
                    check_query,
                    {"source_id": source_unique_id, "target_id": target_unique_id},
                )
                if check_result and check_result[0][0] is not None:
                    existing_rel = check_result[0][0]

                    # Skip if skip_existing is enabled
                    if skip_existing:
                        skipped += 1
                        continue

                    # Skip if incremental and existing entry is newer/equal
                    if incremental and last_updated_threshold:
                        if existing_rel >= last_updated_threshold:
                            skipped += 1
                            continue

            # Merge relationship into graph
            if not dry_run:
                query = build_merge_relationship_query(
                    rel.type,
                    "unique_id = $source_id",
                    "unique_id = $target_id",
                )
                params = {
                    "source_id": source_unique_id,
                    "target_id": target_unique_id,
                    "properties": props,
                }
                result = graph_service.run_query(query, params)
                # Check if relationship was created or updated
                rel_node = result[0][0] if result else None
                if rel_node:
                    rel_id = rel_node.id
                    if result[0][1]:  # created flag
                        ingested += 1
                    else:
                        updated += 1
                else:
                    failed += 1
                    errors.append(
                        f"Failed to ingest relationship {rel.source_entity_id}->{rel.target_entity_id} [{rel.type}]"
                    )
                    continue
            else:
                # Dry run: just count
                ingested += 1
                rel_id = 0  # Dummy ID for dry run

            # Prepare vector entry if description exists
            rel_key = (
                rel.id
                if rel.id
                else f"{rel.source_entity_id}_{rel.target_entity_id}_{rel.type}"
            )
            if rel_key in rel_embeddings:
                vector_meta = {
                    "relationship_id": rel_id,
                    "dataset_id": dataset_id,
                    "source_entity_id": rel.source_entity_id,
                    "target_entity_id": rel.target_entity_id,
                    "type": rel.type,
                    **rel.properties,
                }
                vector_entries.append(
                    {"embedding": rel_embeddings[rel_key], "metadata": vector_meta}
                )

        except Exception as e:
            failed += 1
            errors.append(
                f"Error processing relationship {rel.source_entity_id}->{rel.target_entity_id} [{rel.type}]: {str(e)}"
            )

    # Store vectors in batch
    if vector_entries and not dry_run:
        try:
            vector_store.store_embedding_batch(
                [ve["embedding"] for ve in vector_entries],
                [ve["metadata"] for ve in vector_entries],
            )
        except Exception as e:
            failed += len(vector_entries)
            errors.append(f"Failed to store relationship embeddings: {str(e)}")

    return ingested, updated, skipped, failed, errors


def delete_dataset(
    dataset_id: str, dry_run: bool = False
) -> tuple[bool, int, int, list[str]]:
    """Delete all nodes, relationships, and vector entries for a dataset."""
    errors = []
    nodes_deleted = 0
    rels_deleted = 0

    try:
        if not dry_run:
            # Delete relationships first
            rel_query = """
            MATCH ()-[r]->()
            WHERE r.source_dataset = $dataset_id
            DELETE r
            RETURN count(r) as deleted
            """
            rel_result = graph_service.run_query(rel_query, {"dataset_id": dataset_id})
            rels_deleted = rel_result[0][0] if rel_result else 0

            # Delete nodes
            node_query = """
            MATCH (n)
            WHERE n.source_dataset = $dataset_id
            DELETE n
            RETURN count(n) as deleted
            """
            node_result = graph_service.run_query(
                node_query, {"dataset_id": dataset_id}
            )
            nodes_deleted = node_result[0][0] if node_result else 0

            # Delete vector entries
            vector_store.delete_by_metadata({"dataset_id": dataset_id})

        logger.info(
            f"Deleted dataset {dataset_id}: {nodes_deleted} nodes, {rels_deleted} relationships"
        )
        return True, nodes_deleted, rels_deleted, errors

    except Exception as e:
        errors.append(f"Failed to delete dataset {dataset_id}: {str(e)}")
        return False, nodes_deleted, rels_deleted, errors


def ingest_json_data(
    input_path: str = "",
    dataset_id: str | None = None,
    skip_existing: bool = False,
    batch_size: int = 100,
    incremental: bool = False,
    dry_run: bool = False,
    conflict_resolution: str = "last-write-wins",
    pre_loaded_data: list[tuple[Path, dict[str, Any]]] | None = None,
) -> IngestionResult:
    """
    Ingest JSON data into graph and vector database.

    Returns: IngestionResult with counts of entities/relationships processed, ingested, updated, deleted, skipped, failed
    """
    result = IngestionResult(dataset_id=dataset_id or "", dry_run=dry_run)

    try:
        # Load JSON files or use pre-loaded data
        if pre_loaded_data is not None:
            json_files = pre_loaded_data
        else:
            json_files = load_json_files(input_path)
        logger.info(f"Loaded {len(json_files)} JSON file(s)")

        for file_path, data in json_files:
            logger.info(f"Processing file: {file_path}")

            # Validate input
            valid, validated_data, validation_errors = validate_json_input(data)
            if not valid or not validated_data:
                result.errors.extend(validation_errors)
                logger.error(f"Validation failed for {file_path}: {validation_errors}")
                continue

            # Override dataset ID if provided
            current_dataset_id = dataset_id or validated_data.metadata.dataset_id
            result.dataset_id = current_dataset_id

            # Count entities and relationships to process
            result.entities_processed += len(validated_data.entities)
            result.relationships_processed += len(validated_data.relationships)

            # Generate embeddings
            logger.info("Generating embeddings...")
            entity_embeddings, embed_errors = generate_embeddings_for_entities(
                validated_data.entities
            )
            result.errors.extend(embed_errors)

            rel_embeddings, rel_embed_errors = generate_embeddings_for_relationships(
                validated_data.relationships
            )
            result.errors.extend(rel_embed_errors)

            # Ingest entities
            logger.info("Ingesting entities...")
            e_ingested, e_updated, e_skipped, e_failed, e_errors = ingest_entities(
                current_dataset_id,
                validated_data.entities,
                entity_embeddings,
                skip_existing=skip_existing,
                dry_run=dry_run,
                incremental=incremental,
                conflict_resolution=conflict_resolution,
                last_updated_threshold=validated_data.metadata.last_updated,
            )
            result.entities_ingested += e_ingested
            result.entities_updated += e_updated
            result.entities_skipped += e_skipped
            result.entities_failed += e_failed
            result.errors.extend(e_errors)

            # Ingest relationships
            logger.info("Ingesting relationships...")
            r_ingested, r_updated, r_skipped, r_failed, r_errors = ingest_relationships(
                current_dataset_id,
                validated_data.relationships,
                rel_embeddings,
                skip_existing=skip_existing,
                dry_run=dry_run,
                incremental=incremental,
                conflict_resolution=conflict_resolution,
                last_updated_threshold=validated_data.metadata.last_updated,
            )
            result.relationships_ingested += r_ingested
            result.relationships_updated += r_updated
            result.relationships_skipped += r_skipped
            result.relationships_failed += r_failed
            result.errors.extend(r_errors)

        logger.info(
            f"Ingestion completed: {result.entities_ingested} entities ingested, {result.relationships_ingested} relationships ingested"
        )
        return result

    except Exception as e:
        result.errors.append(f"Ingestion failed: {str(e)}")
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        return result


def handle_json_update_event(
    event: dict[str, Any],
    dataset_id: str,
    conflict_resolution: str = "last-write-wins",
    dry_run: bool = False,
) -> UpdateResult:
    """Process single incremental JSON update event from streaming source (Kafka/RabbitMQ/webhook)."""
    from datetime import datetime

    from .schemas import UpdateResult

    event_id = event.get("id")
    operation = event.get("operation", "add").lower()

    logger.info(
        f"Processing update event {event_id}: operation={operation}, dataset={dataset_id}"
    )

    # Create temporary JSON input structure from event
    json_data = {
        "metadata": {
            "dataset_id": dataset_id,
            "operation": operation,
            "last_updated": event.get("timestamp", datetime.utcnow().isoformat()),
        },
        "entities": event.get("entities", []),
        "relationships": event.get("relationships", []),
    }

    # Validate event data
    valid, validated_data, validation_errors = validate_json_input(json_data)
    if not valid or not validated_data:
        return UpdateResult(
            dataset_id=dataset_id,
            operation=operation,
            event_id=event_id,
            processed_at=datetime.utcnow().isoformat(),
            errors=validation_errors,
            entities_failed=len(json_data.get("entities", [])),
            relationships_failed=len(json_data.get("relationships", [])),
            dry_run=dry_run,
        )

    # Ingest event data with incremental mode enabled
    result = ingest_json_data(
        pre_loaded_data=[(Path("event_stream.json"), json_data)],
        dataset_id=dataset_id,
        incremental=True,
        conflict_resolution=conflict_resolution,
        dry_run=dry_run,
    )

    # Convert to UpdateResult
    return UpdateResult(
        **result.model_dump(),
        operation=operation,
        event_id=event_id,
        processed_at=datetime.utcnow().isoformat(),
    )
