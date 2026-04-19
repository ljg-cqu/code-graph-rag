from __future__ import annotations

import json
import re
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from re import Pattern
from typing import Any

from loguru import logger

from .config import settings
from .utils.token_utils import count_tokens


@dataclass
class CompressionResult:
    original_context: list[dict[str, Any]]
    compressed_context: list[dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    reduction_pct: float
    retention_score: float
    strategy_used: str
    execution_time: float
    archive_id: str | None = None
    was_rolled_back: bool = False
    compression_rationale: str | None = None  # LLM explanation of what was kept and why
    task_state: Any | None = None  # Structured TaskState from semantic compression


@dataclass
class StrategyEvaluationResult:
    strategy_id: str
    strategy_name: str
    compressed_context: list[dict[str, Any]]
    compressed_tokens: int
    semantic_retention_score: float
    token_reduction_pct: float
    execution_time: float
    total_score: float


class ContextArchive:
    _archive: OrderedDict[str, tuple[list[dict[str, Any]], datetime]] = OrderedDict()
    _lock = threading.Lock()
    _ttl = timedelta(hours=settings.CONTEXT_COMPRESSION_ARCHIVE_TTL_HOURS)

    @classmethod
    def store(cls, context: list[dict[str, Any]]) -> str:
        with cls._lock:
            archive_id = f"ctx_arc_{int(time.time())}_{hash(json.dumps(context, sort_keys=True))}"
            cls._archive[archive_id] = (context, datetime.now(UTC))
            cls._evict_expired()
            logger.debug(f"Stored context in archive with ID: {archive_id}")
            return archive_id

    @classmethod
    def retrieve(cls, archive_id: str) -> list[dict[str, Any]] | None:
        with cls._lock:
            if archive_id not in cls._archive:
                return None
            context, stored_at = cls._archive[archive_id]
            if datetime.now(UTC) - stored_at > cls._ttl:
                del cls._archive[archive_id]
                return None
            cls._archive.move_to_end(archive_id)
            return context

    @classmethod
    def _evict_expired(cls) -> None:
        now = datetime.now(UTC)
        expired_ids = [
            arc_id
            for arc_id, (_, stored_at) in cls._archive.items()
            if now - stored_at > cls._ttl
        ]
        for arc_id in expired_ids:
            del cls._archive[arc_id]
        if expired_ids:
            logger.debug(f"Evicted {len(expired_ids)} expired context archive entries")


class ContextCompressor:
    STRATEGIES = [
        ("S1", "_hierarchical_summarization", "Hierarchical Semantic Summarization"),
        ("S2", "_stale_context_pruning", "Stale Context Pruning"),
        ("S3", "_semantic_ranking_filter", "Semantic Ranking Filter"),
        ("S4", "_token_aware_merging", "Token-Aware Merging"),
        (
            "S5",
            "_hybrid_summarization_pruning",
            "Hybrid Summarization + Pruning (Default)",
        ),
    ]

    SCORE_WEIGHTS = {
        "semantic_retention": 0.6,
        "token_reduction": 0.3,
        "execution_speed": 0.1,
    }

    def __init__(
        self,
        context: list[dict[str, Any]],
        max_context: int,
        aggressive_mode: bool = False,
        preserve_pattern: str | None = None,
        min_retention_score: float | None = None,
        worker_count: int | None = None,
    ):
        self.context = context
        self.max_context = max_context
        self.aggressive_mode = aggressive_mode
        self.preserve_pattern: Pattern | None = (
            re.compile(preserve_pattern, re.IGNORECASE) if preserve_pattern else None
        )
        self.min_retention_score = min_retention_score or (
            settings.CONTEXT_COMPRESSION_AGGRESSIVE_RETENTION_THRESHOLD
            if aggressive_mode
            else settings.CONTEXT_COMPRESSION_MIN_RETENTION_SCORE
        )
        self.worker_count = (
            worker_count or settings.CONTEXT_COMPRESSION_PARALLEL_WORKERS
        )
        self.original_tokens = self._count_context_tokens(context)
        self._worker_pool: ThreadPoolExecutor | None = None

    def _get_worker_pool(self) -> ThreadPoolExecutor:
        """Get a ThreadPoolExecutor for CPU-bound compression tasks.

        Note: We use ThreadPoolExecutor instead of ParallelWorkerPool because
        compression strategies are CPU-bound Python operations that don't need
        database connections. ParallelWorkerPool is designed for Memgraph I/O.
        """
        if self._worker_pool is None:
            self._worker_pool = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="compression-worker"
            )
        return self._worker_pool

    def _count_context_tokens(self, context: list[dict[str, Any]]) -> int:
        """Count tokens in message content, excluding JSON structural overhead."""
        total = 0
        for msg in context:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += count_tokens(content)
            else:
                total += count_tokens(json.dumps(content))
        # Add overhead per message (role tokens, formatting)
        total += len(context) * 4  # Approximate overhead: "role", "content", punctuation
        return total

    def _preserve_matching_content(
        self, context: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        preserved = []
        compressible = []

        for msg in context:
            # Always preserve system messages
            if msg.get("role") == "system":
                preserved.append(msg)
            # Preserve tool and function messages (critical execution results)
            elif msg.get("role") in ("tool", "function"):
                preserved.append(msg)
            # Preserve messages matching the pattern
            elif self.preserve_pattern and self.preserve_pattern.search(
                json.dumps(msg)
            ):
                preserved.append(msg)
            else:
                compressible.append(msg)

        logger.debug(
            f"Preserved {len(preserved)} messages (system prompts + pattern matches)"
        )
        return preserved, compressible

    def _truncate_message_to_budget(self, msg: dict[str, Any], token_budget: int) -> dict[str, Any] | None:
        """Truncate a single message's content to fit within token budget.

        Args:
            msg: Message dictionary with 'content' field.
            token_budget: Maximum tokens allowed for this message.

        Returns:
            Truncated message dict, or None if the message cannot fit even with
            minimal content (should be dropped).
        """
        content = msg.get("content", "")
        if not content:
            return msg  # Empty content, fits trivially

        # Convert non-string content to JSON string for truncation
        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except (TypeError, ValueError):
                # Cannot serialize, drop message
                return None

        # Binary search for maximal prefix that fits within token budget
        marker = "... [truncated]"
        low, high = 0, len(content)
        best_fit = 0

        while low <= high:
            mid = (low + high) // 2
            truncated = content[:mid] + marker
            test_msg = {**msg, "content": truncated}
            test_tokens = self._count_context_tokens([test_msg])
            if test_tokens <= token_budget:
                best_fit = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_fit == 0:
            # Cannot fit even with minimal content
            return None

        final_content = content[:best_fit] + marker
        return {**msg, "content": final_content}

    def _hard_truncate_to_budget(
        self, context: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """Truncate context to fit within token budget, preserving system messages and recent context."""
        system_msgs = [m for m in context if m.get("role") == "system"]
        non_system = [m for m in context if m.get("role") != "system"]

        # Keep system messages
        result = list(system_msgs)
        result_tokens = self._count_context_tokens(result)

        # Add recent non-system messages from the end until budget exhausted
        for msg in reversed(non_system):
            msg_tokens = self._count_context_tokens([msg])
            if result_tokens + msg_tokens <= budget:
                result.insert(len(system_msgs), msg)  # Insert after system messages
                result_tokens += msg_tokens
            # Continue evaluating all messages, don't break on first that doesn't fit

        # FIX Bug 4: If still over budget, truncate message content
        # This handles: (1) system messages alone exceed budget, (2) last non-system message exceeds budget
        max_iterations = len(result) * 2  # Prevent infinite loops
        iteration = 0
        while result_tokens > budget and result and iteration < max_iterations:
            iteration += 1
            # Find the newest message that has content we can truncate
            truncated_something = False
            for i in range(len(result) - 1, -1, -1):  # Start from newest
                msg = result[i]
                # Calculate budget for this message if we keep all other messages
                other_messages = result[:i] + result[i+1:]
                other_tokens = self._count_context_tokens(other_messages)
                leftover = budget - other_tokens

                if leftover > 10:
                    # Try to truncate this message to fit within leftover budget
                    truncated_msg = self._truncate_message_to_budget(msg, leftover)
                    if truncated_msg is not None:
                        result[i] = truncated_msg
                        result_tokens = self._count_context_tokens(result)
                        truncated_something = True
                        break
                    else:
                        # Cannot fit even after truncation, remove message
                        result = result[:i] + result[i+1:]
                        result_tokens = self._count_context_tokens(result)
                        truncated_something = True
                        break
                else:
                    # Not enough budget for this message, remove it
                    result = result[:i] + result[i+1:]
                    result_tokens = self._count_context_tokens(result)
                    truncated_something = True
                    break

            if not truncated_something:
                # Can't truncate anything further
                break

        return result

    def _calculate_semantic_retention(
        self, original: list[dict[str, Any]], compressed: list[dict[str, Any]]
    ) -> float:
        # Entity-based score
        entity_score = self._entity_retention_score(original, compressed)

        # Keyword overlap score for user content
        keyword_score = self._keyword_overlap_score(original, compressed)

        # Weighted combination
        return 0.6 * entity_score + 0.4 * keyword_score

    def _entity_retention_score(
        self, original: list[dict[str, Any]], compressed: list[dict[str, Any]]
    ) -> float:
        """Extract and compare code entities (functions, classes, etc.)."""
        def extract_critical_entities(messages: list[dict[str, Any]]) -> set[str]:
            entities = set()
            for msg in messages:
                content = (
                    msg["content"]
                    if isinstance(msg["content"], str)
                    else json.dumps(msg["content"])
                )
                entities.update(re.findall(r"\bdef\s+(\w+)\b", content))
                entities.update(re.findall(r"\bclass\s+(\w+)\b", content))
                entities.update(
                    re.findall(r"\bfunction\s+(\w+)\b", content, re.IGNORECASE)
                )
                entities.update(
                    re.findall(r"\bmethod\s+(\w+)\b", content, re.IGNORECASE)
                )
                entities.update(
                    re.findall(
                        r"\b(error|exception|failed|timeout)\b.*?:.*?$",
                        content,
                        re.IGNORECASE | re.MULTILINE,
                    )
                )
                if msg["role"] == "user":
                    entities.add(content.strip())
            return entities

        original_entities = extract_critical_entities(original)
        compressed_entities = extract_critical_entities(compressed)

        if not original_entities:
            return 1.0

        retention = len(compressed_entities.intersection(original_entities)) / len(
            original_entities
        )
        return max(0.0, min(1.0, retention))

    def _keyword_overlap_score(
        self, original: list[dict[str, Any]], compressed: list[dict[str, Any]]
    ) -> float:
        """Calculate keyword overlap between original and compressed content."""
        def extract_keywords(messages: list[dict[str, Any]]) -> set[str]:
            keywords = set()
            for msg in messages:
                content = (
                    msg["content"].lower()
                    if isinstance(msg["content"], str)
                    else json.dumps(msg["content"]).lower()
                )
                # Extract words with length > 3, excluding common stopwords
                stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy", "did", "she", "use", "her", "way", "many", "oil", "sit", "set", "run", "eat", "far", "sea", "eye", "ago", "off", "too", "any", "say", "man", "try", "ask", "end", "why", "let", "put", "say", "she", "try", "way", "own", "say", "too", "old", "tell", "very", "when", "much", "would", "there", "their", "what", "said", "each", "which", "will", "about", "could", "other", "after", "first", "never", "these", "think", "where", "being", "every", "great", "might", "shall", "still", "those", "while", "this", "that", "with", "have", "from", "they", "know", "want", "been", "good", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"}
                words = re.findall(r"\b[a-z]{4,}\b", content)
                keywords.update(w for w in words if w not in stopwords)
            return keywords

        original_keywords = extract_keywords(original)
        compressed_keywords = extract_keywords(compressed)

        if not original_keywords:
            return 1.0

        overlap = len(original_keywords.intersection(compressed_keywords))
        return max(0.0, min(1.0, overlap / len(original_keywords)))

    def _hierarchical_summarization(
        self, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        warnings.warn(
            "_hierarchical_summarization is deprecated, use SemanticCompressor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # FIX Bug 2: Check token budget, not just message count
        if len(context) <= 3 and self._count_context_tokens(context) <= self.max_context:
            return context

        preserved = context[-2:]
        older_messages = context[:-2]
        summarized = []

        for msg in older_messages:
            content = (
                msg["content"]
                if isinstance(msg["content"], str)
                else json.dumps(msg["content"])
            )
            if len(content) > 500:
                truncate_len = 200 if self.aggressive_mode else 300
                summary = f"[Summary of {msg['role']} message]: {content[:truncate_len]}... [truncated]"
                summarized.append({"role": msg["role"], "content": summary})
            else:
                summarized.append(msg)

        return summarized + preserved

    def _stale_context_pruning(
        self, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        warnings.warn(
            "_stale_context_pruning is deprecated, use SemanticCompressor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # FIX Bug 2: Check token budget, not just message count
        if len(context) <= 5 and self._count_context_tokens(context) <= self.max_context:
            return context

        latest_user_msg = next(
            (m for m in reversed(context) if m["role"] == "user"), None
        )
        if not latest_user_msg:
            return context[-10:]

        latest_query = (
            latest_user_msg["content"].lower()
            if isinstance(latest_user_msg["content"], str)
            else json.dumps(latest_user_msg["content"]).lower()
        )
        scored_messages = []

        for idx, msg in enumerate(context):
            content = (
                msg["content"].lower()
                if isinstance(msg["content"], str)
                else json.dumps(msg["content"]).lower()
            )
            recency_score = idx / len(context) * 0.5
            keyword_matches = sum(1 for word in latest_query.split() if word in content)
            relevance_score = min(
                0.5,
                keyword_matches / len(latest_query.split()) * 0.5
                if latest_query.split()
                else 0,
            )
            total_score = recency_score + relevance_score
            scored_messages.append((total_score, msg))

        threshold = 0.3 if not self.aggressive_mode else 0.15
        kept = [msg for score, msg in scored_messages if score >= threshold]

        user_messages = [m for m in kept if m["role"] == "user"]
        if len(user_messages) < 2:
            kept.extend([m for m in context[-4:] if m not in kept])

        return kept

    def _semantic_ranking_filter(
        self, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        warnings.warn(
            "_semantic_ranking_filter is deprecated, use SemanticCompressor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        seen_hashes = set()
        unique_context = []
        for msg in context:
            msg_hash = hash(json.dumps(msg, sort_keys=True))
            if msg_hash not in seen_hashes:
                seen_hashes.add(msg_hash)
                unique_context.append(msg)

        # FIX Bug 2: Check token budget, not just message count
        if len(unique_context) <= 10 and self._count_context_tokens(unique_context) <= self.max_context:
            return unique_context

        latest_user_msg = next(
            (m for m in reversed(unique_context) if m["role"] == "user"), None
        )
        if not latest_user_msg:
            return unique_context[-10:]

        latest_query = (
            latest_user_msg["content"].lower()
            if isinstance(latest_user_msg["content"], str)
            else json.dumps(latest_user_msg["content"]).lower()
        )
        scored = []

        for msg in unique_context:
            content = (
                msg["content"].lower()
                if isinstance(msg["content"], str)
                else json.dumps(msg["content"]).lower()
            )
            overlap = len(set(latest_query.split()) & set(content.split()))
            score = overlap / len(latest_query.split()) if latest_query.split() else 0
            scored.append((score, msg))

        scored.sort(reverse=True, key=lambda x: x[0])
        keep_ratio = 0.4 if self.aggressive_mode else 0.7
        keep_count = max(5, int(len(scored) * keep_ratio))
        kept = [msg for _, msg in scored[:keep_count]]

        kept.sort(key=lambda x: context.index(x))
        return kept

    def _token_aware_merging(
        self, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        warnings.warn(
            "_token_aware_merging is deprecated, use SemanticCompressor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        merged = []
        last_role = None
        last_content = []

        for msg in context:
            content = (
                msg["content"]
                if isinstance(msg["content"], str)
                else json.dumps(msg["content"])
            )
            content = re.sub(r"#.*?$", "", content, flags=re.MULTILINE)
            content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s+", " ", content).strip()

            if msg["role"] == last_role:
                last_content.append(content)
            else:
                if last_role is not None:
                    merged.append(
                        {"role": last_role, "content": "\n---\n".join(last_content)}
                    )
                last_role = msg["role"]
                last_content = [content]

        if last_role is not None:
            merged.append({"role": last_role, "content": "\n---\n".join(last_content)})

        return merged

    def _hybrid_summarization_pruning(
        self, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        warnings.warn(
            "_hybrid_summarization_pruning is deprecated, use SemanticCompressor instead",
            DeprecationWarning,
            stacklevel=2,
        )
        system_prompts = [m for m in context if m["role"] == "system"]
        user_messages = [m for m in context if m["role"] == "user"]
        preserved_recent = (
            context[-4:]
            if len(user_messages) < 2
            else context[context.index(user_messages[-2]) :]
        )
        tool_messages = [m for m in context if m["role"] in ("tool", "function")]

        older_messages = [
            m
            for m in context
            if m not in system_prompts
            and m not in preserved_recent
            and m not in tool_messages
        ]

        summarized_older = []
        if older_messages:
            summary = f"[{len(older_messages)} older messages summarized]:\n"
            key_points = set()
            for msg in older_messages:
                content = (
                    msg["content"]
                    if isinstance(msg["content"], str)
                    else json.dumps(msg["content"])
                )
                matches = re.findall(
                    r"\b(def|class|function|error|exception|failed|timeout|call|invoke|run)\s+(\w+)\b",
                    content,
                    re.IGNORECASE,
                )
                for match in matches:
                    key_points.add(f"{match[0]} {match[1]}")
            summary += "Key points: " + ", ".join(str(k) for k in key_points)
            summarized_older.append({"role": "assistant", "content": summary})

        final_context = (
            system_prompts + summarized_older + tool_messages + preserved_recent
        )
        seen = set()
        unique_final = []
        for msg in final_context:
            h = hash(json.dumps(msg, sort_keys=True))
            if h not in seen:
                seen.add(h)
                unique_final.append(msg)
        return unique_final

    def _evaluate_strategy(self, params: dict[str, Any]) -> StrategyEvaluationResult:
        strategy_id = params["strategy_id"]
        strategy_func = params["strategy_func"]
        context = params["context"]
        original_compressible_tokens = self._count_context_tokens(context)
        start_time = time.time()
        try:
            compressed_context = strategy_func(context.copy())
            execution_time = time.time() - start_time

            compressed_tokens = self._count_context_tokens(compressed_context)
            token_reduction_pct = (
                (original_compressible_tokens - compressed_tokens) / original_compressible_tokens
                if original_compressible_tokens > 0
                else 0
            )
            semantic_retention = self._calculate_semantic_retention(
                context, compressed_context
            )

            speed_score = min(1.0, max(0.0, 1.0 - (execution_time * 10)))
            # Adjust weights for aggressive mode: prioritize higher compression
            if self.aggressive_mode:
                ret_weight = 0.4
                comp_weight = 0.5
            else:
                ret_weight = self.SCORE_WEIGHTS["semantic_retention"]
                comp_weight = self.SCORE_WEIGHTS["token_reduction"]

            total_score = (
                semantic_retention * ret_weight
                + token_reduction_pct * comp_weight
                + speed_score * self.SCORE_WEIGHTS["execution_speed"]
            )

            return StrategyEvaluationResult(
                strategy_id=strategy_id,
                strategy_name=strategy_func.__name__,
                compressed_context=compressed_context,
                compressed_tokens=compressed_tokens,
                semantic_retention_score=semantic_retention,
                token_reduction_pct=token_reduction_pct,
                execution_time=execution_time,
                total_score=total_score,
            )
        except Exception as e:
            logger.error(f"Strategy {strategy_id} evaluation failed: {str(e)}")
            return StrategyEvaluationResult(
                strategy_id=strategy_id,
                strategy_name=strategy_func.__name__,
                compressed_context=context,
                compressed_tokens=original_compressible_tokens,
                semantic_retention_score=1.0,
                token_reduction_pct=0.0,
                execution_time=time.time() - start_time,
                total_score=0.0,
            )

    def compress_sync(self) -> CompressionResult:
        start_time = time.time()

        if self.original_tokens == 0:
            return CompressionResult(
                original_context=self.context,
                compressed_context=self.context,
                original_tokens=0,
                compressed_tokens=0,
                reduction_pct=0.0,
                retention_score=1.0,
                strategy_used="no-op",
                execution_time=time.time() - start_time,
            )

        preserved_content, compressible_content = self._preserve_matching_content(
            self.context
        )

        # Always compress if we're over budget, even with few messages
        if self.original_tokens <= self.max_context and len(compressible_content) < 3:
            return CompressionResult(
                original_context=self.context,
                compressed_context=self.context,
                original_tokens=self.original_tokens,
                compressed_tokens=self.original_tokens,
                reduction_pct=0.0,
                retention_score=1.0,
                strategy_used="no-op",
                execution_time=time.time() - start_time,
            )

        # FIX Bug 5: Only use thread pool for non-trivial workloads
        task_funcs = [getattr(self, func_name) for (_, func_name, _) in self.STRATEGIES]
        tasks = [
            {
                "strategy_id": strategy_id,
                "strategy_func": func,
                "context": compressible_content,
            }
            for (strategy_id, _, _), func in zip(self.STRATEGIES, task_funcs)
        ]

        compressible_tokens = self._count_context_tokens(compressible_content)
        if len(self.STRATEGIES) <= 2 or compressible_tokens < 500:
            # Sequential evaluation for small workloads
            results = [self._evaluate_strategy(task) for task in tasks]
        else:
            pool = self._get_worker_pool()
            try:
                # Submit all strategy evaluations to the thread pool
                futures = [pool.submit(self._evaluate_strategy, task) for task in tasks]
                results = [f.result() for f in futures]
            finally:
                # Shutdown the thread pool to free resources
                pool.shutdown(wait=True)

        valid_results = [
            res for res in results if isinstance(res, StrategyEvaluationResult)
        ]

        if not valid_results:
            logger.warning(
                "No valid compression strategy results, falling back to default hybrid strategy"
            )
            best_result = self._evaluate_strategy(
                {
                    "strategy_id": "S5",
                    "strategy_func": self._hybrid_summarization_pruning,
                    "context": compressible_content,
                },
            )
        else:
            valid_results.sort(reverse=True, key=lambda x: x.total_score)
            best_result = valid_results[0]

        final_compressed = preserved_content + best_result.compressed_context
        final_tokens = self._count_context_tokens(final_compressed)

        # Ensure compressed result fits within max_context budget
        if final_tokens > self.max_context:
            logger.warning(
                f"Best strategy exceeded max_context ({final_tokens} > {self.max_context}), "
                "applying hard truncation"
            )
            final_compressed = self._hard_truncate_to_budget(
                preserved_content + best_result.compressed_context, self.max_context
            )
            final_tokens = self._count_context_tokens(final_compressed)

        reduction_pct = (
            (self.original_tokens - final_tokens) / self.original_tokens
            if self.original_tokens > 0
            else 0.0
        )

        was_rolled_back = False
        # min_retention_score is stored as percentage (e.g. 70 = 70%), convert to decimal for comparison
        min_retention_decimal = self.min_retention_score / 100

        # FIX Bug 1: Only rollback if original fits within budget AND retention is too low.
        # If original exceeds budget, hard-truncated result is the best possible fit.
        original_fits_budget = self.original_tokens <= self.max_context
        if original_fits_budget and best_result.semantic_retention_score < min_retention_decimal:
            logger.warning(
                f"Compression retention score {best_result.semantic_retention_score * 100:.2f}% "
                f"below minimum {self.min_retention_score:.2f}%, rolling back to original"
            )
            final_compressed = self.context
            final_tokens = self.original_tokens
            reduction_pct = 0.0
            was_rolled_back = True

        archive_id = ContextArchive.store(self.context) if not was_rolled_back else None

        return CompressionResult(
            original_context=self.context,
            compressed_context=final_compressed,
            original_tokens=self.original_tokens,
            compressed_tokens=final_tokens,
            reduction_pct=reduction_pct,
            retention_score=best_result.semantic_retention_score,
            strategy_used=f"{best_result.strategy_id}: {best_result.strategy_name}",
            execution_time=time.time() - start_time,
            archive_id=archive_id,
            was_rolled_back=was_rolled_back,
        )


    async def compress_async(
        self, pending_query: str | None = None
    ) -> CompressionResult:
        """Asynchronous wrapper that delegates to SemanticCompressor.

        Provides backward compatibility for callers migrating from sync to async.

        Args:
            pending_query: Optional query to optimize compression for (query-aware mode).
        """
        from .semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor(
            context=self.context,
            max_context=self.max_context,
            pending_query=pending_query,
            aggressive_mode=self.aggressive_mode,
            preserve_pattern=self.preserve_pattern.pattern
            if self.preserve_pattern
            else None,
        )
        return await compressor.compress()
