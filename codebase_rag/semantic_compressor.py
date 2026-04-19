from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .compression_prompts import (
    build_distillation_prompt,
    reconstruct_messages,
)
from .compression_schemas import CompressedState
from .config import settings
from .context_compressor import CompressionResult, ContextArchive
from .utils.token_utils import count_tokens


class LLMContextDistiller:
    """Wraps PydanticAI agent for context distillation."""

    # Reserve for system prompt (~1k), tools (~5k), schema (~2k), and serialization overhead
    PROMPT_OVERHEAD_RESERVE: int = 10000

    # Serialization overhead factor: _serialize_messages adds "[N] role:\n" prefix per message
    SERIALIZATION_OVERHEAD_FACTOR: float = 1.15  # 15% overhead for message formatting

    def __init__(self, agent: Agent | None = None):
        self._agent = agent

    @property
    def _llm_input_cap(self) -> int:
        """Get the LLM input token cap from settings."""
        return settings.SEMANTIC_COMPRESSION_LLM_INPUT_CAP

    async def distill(
        self,
        messages: list[dict[str, Any]],
        pending_query: str | None,
        max_context: int,
    ) -> tuple[list[dict[str, Any]], CompressedState]:
        """Distill conversation history into compressed state.

        Args:
            messages: Conversation history to compress.
            pending_query: Optional query to optimize compression for.
            max_context: Maximum allowed token budget.

        Returns:
            Tuple of (reconstructed_messages, compressed_state) where
            compressed_state contains the structured TaskState and rationale.
        """
        agent = self._agent or self._create_default_agent()

        # Apply verbatim budget for recent messages (output constraint)
        messages, verbatim_budget_used = self._apply_verbatim_budget(messages, max_context)
        current_tokens = _count_context_tokens(messages)

        # Calculate effective cap accounting for prompt overhead AND serialization overhead
        # Serialization overhead: _serialize_messages adds "[N] role:\n" prefix per message
        serialized_estimate = int(current_tokens * self.SERIALIZATION_OVERHEAD_FACTOR)
        effective_cap = self._llm_input_cap - self.PROMPT_OVERHEAD_RESERVE

        # Apply LLM input cap if still too large (accounting for serialization)
        if serialized_estimate > effective_cap:
            # Target tokens accounting for serialization overhead
            target_tokens = int(effective_cap / self.SERIALIZATION_OVERHEAD_FACTOR)
            messages = self._truncate_input_for_distillation(messages, target_tokens)
            current_tokens = _count_context_tokens(messages)

        prompt = build_distillation_prompt(messages, pending_query, max_context, current_tokens, verbatim_budget_used)

        try:
            result = await agent.run(
                prompt,
                usage_limits=UsageLimits(request_limit=settings.AGENT_REQUEST_LIMIT),
            )
            compressed_state: CompressedState = result.output
            return reconstruct_messages(compressed_state), compressed_state
        except Exception as e:
            logger.warning(f"LLM distillation call failed: {e}")
            raise

    def _truncate_input_for_distillation(
        self, messages: list[dict[str, Any]], cap: int
    ) -> list[dict[str, Any]]:
        """If input exceeds LLM cap, apply hard truncation while preserving system + recent."""
        return _hard_truncate_to_budget(messages, cap)

    def _apply_verbatim_budget(
        self, messages: list[dict[str, Any]], max_context: int
    ) -> tuple[list[dict[str, Any]], int]:
        """Apply verbatim budget to recent messages.

        Args:
            messages: Full conversation history.
            max_context: Maximum allowed token budget.

        Returns:
            Tuple of (messages_with_truncated_recent, verbatim_budget_used)
        """
        if not messages:
            return messages, 0

        verbatim_budget = int(max_context * settings.SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT / 100)
        if verbatim_budget <= 0:
            return messages, 0
        max_recent = settings.SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES

        # Determine recent messages (last N, but ensure we don't exceed length)
        recent_count = min(max_recent, len(messages))
        recent_messages = messages[-recent_count:]
        other_messages = messages[:-recent_count] if recent_count < len(messages) else []

        # Compute token count of recent messages
        recent_tokens = _count_context_tokens(recent_messages)

        # If recent messages exceed verbatim budget, truncate oldest first
        if recent_tokens > verbatim_budget:
            # Drop oldest messages until we fit within budget
            while recent_messages and recent_tokens > verbatim_budget:
                # Remove oldest message (first in the recent_messages list)
                removed = recent_messages.pop(0)
                recent_tokens = _count_context_tokens(recent_messages)

            # If we still exceed budget (unlikely), truncate message content
            if recent_messages and recent_tokens > verbatim_budget:
                # Apply hard truncation to remaining messages
                truncated_recent = _hard_truncate_to_budget(recent_messages, verbatim_budget)
                recent_tokens = _count_context_tokens(truncated_recent)
                recent_messages = truncated_recent

        # Combine back with other messages
        combined = other_messages + recent_messages
        return combined, recent_tokens

    def _create_default_agent(self) -> Agent:
        from .services.llm import create_compression_agent

        return create_compression_agent()


class SemanticCompressor:
    """Tiered context compressor with LLM semantic distillation."""

    def __init__(
        self,
        context: list[dict[str, Any]],
        max_context: int,
        pending_query: str | None = None,
        aggressive_mode: bool = False,
        preserve_pattern: str | None = None,
        agent: Agent | None = None,
    ):
        self.context = context
        self.max_context = max_context
        self.pending_query = pending_query
        self.aggressive_mode = aggressive_mode
        self.preserve_pattern = (
            __import__("re").compile(preserve_pattern, __import__("re").IGNORECASE)
            if preserve_pattern
            else None
        )
        self._agent = agent
        self.original_tokens = _count_context_tokens(context)

    def compress_sync(self) -> CompressionResult:
        """Synchronous compression: mechanical tiers only.

        Does not invoke LLM distillation. Use compress() for full semantic compression.
        """
        return self._run_tiers(use_llm=False)

    async def compress(self) -> CompressionResult:
        """Asynchronous compression: full pipeline with LLM distillation."""
        return await self._run_tiers_async()

    def _run_tiers(self, use_llm: bool) -> CompressionResult:
        """Run tiers 0, 1, and optionally 3 (truncation).

        Args:
            use_llm: If False (sync mode), skip LLM distillation (Tier 2).

        Returns:
            CompressionResult with mechanical or truncated output.
        """
        start_time = time.time()
        result = self._run_mechanical_tiers()
        result_tokens = _count_context_tokens(result)
        if result_tokens > self.max_context:
            result = _hard_truncate_to_budget(result, self.max_context)
        return self._build_result(result, start_time, strategy="mechanical")

    async def _run_tiers_async(self) -> CompressionResult:
        """Run tiers 0, 1, 2 (LLM), and 3 if needed.

        Returns:
            CompressionResult with semantic, mechanical, or truncated output.
        """
        start_time = time.time()

        if not settings.SEMANTIC_COMPRESSION_ENABLED:
            result = self._run_mechanical_tiers()
            result_tokens = _count_context_tokens(result)
            if result_tokens > self.max_context:
                result = _hard_truncate_to_budget(result, self.max_context)
            return self._build_result(result, start_time, strategy="mechanical")

        result = self._run_mechanical_tiers()
        result_tokens = _count_context_tokens(result)

        if result_tokens <= self.max_context:
            return self._build_result(result, start_time, strategy="mechanical")

        try:
            distiller = LLMContextDistiller(agent=self._agent)
            distilled, compressed_state = await asyncio.wait_for(
                distiller.distill(result, self.pending_query, self.max_context),
                timeout=settings.SEMANTIC_COMPRESSION_TIMEOUT_SECONDS,
            )
            distilled_tokens = _count_context_tokens(distilled)

            if distilled_tokens <= self.max_context:
                return self._build_result(
                    distilled, start_time, strategy="semantic", compressed_state=compressed_state
                )

            logger.warning(
                f"LLM distillation produced {distilled_tokens} tokens, exceeds budget "
                f"{self.max_context}, applying hard truncation"
            )
            result = _hard_truncate_to_budget(distilled, self.max_context)
        except TimeoutError:
            logger.warning("LLM distillation timed out, falling back to hard truncation")
            result = _hard_truncate_to_budget(result, self.max_context)
        except Exception as e:
            if settings.SEMANTIC_COMPRESSION_FALLBACK_ON_ERROR:
                logger.warning(f"LLM distillation failed: {e}, falling back to hard truncation")
                result = _hard_truncate_to_budget(result, self.max_context)
            else:
                raise

        return self._build_result(result, start_time, strategy="truncation")

    def _run_mechanical_tiers(self) -> list[dict[str, Any]]:
        """Run tier 1: mechanical pre-processing (lossless where possible)."""
        if self.original_tokens <= self.max_context:
            return self.context

        return _mechanical_preprocess(
            self.context,
            self.preserve_pattern,
            self.aggressive_mode,
        )

    def _build_result(
        self,
        compressed: list[dict[str, Any]],
        start_time: float,
        strategy: str,
        compressed_state: CompressedState | None = None,
    ) -> CompressionResult:
        final_tokens = _count_context_tokens(compressed)
        reduction_pct = (
            (self.original_tokens - final_tokens) / self.original_tokens
            if self.original_tokens > 0
            else 0.0
        )

        archive_id = ContextArchive.store(self.context)

        return CompressionResult(
            original_context=self.context,
            compressed_context=compressed,
            original_tokens=self.original_tokens,
            compressed_tokens=final_tokens,
            reduction_pct=reduction_pct,
            retention_score=1.0,
            strategy_used=strategy,
            execution_time=time.time() - start_time,
            archive_id=archive_id,
            was_rolled_back=False,
            compression_rationale=compressed_state.compression_rationale if compressed_state else None,
            task_state=compressed_state.task_state if compressed_state else None,
        )


def _count_context_tokens(context: list[dict[str, Any]]) -> int:
    """Count tokens in message content, excluding JSON structural overhead."""
    total = 0
    for msg in context:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        else:
            total += count_tokens(json.dumps(content))
    total += len(context) * 4
    return total


def _preserve_matching_content(
    context: list[dict[str, Any]],
    preserve_pattern: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preserved = []
    compressible = []

    for msg in context:
        if msg.get("role") in settings.SEMANTIC_COMPRESSION_PRESERVE_VERBATIM_ROLES:
            preserved.append(msg)
        elif preserve_pattern and preserve_pattern.search(json.dumps(msg)):
            preserved.append(msg)
        else:
            compressible.append(msg)

    logger.debug(f"Preserved {len(preserved)} messages (verbatim roles {settings.SEMANTIC_COMPRESSION_PRESERVE_VERBATIM_ROLES} + pattern matches)")
    return preserved, compressible


def _mechanical_preprocess(
    context: list[dict[str, Any]],
    preserve_pattern: Any | None = None,
    aggressive_mode: bool = False,
) -> list[dict[str, Any]]:
    """Lossless or low-loss mechanical reductions."""
    preserved, compressible = _preserve_matching_content(context, preserve_pattern)

    seen_hashes = set()
    deduped = []
    for msg in compressible:
        h = hash(json.dumps(msg, sort_keys=True))
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(msg)

    merged = _merge_consecutive_same_role(deduped, aggressive_mode)
    cleaned = [m for m in merged if str(m.get("content", "")).strip()]

    return preserved + cleaned


def _merge_consecutive_same_role(
    messages: list[dict[str, Any]], aggressive_mode: bool = False
) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role."""
    import re

    merged = []
    last_role = None
    last_content = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)

        if aggressive_mode:
            content = re.sub(r"#.*?$", "", content, flags=re.MULTILINE)
            content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s+", " ", content).strip()

        if msg["role"] == last_role:
            last_content.append(content)
        else:
            if last_role is not None:
                merged.append({"role": last_role, "content": "\n---\n".join(last_content)})
            last_role = msg["role"]
            last_content = [content]

    if last_role is not None:
        merged.append({"role": last_role, "content": "\n---\n".join(last_content)})

    return merged


def _hard_truncate_to_budget(
    context: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Truncate context to fit within token budget."""
    system_msgs = [m for m in context if m.get("role") == "system"]
    non_system = [m for m in context if m.get("role") != "system"]

    result = list(system_msgs)
    result_tokens = _count_context_tokens(result)

    for msg in reversed(non_system):
        msg_tokens = _count_context_tokens([msg])
        if result_tokens + msg_tokens <= budget:
            result.insert(len(system_msgs), msg)
            result_tokens += msg_tokens
        # Continue evaluating all messages, don't break on first that doesn't fit

    max_iterations = len(result) * 2
    iteration = 0
    while result_tokens > budget and result and iteration < max_iterations:
        iteration += 1
        truncated_something = False
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            other_messages = result[:i] + result[i + 1 :]
            other_tokens = _count_context_tokens(other_messages)
            leftover = budget - other_tokens

            if leftover > 10:
                truncated_msg = _truncate_message_to_budget(msg, leftover)
                if truncated_msg is not None:
                    result[i] = truncated_msg
                    result_tokens = _count_context_tokens(result)
                    truncated_something = True
                    break
                else:
                    result = result[:i] + result[i + 1 :]
                    result_tokens = _count_context_tokens(result)
                    truncated_something = True
                    break
            else:
                result = result[:i] + result[i + 1 :]
                result_tokens = _count_context_tokens(result)
                truncated_something = True
                break

        if not truncated_something:
            break

    return result


def _truncate_message_to_budget(
    msg: dict[str, Any], token_budget: int
) -> dict[str, Any] | None:
    """Truncate a single message's content to fit within token budget."""
    content = msg.get("content", "")
    if not content:
        return msg

    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except (TypeError, ValueError):
            return None

    marker = "... [truncated]"
    low, high = 0, len(content)
    best_fit = 0

    while low <= high:
        mid = (low + high) // 2
        truncated = content[:mid] + marker
        test_msg = {**msg, "content": truncated}
        test_tokens = _count_context_tokens([test_msg])
        if test_tokens <= token_budget:
            best_fit = mid
            low = mid + 1
        else:
            high = mid - 1

    if best_fit == 0:
        return None

    final_content = content[:best_fit] + marker
    return {**msg, "content": final_content}
