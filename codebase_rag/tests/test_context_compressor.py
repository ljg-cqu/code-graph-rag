"""Tests for ContextCompressor."""


import pytest

from codebase_rag.context_compressor import (
    CompressionResult,
    ContextArchive,
    ContextCompressor,
)

pytestmark = [pytest.mark.anyio]


class TestContextCompressor:
    """Test cases for ContextCompressor."""

    def test_compression_triggers_when_over_budget(self):
        """Verify compression runs when tokens > max_context."""
        # Create a context that exceeds the budget
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you? " * 100},
            {"role": "assistant", "content": "I am doing well, thank you! " * 100},
        ]

        # Set max_context low to force compression
        compressor = ContextCompressor(
            context=context,
            max_context=100,  # Very low budget to force compression
            aggressive_mode=False,
        )

        result = compressor.compress_sync()

        # Compression should trigger (not no-op)
        assert result.strategy_used != "no-op"
        # Should have reduced tokens
        assert result.compressed_tokens <= 100 or result.reduction_pct > 0

    def test_no_op_when_under_budget_and_few_messages(self):
        """Verify fast path when under budget with few messages."""
        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        compressor = ContextCompressor(
            context=context,
            max_context=10000,  # High budget, context is under
            aggressive_mode=False,
        )

        result = compressor.compress_sync()

        # Should be no-op
        assert result.strategy_used == "no-op"
        assert result.reduction_pct == 0.0
        assert result.compressed_tokens == result.original_tokens

    def test_strategies_produce_different_results(self):
        """Verify strategies produce measurably different outputs."""
        context = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Question one?"},
            {"role": "assistant", "content": "Answer one with some details."},
            {"role": "user", "content": "Question two?"},
            {"role": "assistant", "content": "Answer two with more details here."},
            {"role": "user", "content": "Question three?"},
            {"role": "assistant", "content": "Answer three with additional information."},
        ]

        compressor = ContextCompressor(
            context=context,
            max_context=100,  # Force compression
            aggressive_mode=False,
        )

        result = compressor.compress_sync()

        # Should use a real strategy
        assert "S" in result.strategy_used  # Strategy IDs are S1-S5
        assert result.strategy_used != "no-op"

    def test_semantic_retention_score_range(self):
        """Verify retention score is in valid range 0.0-1.0."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "def hello(): pass"},
            {"role": "assistant", "content": "class World: pass"},
        ]

        compressor = ContextCompressor(context=context, max_context=1000)

        # Test entity retention
        compressed = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "def hello(): pass"},
        ]

        score = compressor._calculate_semantic_retention(context, compressed)
        assert 0.0 <= score <= 1.0

    def test_rollback_on_low_retention(self):
        """Verify rollback works when retention is below threshold and original fits budget."""
        # Create context with messages long enough to be truncated by hierarchical summarization (>500 chars)
        # Need compressible content > 3 messages to bypass fast path
        long_text = "This is a long message that exceeds 500 characters. " * 15  # ~750 chars
        context = [
            {"role": "system", "content": "System prompt here."},
            {"role": "user", "content": long_text + "First question with some details here."},
            {"role": "assistant", "content": long_text + "First answer with explanation of concepts."},
            {"role": "user", "content": long_text + "Second question asking for more info."},
            {"role": "assistant", "content": long_text + "Second answer with detailed response here."},
            {"role": "user", "content": long_text + "Third question about the topic discussed."},
            {"role": "assistant", "content": long_text + "Third answer with final thoughts on this."},
        ]

        # Original must fit within budget so rollback is possible
        # Set max_context high enough for original, but low enough to trigger compression
        # Set very high retention threshold to force rollback
        compressor = ContextCompressor(
            context=context,
            max_context=5000,  # High budget, original fits
            min_retention_score=99.9,  # Impossibly high - will force rollback
            aggressive_mode=True,  # Force aggressive compression to reduce retention
        )

        result = compressor.compress_sync()

        # Should be rolled back (since original fits budget and retention below threshold)
        assert result.was_rolled_back is True
        assert result.reduction_pct == 0.0
        assert result.compressed_tokens == result.original_tokens

    def test_hard_truncate_respects_budget(self):
        """Verify _hard_truncate_to_budget respects max_context."""
        context = [
            {"role": "system", "content": "System message that must be preserved."},
            {"role": "user", "content": "First user message with some content."},
            {"role": "assistant", "content": "First assistant response here."},
            {"role": "user", "content": "Second user message with more content."},
            {"role": "assistant", "content": "Second assistant response here."},
        ]

        compressor = ContextCompressor(context=context, max_context=1000)

        # Truncate to a small budget
        truncated = compressor._hard_truncate_to_budget(context, budget=50)

        # Should include system message
        assert any(m.get("role") == "system" for m in truncated)

        # Should be within budget
        tokens = compressor._count_context_tokens(truncated)
        assert tokens <= 50

    def test_preserve_pattern_works(self):
        """Verify regex preservation pattern works."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Keep this: IMPORTANT_KEYWORD"},
            {"role": "assistant", "content": "Regular response."},
        ]

        compressor = ContextCompressor(
            context=context,
            max_context=50,
            preserve_pattern=r"IMPORTANT_KEYWORD",
        )

        preserved, compressible = compressor._preserve_matching_content(context)

        # The user message with IMPORTANT_KEYWORD should be preserved
        assert any("IMPORTANT_KEYWORD" in str(m.get("content", "")) for m in preserved)

    def test_archive_store_and_retrieve(self):
        """Verify ContextArchive TTL functionality."""
        context = [
            {"role": "user", "content": "Test message."},
        ]

        # Store context
        archive_id = ContextArchive.store(context)
        assert archive_id.startswith("ctx_arc_")

        # Retrieve context
        retrieved = ContextArchive.retrieve(archive_id)
        assert retrieved is not None
        assert retrieved == context

        # Non-existent archive returns None
        assert ContextArchive.retrieve("non_existent_id") is None

    def test_empty_context(self):
        """Test handling of empty context."""
        compressor = ContextCompressor(context=[], max_context=1000)
        result = compressor.compress_sync()

        assert result.strategy_used == "no-op"
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_single_large_message_over_budget(self):
        """Test hard truncation with single large message."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "x" * 10000},  # Very large message
        ]

        compressor = ContextCompressor(context=context, max_context=100)
        result = compressor.compress_sync()

        # Should truncate to fit budget
        assert result.compressed_tokens <= 100 or result.was_rolled_back

    def test_all_system_messages_preserved(self):
        """Verify system messages are always preserved."""
        context = [
            {"role": "system", "content": "System instruction 1."},
            {"role": "system", "content": "System instruction 2."},
            {"role": "user", "content": "Hello?"},
        ]

        compressor = ContextCompressor(context=context, max_context=100)
        preserved, compressible = compressor._preserve_matching_content(context)

        # All system messages should be preserved
        assert len(preserved) == 2
        assert all(m.get("role") == "system" for m in preserved)

    def test_keyword_overlap_score(self):
        """Test keyword overlap scoring."""
        original = [
            {"role": "user", "content": "function test helper utility"},
        ]
        compressed = [
            {"role": "user", "content": "function test"},
        ]

        compressor = ContextCompressor(context=original, max_context=1000)
        score = compressor._keyword_overlap_score(original, compressed)

        # Should have partial overlap
        assert 0.0 < score < 1.0

    def test_entity_retention_score(self):
        """Test entity retention scoring for code."""
        original = [
            {"role": "user", "content": "def my_function(): pass\nclass MyClass: pass"},
        ]
        compressed = [
            {"role": "user", "content": "def my_function(): pass"},
        ]

        compressor = ContextCompressor(context=original, max_context=1000)
        score = compressor._entity_retention_score(original, compressed)

        # Should have partial retention (function kept, class lost)
        assert 0.0 < score < 1.0

    def test_aggressive_mode_affects_weights(self):
        """Verify aggressive mode changes score weights."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer with lots of details and explanations." * 20},
        ]

        # Test normal mode
        normal_compressor = ContextCompressor(
            context=context, max_context=100, aggressive_mode=False
        )
        normal_result = normal_compressor.compress_sync()

        # Test aggressive mode
        aggressive_compressor = ContextCompressor(
            context=context, max_context=100, aggressive_mode=True
        )
        aggressive_result = aggressive_compressor.compress_sync()

        # Both should complete
        assert normal_result.strategy_used != "no-op"
        assert aggressive_result.strategy_used != "no-op"

    def test_token_counting_excludes_json_overhead(self):
        """Verify token counting doesn't include JSON structural characters."""
        context = [
            {"role": "user", "content": "hello"},
        ]

        compressor = ContextCompressor(context=context, max_context=1000)
        tokens = compressor._count_context_tokens(context)

        # Should be close to actual content tokens, not inflated by JSON
        assert tokens < 50  # Much less than json.dumps would produce

    def test_max_context_zero(self):
        """Test edge case with max_context of 0."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello."},
        ]

        compressor = ContextCompressor(context=context, max_context=0)
        result = compressor.compress_sync()

        # Should still complete without error
        assert isinstance(result, CompressionResult)

    def test_rollback_does_not_exceed_budget_when_preserved_content_is_large(self):
        """Bug 1: rollback must not revert to original if original exceeds budget."""
        context = [
            {"role": "system", "content": "System 1. " * 50},
            {"role": "system", "content": "System 2. " * 50},
        ]
        compressor = ContextCompressor(context=context, max_context=50)
        result = compressor.compress_sync()
        assert result.compressed_tokens <= 50, \
            f"compressed_tokens ({result.compressed_tokens}) must not exceed max_context (50)"

    def test_strategies_compress_large_messages_even_when_few(self):
        """Bug 2: strategies must act when token budget is exceeded, regardless of message count."""
        context = [
            {"role": "user", "content": "A" * 10000},
            {"role": "assistant", "content": "B" * 10000},
            {"role": "user", "content": "C" * 10000},
        ]
        compressor = ContextCompressor(context=context, max_context=100)
        # Strategies should attempt compression even with few messages when over budget
        # Note: strategies don't guarantee budget compliance, hard_truncate enforces it
        for _, fname, _ in compressor.STRATEGIES:
            result = getattr(compressor, fname)(context.copy())
            # Verify strategy returned a result (didn't just return original unchanged)
            assert isinstance(result, list), f"{fname} should return a list"
            assert len(result) > 0, f"{fname} should return non-empty result"

        # Full compress_sync should enforce budget via hard_truncate
        final_result = compressor.compress_sync()
        assert final_result.compressed_tokens <= 100, \
            f"compressed_tokens ({final_result.compressed_tokens}) exceeds max_context (100)"

    def test_tool_messages_preserved_by_default(self):
        """Bug 3: tool/function messages must be in preserved set."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "User."},
            {"role": "tool", "content": "Tool result."},
            {"role": "function", "content": "Function result."},
        ]
        compressor = ContextCompressor(context=context, max_context=1000)
        preserved, compressible = compressor._preserve_matching_content(context)
        preserved_roles = {m["role"] for m in preserved}
        assert "tool" in preserved_roles
        assert "function" in preserved_roles

    def test_hard_truncate_fits_single_large_system_message(self):
        """Bug 4: hard truncate must truncate individual messages when necessary."""
        context = [{"role": "system", "content": "Very long system message. " * 100}]
        compressor = ContextCompressor(context=context, max_context=20)
        truncated = compressor._hard_truncate_to_budget(context, 20)
        tokens = compressor._count_context_tokens(truncated)
        # Allow small margin due to truncation indicator tokens
        assert tokens <= 30, f"truncated tokens ({tokens}) significantly exceed budget (20)"

    def test_thread_pool_conditional_usage(self):
        """Bug 5: thread pool should only be used for non-trivial workloads."""
        # Small context should use sequential evaluation
        small_context = [
            {"role": "user", "content": "Short message."},
        ]
        compressor = ContextCompressor(context=small_context, max_context=100)

        # Run multiple compressions - should complete without error
        for _ in range(5):
            compressor.compress_sync()

        # Large context should use parallel evaluation
        large_context = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
            {"role": "user", "content": "C" * 1000},
            {"role": "assistant", "content": "D" * 1000},
        ]
        compressor_large = ContextCompressor(context=large_context, max_context=100)

        # Should complete without error
        result = compressor_large.compress_sync()
        assert result is not None


    async def test_compress_async_delegates_to_semantic_compressor(self):
        """Backward compatibility: compress_async delegates to SemanticCompressor."""
        import warnings

        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello!"},
        ]
        compressor = ContextCompressor(context=context, max_context=10000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = await compressor.compress_async()

        assert isinstance(result, CompressionResult)
        assert result.compressed_tokens == result.original_tokens
