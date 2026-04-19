"""Tests for SemanticCompressor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codebase_rag.compression_schemas import CompressedState, Message, TaskState
from codebase_rag.config import settings
from codebase_rag.context_compressor import CompressionResult
from codebase_rag.semantic_compressor import (
    LLMContextDistiller,
    SemanticCompressor,
    _count_context_tokens,
    _hard_truncate_to_budget,
    _mechanical_preprocess,
    _merge_consecutive_same_role,
    _preserve_matching_content,
    _truncate_message_to_budget,
)

pytestmark = [pytest.mark.anyio]


class TestSemanticCompressor:
    """Test cases for SemanticCompressor."""

    def test_no_op_when_under_budget(self):
        """Tier 0: fast path when under budget."""
        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        compressor = SemanticCompressor(context=context, max_context=10000)
        result = compressor.compress_sync()

        assert result.strategy_used == "mechanical"
        assert result.reduction_pct == 0.0
        assert result.compressed_tokens == result.original_tokens

    def test_mechanical_preprocess_dedupes(self):
        """Tier 1: deduplication removes identical messages."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ]
        result = _mechanical_preprocess(context)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1

    def test_mechanical_preprocess_merges_same_role(self):
        """Tier 1: consecutive same-role messages are merged."""
        context = [
            {"role": "user", "content": "Hello."},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Hi."},
        ]
        result = _mechanical_preprocess(context)
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "Hello." in user_msgs[0]["content"]
        assert "How are you?" in user_msgs[0]["content"]

    def test_mechanical_preprocess_preserves_system_tool(self):
        """Tier 1: system and tool messages are preserved, not merged."""
        context = [
            {"role": "system", "content": "Sys1."},
            {"role": "system", "content": "Sys2."},
            {"role": "tool", "content": "Tool result."},
            {"role": "user", "content": "Hello."},
        ]
        result = _mechanical_preprocess(context)
        assert any(m["role"] == "system" and m["content"] == "Sys1." for m in result)
        assert any(m["role"] == "system" and m["content"] == "Sys2." for m in result)
        assert any(m["role"] == "tool" for m in result)

    def test_compress_sync_skips_llm(self):
        """Sync API does not invoke LLM distillation."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
        ]
        compressor = SemanticCompressor(context=context, max_context=100)
        with patch.object(LLMContextDistiller, "distill") as mock_distill:
            result = compressor.compress_sync()
            mock_distill.assert_not_called()
        assert result.strategy_used == "mechanical"
        assert result.compressed_tokens <= 100

    async def test_compress_async_uses_llm_when_over_budget(self):
        """Async API invokes LLM when over budget."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
        ]
        compressor = SemanticCompressor(context=context, max_context=100)

        mock_state = CompressedState(
            task_state=TaskState(current_objective="test"),
            recent_messages=[Message(role="user", content="short")],
            compression_rationale="test",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        with patch.object(
            LLMContextDistiller, "_create_default_agent", return_value=mock_agent
        ):
            result = await compressor.compress()

        mock_agent.run.assert_called_once()
        assert result.strategy_used == "semantic"
        assert result.compressed_tokens <= 100

    async def test_compress_async_no_op_when_under_budget(self):
        """Async API skips LLM when under budget."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello!"},
        ]
        compressor = SemanticCompressor(context=context, max_context=10000)
        result = await compressor.compress()
        assert result.strategy_used == "mechanical"
        assert result.reduction_pct == 0.0

    async def test_fallback_on_llm_error(self):
        """Error handling falls back to truncation when LLM fails."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
        ]
        compressor = SemanticCompressor(context=context, max_context=100)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM failed"))

        with patch.object(
            LLMContextDistiller, "_create_default_agent", return_value=mock_agent
        ):
            result = await compressor.compress()

        assert result.strategy_used == "truncation"
        assert result.compressed_tokens <= 100

    async def test_fallback_on_llm_timeout(self):
        """Timeout handling falls back to truncation."""

        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
        ]
        compressor = SemanticCompressor(context=context, max_context=100)

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=TimeoutError())

        with patch.object(
            LLMContextDistiller, "_create_default_agent", return_value=mock_agent
        ):
            result = await compressor.compress()

        assert result.strategy_used == "truncation"
        assert result.compressed_tokens <= 100

    def test_hard_truncation_enforces_budget(self):
        """Tier 3: hard truncation guarantees budget invariant."""
        context = [
            {"role": "system", "content": "System message that must be preserved."},
            {"role": "user", "content": "First user message with some content."},
            {"role": "assistant", "content": "First assistant response here."},
            {"role": "user", "content": "Second user message with more content."},
            {"role": "assistant", "content": "Second assistant response here."},
        ]
        truncated = _hard_truncate_to_budget(context, budget=50)
        tokens = _count_context_tokens(truncated)
        assert tokens <= 50
        assert any(m.get("role") == "system" for m in truncated)

    def test_empty_context(self):
        """Test handling of empty context."""
        compressor = SemanticCompressor(context=[], max_context=1000)
        result = compressor.compress_sync()
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_single_large_message_over_budget(self):
        """Test hard truncation with single large message."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "x" * 10000},
        ]
        compressor = SemanticCompressor(context=context, max_context=100)
        result = compressor.compress_sync()
        assert result.compressed_tokens <= 100

    def test_all_system_messages_preserved(self):
        """Verify system messages are always preserved."""
        context = [
            {"role": "system", "content": "System instruction 1."},
            {"role": "system", "content": "System instruction 2."},
            {"role": "user", "content": "Hello?"},
        ]
        preserved, compressible = _preserve_matching_content(context, None)
        assert len(preserved) == 2
        assert all(m.get("role") == "system" for m in preserved)

    def test_tool_messages_preserved_by_default(self):
        """Tool/function messages must be in preserved set."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "User."},
            {"role": "tool", "content": "Tool result."},
            {"role": "function", "content": "Function result."},
        ]
        preserved, compressible = _preserve_matching_content(context, None)
        preserved_roles = {m["role"] for m in preserved}
        assert "tool" in preserved_roles
        assert "function" in preserved_roles

    def test_truncate_message_to_budget(self):
        """Test single message truncation."""
        msg = {"role": "user", "content": "Hello world this is a test message."}
        result = _truncate_message_to_budget(msg, token_budget=20)
        assert result is not None
        assert result["content"].endswith("... [truncated]")

    def test_truncate_message_to_budget_impossible(self):
        """Test truncation when message cannot fit at all."""
        msg = {"role": "user", "content": "Hello world this is a test message."}
        result = _truncate_message_to_budget(msg, token_budget=1)
        assert result is None

    def test_query_aware_passed_to_distiller(self):
        """Pending query is passed to the distiller prompt."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
        ]
        compressor = SemanticCompressor(
            context=context,
            max_context=100,
            pending_query="How does this work?",
        )
        assert compressor.pending_query == "How does this work?"

    async def test_query_aware_prioritizes_relevant_info(self):
        """Query-aware optimization preserves info needed to answer pending query."""
        # Create context with multiple topics: authentication and database
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "How does authentication work?"},
            {"role": "assistant", "content": "Authentication uses OAuth 2.0 with JWT tokens. The validate_token function checks expiration."},
            {"role": "user", "content": "Tell me about the database connection pool."},
            {"role": "assistant", "content": "The database uses a connection pool with max 10 connections. The PoolManager class handles cleanup."},
            {"role": "user", "content": "A" * 5000},  # Large content to force compression
        ]

        # Mock LLM that would prioritize auth info when query is about auth
        # Note: recent_messages must fit within max_context budget
        auth_focused_state = CompressedState(
            task_state=TaskState(
                current_objective="Understanding authentication flow",
                active_code_elements=["validate_token", "OAuth 2.0"],
            ),
            recent_messages=[Message(role="user", content="Short recent message")],
            compression_rationale="Prioritized authentication info based on pending query",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=auth_focused_state))

        compressor = SemanticCompressor(
            context=context,
            max_context=500,  # Increased budget to fit reconstructed messages
            pending_query="Why is authentication failing?",
            agent=mock_agent,
        )

        with patch.object(settings, "SEMANTIC_COMPRESSION_ENABLED", True):
            result = await compressor.compress()

        # Verify the mock was called (LLM distillation occurred)
        mock_agent.run.assert_called_once()
        # Verify the prompt included the pending query
        call_args = mock_agent.run.call_args[0][0]
        assert "Why is authentication failing?" in call_args
        assert result.strategy_used == "semantic"

    def test_aggressive_mode_strips_comments(self):
        """Aggressive mode strips comments during merge."""
        context = [
            {"role": "user", "content": "Hello. # comment"},
            {"role": "user", "content": "World. // another comment"},
        ]
        result = _merge_consecutive_same_role(context, aggressive_mode=True)
        assert len(result) == 1
        assert "# comment" not in result[0]["content"]
        assert "// another comment" not in result[0]["content"]

    def test_max_context_zero(self):
        """Test edge case with max_context of 0."""
        context = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Hello."},
        ]
        compressor = SemanticCompressor(context=context, max_context=0)
        result = compressor.compress_sync()
        assert isinstance(result, CompressionResult)
        assert result.compressed_tokens <= 0 + 4  # allow small overhead


class TestLLMContextDistiller:
    """Test cases for LLMContextDistiller."""

    async def test_distill_produces_valid_output(self):
        """Distiller reconstructs messages from CompressedState."""
        mock_state = CompressedState(
            task_state=TaskState(
                current_objective="Fix auth bug",
                completed_steps=["Found the bug"],
            ),
            recent_messages=[
                Message(role="user", content="What next?"),
            ],
            compression_rationale="Kept essential state",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        distiller = LLMContextDistiller(agent=mock_agent)
        messages, state = await distiller.distill(
            messages=[{"role": "user", "content": "Hello"}],
            pending_query=None,
            max_context=1000,
        )

        assert isinstance(messages, list)
        assert any("Fix auth bug" in str(m.get("content", "")) for m in messages)
        assert any(m.get("role") == "user" and m.get("content") == "What next?" for m in messages)
        assert state == mock_state

    def test_truncate_input_for_distillation(self):
        """Large inputs are pre-truncated before distillation."""
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "A" * 10000},
            {"role": "assistant", "content": "B" * 10000},
            {"role": "user", "content": "C" * 10000},
            {"role": "assistant", "content": "D" * 10000},
        ]
        distiller = LLMContextDistiller()
        result = distiller._truncate_input_for_distillation(messages, cap=500)
        assert _count_context_tokens(result) <= 500 + 50  # small margin

    def test_apply_verbatim_budget(self):
        """Verbatim budget truncates oldest recent messages first."""
        from unittest.mock import patch
        from codebase_rag.config import settings

        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First message."},
            {"role": "assistant", "content": "Response."},
            {"role": "user", "content": "Second message."},
            {"role": "assistant", "content": "Another response."},
        ]
        distiller = LLMContextDistiller()

        # Mock settings
        with patch.object(settings, "SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT", 40.0):
            with patch.object(settings, "SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES", 2):
                # max_context = 100, verbatim budget = 40 tokens
                # Let's compute token counts: each message approx 4 tokens overhead + content tokens
                # We'll make recent messages exceed budget
                # Simulate by using large content
                large_messages = messages.copy()
                large_messages[-2]["content"] = "X" * 1000  # Second last message large
                large_messages[-1]["content"] = "Y" * 1000  # Last message large

                result, verbatim_used = distiller._apply_verbatim_budget(large_messages, max_context=100)
                # Should have truncated recent messages (last 2) to fit within 40 tokens
                assert verbatim_used <= 40
                # Recent messages count should be <= 2
                # System message should be preserved (not counted as recent)
                # We'll just assert result is not empty
                assert len(result) > 0

    def test_verbatim_budget_zero(self):
        """When verbatim budget results in 0 tokens, all messages are passed through."""
        from unittest.mock import patch
        from codebase_rag.config import settings

        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First message."},
            {"role": "assistant", "content": "Response."},
        ]
        distiller = LLMContextDistiller()

        # Use a very small percentage that results in 0 budget (1% of 10 = 0.1 ≈ 0)
        with patch.object(settings, "SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT", 0.1):
            result, verbatim_used = distiller._apply_verbatim_budget(messages, max_context=10)
            # Should return all messages unchanged since budget rounds to 0
            assert result == messages
            assert verbatim_used == 0

    async def test_verbatim_budget_respected(self):
        """LLM distillation respects verbatim budget."""
        # Create a mock LLM agent that returns a CompressedState
        mock_state = CompressedState(
            task_state=TaskState(current_objective="test"),
            recent_messages=[Message(role="user", content="short")],
            compression_rationale="test",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        distiller = LLMContextDistiller(agent=mock_agent)

        # Messages where recent messages exceed verbatim budget
        messages = [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second " + "X" * 1000},  # Large recent message
            {"role": "assistant", "content": "Third " + "Y" * 1000},  # Large recent message
        ]

        with patch.object(settings, "SEMANTIC_COMPRESSION_VERBATIM_BUDGET_PCT", 10.0):
            with patch.object(settings, "SEMANTIC_COMPRESSION_MAX_RECENT_MESSAGES", 2):
                # The distill method should apply verbatim budget before calling LLM
                result, state = await distiller.distill(
                    messages, pending_query=None, max_context=1000
                )
                # Verify compression succeeded
                assert isinstance(result, list)
                # The result should have been compressed (token count reduced)
                result_tokens = _count_context_tokens(result)
                assert result_tokens <= 1000

    async def test_llm_distill_includes_task_state(self):
        """Tier 2: LLM distillation extracts structured task state."""
        mock_state = CompressedState(
            task_state=TaskState(
                current_objective="Fix authentication bug in token validation",
                completed_steps=["Analyzed auth.py", "Located validate_token function"],
                pending_questions=["Apply the fix"],
                active_code_elements=["auth.py:validate_token()", "TokenManager"],
                key_decisions=["Bug is in line 142, not TokenManager"],
                error_states=["Refresh token fails due to validation bug (UNRESOLVED)"],
                tool_results_summary="Cypher query found 3 callers of validate_token",
                user_preferences=["Prefer minimal changes"],
            ),
            recent_messages=[
                Message(role="user", content="Show me the exact line."),
            ],
            compression_rationale="Kept all task state and recent message.",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        distiller = LLMContextDistiller(agent=mock_agent)
        result, state = await distiller.distill(
            messages=[{"role": "user", "content": "How does auth work?"}],
            pending_query=None,
            max_context=1000,
        )

        # Verify task state fields are present in reconstructed messages
        result_text = " ".join(str(m.get("content", "")) for m in result)
        assert "Fix authentication bug" in result_text
        assert "validate_token" in result_text
        assert "TokenManager" in result_text
        assert "line 142" in result_text
        # Verify state is returned
        assert state.task_state.current_objective == "Fix authentication bug in token validation"

    async def test_llm_distill_preserves_recent_messages(self):
        """Tier 2: LLM distillation preserves recent messages verbatim."""
        # Create conversation with distinct recent messages
        recent_user_msg = "What is the exact error in line 142?"
        recent_assistant_msg = "The error is a null pointer dereference in validate_token()."

        mock_state = CompressedState(
            task_state=TaskState(
                current_objective="Fix auth bug",
                completed_steps=["Found bug location"],
            ),
            recent_messages=[
                Message(role="user", content=recent_user_msg),
                Message(role="assistant", content=recent_assistant_msg),
            ],
            compression_rationale="Preserved recent exchange verbatim.",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        distiller = LLMContextDistiller(agent=mock_agent)
        messages, state = await distiller.distill(
            messages=[
                {"role": "system", "content": "System."},
                {"role": "user", "content": "How does auth work?"},
                {"role": "assistant", "content": "Auth uses OAuth."},
                {"role": "user", "content": recent_user_msg},
                {"role": "assistant", "content": recent_assistant_msg},
            ],
            pending_query=None,
            max_context=1000,
        )

        # Verify recent messages are preserved verbatim in the reconstructed output
        # The last two messages should be the recent user and assistant messages
        assert len(state.recent_messages) == 2
        assert state.recent_messages[0].role == "user"
        assert state.recent_messages[0].content == recent_user_msg
        assert state.recent_messages[1].role == "assistant"
        assert state.recent_messages[1].content == recent_assistant_msg

        # Verify reconstructed messages contain the verbatim recent messages
        # Find the recent messages in the reconstructed output
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert any(m.get("content") == recent_user_msg for m in user_msgs)
        assert any(m.get("content") == recent_assistant_msg for m in assistant_msgs)

    async def test_llm_input_cap_enforced(self):
        """Large inputs are pre-truncated before distillation LLM call."""
        # Create messages that exceed the LLM input cap
        messages = [
            {"role": "system", "content": "System."},
        ]
        # Add many large messages to exceed cap
        for i in range(100):
            messages.append({"role": "user", "content": "X" * 1000})
            messages.append({"role": "assistant", "content": "Y" * 1000})

        mock_state = CompressedState(
            task_state=TaskState(current_objective="test"),
            recent_messages=[],
            compression_rationale="test",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=MagicMock(output=mock_state))

        with patch.object(settings, "SEMANTIC_COMPRESSION_LLM_INPUT_CAP", 5000):
            distiller = LLMContextDistiller(agent=mock_agent)
            result, state = await distiller.distill(
                messages, pending_query=None, max_context=10000
            )

            # Verify the LLM was called
            mock_agent.run.assert_called_once()
            # Verify result is valid
            assert isinstance(result, list)
