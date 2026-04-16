from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codebase_rag import constants as cs
from codebase_rag import exceptions as ex
from codebase_rag.services.llm import (
    CypherGenerator,
    _clean_cypher_response,
    create_rag_orchestrator,
)

pytestmark = [pytest.mark.anyio]


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


class TestCleanCypherResponse:
    def test_removes_leading_whitespace(self) -> None:
        result = _clean_cypher_response("   MATCH (n) RETURN n")
        assert result.startswith("MATCH")

    def test_removes_trailing_whitespace(self) -> None:
        result = _clean_cypher_response("MATCH (n) RETURN n   ")
        assert result.endswith(";")

    def test_removes_backticks(self) -> None:
        result = _clean_cypher_response("```MATCH (n) RETURN n```")
        assert "```" not in result

    def test_removes_cypher_prefix(self) -> None:
        result = _clean_cypher_response("cypher MATCH (n) RETURN n")
        assert not result.lower().startswith("cypher ")

    def test_adds_semicolon_if_missing(self) -> None:
        result = _clean_cypher_response("MATCH (n) RETURN n")
        assert result.endswith(";")

    def test_keeps_existing_semicolon(self) -> None:
        result = _clean_cypher_response("MATCH (n) RETURN n;")
        assert result == "MATCH (n) RETURN n;"
        assert not result.endswith(";;")

    def test_handles_complex_query(self) -> None:
        query = """```cypher
MATCH (n:Function)-[:CALLS]->(m:Function)
WHERE n.name = 'main'
RETURN m.name
```"""
        result = _clean_cypher_response(query)
        assert result.startswith("MATCH")
        assert result.endswith(";")
        assert "```" not in result

    def test_handles_multiline_query(self) -> None:
        query = """MATCH (n)
WHERE n.type = 'class'
RETURN n.name"""
        result = _clean_cypher_response(query)
        assert result.endswith(";")
        assert "MATCH" in result


class TestCypherGenerator:
    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    def test_init_creates_agent(
        self,
        mock_agent: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.GOOGLE
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        generator = CypherGenerator()

        mock_agent.assert_called_once()
        assert generator.agent is not None

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    def test_init_raises_on_error(
        self, mock_get_provider: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.active_cypher_config = MagicMock()
        mock_get_provider.side_effect = Exception("Provider error")

        with pytest.raises(ex.LLMGenerationError):
            CypherGenerator()

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    def test_uses_local_prompt_for_ollama(
        self,
        mock_agent: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.OLLAMA
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        CypherGenerator()

        call_kwargs = mock_agent.call_args.kwargs
        assert "system_prompt" in call_kwargs


class TestCypherGeneratorGenerate:
    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    async def test_generate_returns_cleaned_query(
        self,
        mock_agent_cls: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.GOOGLE
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_result = MagicMock()
        mock_result.output = "MATCH (n) RETURN n"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent

        generator = CypherGenerator()
        result = await generator.generate("Find all nodes")

        assert result == "MATCH (n) RETURN n;"

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    async def test_generate_raises_on_invalid_output(
        self,
        mock_agent_cls: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.GOOGLE
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_result = MagicMock()
        mock_result.output = "Invalid response with no query keyword"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent

        generator = CypherGenerator()
        with pytest.raises(ex.LLMGenerationError):
            await generator.generate("Find all nodes")

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    async def test_generate_raises_on_agent_error(
        self,
        mock_agent_cls: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.GOOGLE
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("API error"))
        mock_agent_cls.return_value = mock_agent

        generator = CypherGenerator()
        with pytest.raises(ex.LLMGenerationError):
            await generator.generate("Find all nodes")

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    async def test_repair_returns_cleaned_query(
        self,
        mock_agent_cls: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_config.provider = cs.Provider.GOOGLE
        mock_settings.active_cypher_config = mock_config
        mock_settings.AGENT_RETRIES = 3

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_result = MagicMock()
        mock_result.output = "MATCH (f:File) RETURN f.path AS path"
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent

        generator = CypherGenerator()
        result = await generator.repair(
            "Find orphan files",
            "MATCH (f:File) WHERE NOT (f) RETURN f.path AS path",
            "Invalid type vertex for 'NOT'.",
        )

        assert result == "MATCH (f:File) RETURN f.path AS path;"
        repair_prompt = mock_agent.run.await_args.args[0]
        assert "Find orphan files" in repair_prompt
        assert "Invalid type vertex for 'NOT'." in repair_prompt


class TestCleanCypherResponse:
    def test_remove_line_comments(self):
        input_query = """-- Count all users
        MATCH (u:User)
        -- Return only active users
        WHERE u.active = true
        RETURN count(u) as total -- Add comment at end of line"""
        cleaned = _clean_cypher_response(input_query)
        assert "--" not in cleaned
        assert "MATCH (u:User)" in cleaned
        assert "WHERE u.active = true" in cleaned
        assert "RETURN count(u) as total" in cleaned

    def test_remove_block_comments(self):
        input_query = """/* This is a multi
        line block comment */
        MATCH (n) /* inline comment */
        RETURN n /* another comment */"""
        cleaned = _clean_cypher_response(input_query)
        assert "/*" not in cleaned
        assert "*/" not in cleaned
        assert "MATCH (n)" in cleaned
        assert "RETURN n" in cleaned

    def test_remove_multiple_queries(self):
        input_query = """MATCH (n) RETURN count(n);
        MATCH (u:User) RETURN u.name;"""
        cleaned = _clean_cypher_response(input_query)
        assert cleaned.count(";") == 1
        assert "MATCH (n) RETURN count(n)" in cleaned
        assert "User" not in cleaned

    def test_convert_node_label_pipe_syntax(self):
        input_query = (
            "MATCH (n:Class|Function|Method) WHERE n.name = 'test' RETURN n.path"
        )
        cleaned = _clean_cypher_response(input_query)
        assert "(n:Class|Function|Method)" not in cleaned
        assert "labels(n)[0] IN ['Class', 'Function', 'Method']" in cleaned
        assert "MATCH (n) WHERE" in cleaned

    def test_convert_relationship_pipe_syntax_all_directions(self):
        # Undirected
        input1 = "MATCH (a)-[r:CALLS|USES|IMPORTS]-(b) RETURN count(r)"
        cleaned1 = _clean_cypher_response(input1)
        assert "[r:CALLS|USES|IMPORTS]" not in cleaned1
        assert "type(r) IN ['CALLS', 'USES', 'IMPORTS']" in cleaned1

        # Directed right
        input2 = "MATCH (a)-[r:CALLS|USES]->(b) RETURN count(r)"
        cleaned2 = _clean_cypher_response(input2)
        assert "[r:CALLS|USES]" not in cleaned2
        assert "-[r]->" in cleaned2
        assert "type(r) IN ['CALLS', 'USES']" in cleaned2

        # Directed left
        input3 = "MATCH (a)<-[r:CALLS|IMPORTS]-(b) RETURN count(r)"
        cleaned3 = _clean_cypher_response(input3)
        assert "[r:CALLS|IMPORTS]" not in cleaned3
        assert "<-[r]-" in cleaned3
        assert "type(r) IN ['CALLS', 'IMPORTS']" in cleaned3

        # No relationship variable
        input4 = "MATCH (a)-[:CALLS|USES]-(b) RETURN count(*)"
        cleaned4 = _clean_cypher_response(input4)
        assert "[:CALLS|USES]" not in cleaned4
        assert "type(r) IN ['CALLS', 'USES']" in cleaned4

    def test_combine_multiple_pipe_syntax_in_same_query(self):
        input_query = """MATCH (a:Class|Interface)-[r:CALLS|IMPORTS]->(b:Function|Method)
        WHERE a.name = 'Test'
        RETURN b.path"""
        cleaned = _clean_cypher_response(input_query)
        assert ":Class|Interface" not in cleaned
        assert ":Function|Method" not in cleaned
        assert ":CALLS|IMPORTS" not in cleaned
        assert "labels(a)[0] IN ['Class', 'Interface']" in cleaned
        assert "labels(b)[0] IN ['Function', 'Method']" in cleaned
        assert "type(r) IN ['CALLS', 'IMPORTS']" in cleaned

    def test_survives_valid_cypher(self):
        input_query = """MATCH (f:File)
        WHERE f.extension = '.py'
        RETURN f.path, f.size
        LIMIT 100;"""
        cleaned = _clean_cypher_response(input_query)
        assert cleaned.strip() == input_query.strip()


class TestCreateRagOrchestrator:
    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    @patch("codebase_rag.services.llm.Agent")
    @patch("codebase_rag.services.llm.build_rag_orchestrator_prompt")
    def test_creates_agent_with_tools(
        self,
        mock_build_prompt: MagicMock,
        mock_agent: MagicMock,
        mock_get_provider: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        mock_config = MagicMock()
        mock_settings.active_orchestrator_config = mock_config
        mock_settings.AGENT_RETRIES = 3
        mock_settings.ORCHESTRATOR_OUTPUT_RETRIES = 2

        mock_provider = MagicMock()
        mock_provider.create_model.return_value = MagicMock()
        mock_get_provider.return_value = mock_provider

        mock_build_prompt.return_value = "System prompt"
        mock_agent.return_value = MagicMock()

        tools = [MagicMock(), MagicMock()]
        result = create_rag_orchestrator(tools)

        mock_agent.assert_called_once()
        call_kwargs = mock_agent.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert result is not None

    @patch("codebase_rag.services.llm.settings")
    @patch("codebase_rag.services.llm.get_provider_from_config")
    def test_raises_on_error(
        self, mock_get_provider: MagicMock, mock_settings: MagicMock
    ) -> None:
        mock_settings.active_orchestrator_config = MagicMock()
        mock_get_provider.side_effect = Exception("Config error")

        with pytest.raises(ex.LLMGenerationError):
            create_rag_orchestrator([])
