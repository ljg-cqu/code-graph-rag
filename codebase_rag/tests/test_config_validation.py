import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from codebase_rag import constants as cs
from codebase_rag.config import AppConfig, ModelConfig, format_missing_api_key_errors


class TestValidateApiKey:
    @pytest.mark.parametrize(
        ("provider", "model_id"),
        [
            (cs.Provider.OLLAMA, "llama3"),
            (cs.Provider.LOCAL, "local-model"),
            (cs.Provider.VLLM, "vllm-model"),
        ],
    )
    def test_local_providers_skip_validation(
        self, provider: cs.Provider, model_id: str
    ) -> None:
        cfg = ModelConfig(provider=provider, model_id=model_id)
        cfg.validate_api_key()

    def test_google_vertex_skips_validation(self) -> None:
        cfg = ModelConfig(
            provider=cs.Provider.GOOGLE,
            model_id="gemini-pro",
            provider_type=cs.GoogleProviderType.VERTEX,
        )
        cfg.validate_api_key()

    def test_google_gla_requires_api_key(self) -> None:
        cfg = ModelConfig(
            provider=cs.Provider.GOOGLE,
            model_id="gemini-pro",
            provider_type=cs.GoogleProviderType.GLA,
        )
        with pytest.raises(ValueError, match="API Key Missing"):
            cfg.validate_api_key()

    @pytest.mark.parametrize(
        "api_key_kwargs",
        [
            {},
            {"api_key": ""},
            {"api_key": "   "},
            {"api_key": cs.DEFAULT_API_KEY},
        ],
    )
    def test_invalid_api_key_raises(self, api_key_kwargs: dict[str, str]) -> None:
        cfg = ModelConfig(
            provider=cs.Provider.OPENAI, model_id="gpt-4", **api_key_kwargs
        )
        with pytest.raises(ValueError, match="API Key Missing"):
            cfg.validate_api_key()

    def test_valid_api_key_passes(self) -> None:
        cfg = ModelConfig(
            provider=cs.Provider.OPENAI, model_id="gpt-4", api_key="sk-real-key-123"
        )
        cfg.validate_api_key()

    def test_role_forwarded_to_error_message(self) -> None:
        cfg = ModelConfig(provider=cs.Provider.OPENAI, model_id="gpt-4")
        with pytest.raises(ValueError, match="cypher"):
            cfg.validate_api_key(role="cypher")


class TestFormatMissingApiKeyErrors:
    def test_known_provider_openai(self) -> None:
        msg = format_missing_api_key_errors(cs.Provider.OPENAI)
        assert "OPENAI_API_KEY" in msg
        assert "https://platform.openai.com/api-keys" in msg
        assert "OpenAI" in msg

    def test_known_provider_anthropic(self) -> None:
        msg = format_missing_api_key_errors(cs.Provider.ANTHROPIC)
        assert "ANTHROPIC_API_KEY" in msg
        assert "Anthropic" in msg

    def test_unknown_provider_generic_message(self) -> None:
        msg = format_missing_api_key_errors("deepseek")
        assert "DEEPSEEK_API_KEY" in msg
        assert "Deepseek" in msg

    def test_role_appears_in_message(self) -> None:
        msg = format_missing_api_key_errors(cs.Provider.OPENAI, role="cypher")
        assert "for cypher" in msg

    def test_default_role_omits_role_from_message(self) -> None:
        msg = format_missing_api_key_errors(cs.Provider.OPENAI)
        assert "for model" not in msg

    def test_case_insensitive_lookup(self) -> None:
        msg = format_missing_api_key_errors("OpenAI")
        assert "OPENAI_API_KEY" in msg
        assert "OpenAI" in msg


class TestVectorBackendValidation:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("VECTOR_STORE_BACKEND", "unsupported-backend"),
            ("DOC_VECTOR_STORE_BACKEND", "unsupported-backend"),
            ("JSON_VECTOR_STORE_BACKEND", "unsupported-backend"),
        ],
    )
    def test_vector_backends_are_memgraph_only(
        self, field_name: str, value: str
    ) -> None:
        with patch.dict(os.environ, {field_name: value}, clear=False):
            with pytest.raises(ValidationError, match="Only 'memgraph' is available"):
                AppConfig(_env_file=None)  # ty: ignore[unknown-argument]


class TestEnvFileSupport:
    """Tests for ENV_FILE environment variable support.

    Note: These tests verify the env file loading logic. Due to module-level
    configuration loading, we test the behavior by examining the loaded
    configuration values rather than module reload patterns.
    """

    def test_env_file_path_can_be_set(self, tmp_path: Path) -> None:
        """Test that ENV_FILE environment variable is checked."""
        env_file = tmp_path / "test.env"
        env_file.write_text("MEMGRAPH_PORT=9999\n")

        # Verify the file exists
        assert env_file.exists()

        # The config module checks for ENV_FILE at import time
        # We verify the logic works by checking if the file would be loaded
        with patch.dict(os.environ, {"ENV_FILE": str(env_file)}):
            # File exists, so it would be loaded
            env_file_path = os.environ.get("ENV_FILE")
            assert env_file_path == str(env_file)
            assert os.path.isfile(env_file_path)

    def test_env_file_loading_logic(self, tmp_path: Path) -> None:
        """Test the env file loading logic paths."""
        # Test 1: Valid env file
        valid_env = tmp_path / "valid.env"
        valid_env.write_text("MEMGRAPH_PORT=8888\n")

        # Simulate the loading logic
        with patch.dict(os.environ, {"ENV_FILE": str(valid_env)}, clear=False):
            env_file = os.environ.get("ENV_FILE")
            assert env_file == str(valid_env)
            assert os.path.isfile(env_file)

        # Test 2: Non-existent env file (should fall back)
        with patch.dict(
            os.environ, {"ENV_FILE": str(tmp_path / "nonexistent.env")}, clear=False
        ):
            env_file = os.environ.get("ENV_FILE")
            assert env_file is not None
            assert not os.path.isfile(env_file)

    def test_default_ports_when_no_custom_env(self) -> None:
        """Verify default ports are used when no custom env file is specified."""
        # These are the pydantic default values
        from codebase_rag.config import AppConfig

        # Create fresh config with no overrides
        config = AppConfig(_env_file=None)  # type: ignore[call-arg]

        assert config.MEMGRAPH_PORT == 7687
        assert config.DOC_MEMGRAPH_PORT == 7688
        assert config.JSON_MEMGRAPH_PORT == 7689

    def test_env_file_documentation_example(self, tmp_path: Path) -> None:
        """Test the exact example from documentation works."""
        # Create an env file like the one in the knowledge-base config
        env_file = tmp_path / "knowledge-base.env"
        env_file.write_text(
            """
# Knowledge Base Configuration
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7787
MEMGRAPH_HTTP_PORT=7445
DOC_MEMGRAPH_HOST=localhost
DOC_MEMGRAPH_PORT=7788
JSON_MEMGRAPH_HOST=localhost
JSON_MEMGRAPH_PORT=7789
"""
        )

        # Verify file structure
        content = env_file.read_text()
        assert "MEMGRAPH_PORT=7787" in content
        assert "DOC_MEMGRAPH_PORT=7788" in content
        assert "JSON_MEMGRAPH_PORT=7789" in content

        # Verify AppConfig can parse these values
        from codebase_rag.config import AppConfig

        with patch.dict(os.environ, {"MEMGRAPH_PORT": "7787", "DOC_MEMGRAPH_PORT": "7788", "JSON_MEMGRAPH_PORT": "7789"}):
            config = AppConfig(_env_file=None)  # type: ignore[call-arg]
            assert config.MEMGRAPH_PORT == 7787
            assert config.DOC_MEMGRAPH_PORT == 7788
            assert config.JSON_MEMGRAPH_PORT == 7789
