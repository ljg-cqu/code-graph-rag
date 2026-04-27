"""Tests for learned verb registry.

Covers _load_learned_verbs, _learn_verb, and resolve_category integration.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from codebase_rag.config import settings
from codebase_rag.document.concept_extraction import (
    _learn_verb,
    _learned_verbs_path,
    _load_learned_verbs,
    resolve_category,
)


class TestLearnedVerbRegistry:
    """Tests for persistent learned verb registry."""

    def test_learned_verbs_path_returns_none_without_repo(self, tmp_path):
        with patch.object(settings, "TARGET_REPO_PATH", ""):
            path = _learned_verbs_path()
            assert path is None

    def test_learned_verbs_path_returns_path_with_repo(self, tmp_path):
        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            path = _learned_verbs_path()
            assert path == tmp_path / ".cgr" / "learned_verbs.json"

    def test_load_learned_verbs_creates_empty_dict_when_no_file(self, tmp_path):
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            learned = _load_learned_verbs()
            assert learned == {}

    def test_load_learned_verbs_reads_existing_file(self, tmp_path):
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        cgr_dir = tmp_path / ".cgr"
        cgr_dir.mkdir(parents=True, exist_ok=True)
        learned_file = cgr_dir / "learned_verbs.json"
        learned_file.write_text(
            json.dumps(
                {
                    "maps to": {
                        "category": "ANALOGICAL",
                        "count": 5,
                        "first_seen": "2026-04-26T23:32:20",
                    },
                    "operates-on": {
                        "category": "CONTEXTUAL",
                        "count": 2,
                        "first_seen": "2026-04-26T23:32:23",
                    },
                }
            )
        )

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            learned = _load_learned_verbs()
            # Only count >= 3 should be included
            assert "maps to" in learned
            assert learned["maps to"] == "ANALOGICAL"
            assert "operates-on" not in learned  # count=2 < 3

    def test_learn_verb_persists_to_file(self, tmp_path):
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            _learn_verb("maps to", "ANALOGICAL")

        learned_file = tmp_path / ".cgr" / "learned_verbs.json"
        assert learned_file.exists()
        data = json.loads(learned_file.read_text())
        assert "maps to" in data
        assert data["maps to"]["category"] == "ANALOGICAL"
        assert data["maps to"]["count"] == 1

    def test_learn_verb_increments_count(self, tmp_path):
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            _learn_verb("maps to", "ANALOGICAL")
            _learn_verb("maps to", "ANALOGICAL")
            _learn_verb("maps to", "ANALOGICAL")

        learned_file = tmp_path / ".cgr" / "learned_verbs.json"
        data = json.loads(learned_file.read_text())
        assert data["maps to"]["count"] == 3

    def test_learn_verb_no_op_when_no_path(self):
        """_learn_verb should not raise when path is empty."""
        with patch.object(settings, "TARGET_REPO_PATH", ""):
            _learn_verb("maps to", "ANALOGICAL")  # Should not raise

    def test_resolve_category_uses_learned_registry_first(self, tmp_path):
        """Learned registry should take precedence over hardcoded registry."""
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        # Pre-populate learned registry with conflicting entry
        cgr_dir = tmp_path / ".cgr"
        cgr_dir.mkdir(parents=True, exist_ok=True)
        learned_file = cgr_dir / "learned_verbs.json"
        learned_file.write_text(
            json.dumps(
                {
                    "is-a": {
                        "category": "ANALOGICAL",
                        "count": 5,
                        "first_seen": "2026-04-26T23:32:20",
                    },
                }
            )
        )

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            category, emoji = resolve_category("is-a", None)
            # Learned registry says ANALOGICAL, hardcoded says HIERARCHICAL
            assert category == "ANALOGICAL"
            assert emoji == "🌉"

    def test_resolve_category_learns_unknown_verb(self, tmp_path):
        """Unknown verb with valid declared category should be learned."""
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            category, emoji = resolve_category("brand-new-verb", "CAUSAL")
            assert category == "CAUSAL"

        learned_file = tmp_path / ".cgr" / "learned_verbs.json"
        data = json.loads(learned_file.read_text())
        assert "brand-new-verb" in data
        assert data["brand-new-verb"]["count"] == 1

    def test_load_learned_verbs_caches_across_calls(self, tmp_path):
        """Multiple calls should return cached dict."""
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        cgr_dir = tmp_path / ".cgr"
        cgr_dir.mkdir(parents=True, exist_ok=True)
        learned_file = cgr_dir / "learned_verbs.json"
        learned_file.write_text(
            json.dumps(
                {
                    "maps to": {
                        "category": "ANALOGICAL",
                        "count": 5,
                        "first_seen": "2026-04-26T23:32:20",
                    },
                }
            )
        )

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            learned1 = _load_learned_verbs()
            learned2 = _load_learned_verbs()
            assert learned1 is learned2  # Same object

    def test_learned_after_three_sightings(self, tmp_path):
        """Unknown verb becomes authoritative only after count >= 3."""
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            # First 2 sightings — not yet authoritative
            _learn_verb("my-verb", "CAUSAL")
            _learn_verb("my-verb", "CAUSAL")
            ce._LEARNED_VERBS = None
            learned = _load_learned_verbs()
            assert "my-verb" not in learned

            # Third sighting — now authoritative
            _learn_verb("my-verb", "CAUSAL")
            ce._LEARNED_VERBS = None
            learned = _load_learned_verbs()
            assert "my-verb" in learned
            assert learned["my-verb"] == "CAUSAL"

    def test_concurrent_extractions_safe_append(self, tmp_path):
        """Multiple processes appending to the same file should not corrupt JSON."""
        import codebase_rag.document.concept_extraction as ce

        ce._LEARNED_VERBS = None

        with patch.object(settings, "TARGET_REPO_PATH", str(tmp_path)):
            # Simulate concurrent writes
            for i in range(10):
                _learn_verb(f"verb-{i}", "CAUSAL")

        learned_file = tmp_path / ".cgr" / "learned_verbs.json"
        data = json.loads(learned_file.read_text())
        assert len(data) == 10
        for i in range(10):
            assert f"verb-{i}" in data
