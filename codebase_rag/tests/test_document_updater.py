from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_rag.document.document_updater import DocumentGraphUpdater


def test_collect_documents_skips_internal_artifacts(tmp_path: Path) -> None:
    included = tmp_path / "guide.md"
    included.write_text("# Guide\n", encoding="utf-8")

    egg_info_dir = tmp_path / "demo.egg-info"
    egg_info_dir.mkdir()
    (egg_info_dir / "artifact.md").write_text("# Artifact\n", encoding="utf-8")

    cgr_dir = tmp_path / ".cgr"
    cgr_dir.mkdir()
    (cgr_dir / "state.md").write_text("# Internal\n", encoding="utf-8")

    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    documents = updater._collect_documents()

    assert included in documents
    assert egg_info_dir / "artifact.md" not in documents
    assert cgr_dir / "state.md" not in documents