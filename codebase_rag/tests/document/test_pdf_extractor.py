from __future__ import annotations

from codebase_rag.document.extractors.pdf_extractor import PDFExtractor


def test_build_document_from_pages_creates_line_spans(tmp_path) -> None:
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")

    extractor = PDFExtractor()
    document = extractor._build_document_from_pages(
        page_texts=["first line\nsecond line", "third line"],
        validated_path=pdf_file,
        original_path=pdf_file,
    )

    assert document.content == "Page 1\nfirst line\nsecond line\nPage 2\nthird line"
    assert len(document.sections) == 2
    assert document.sections[0].title == "Page 1"
    assert document.sections[0].start_line == 0
    assert document.sections[0].end_line == 2
    assert document.sections[1].title == "Page 2"
    assert document.sections[1].start_line == 3
    assert document.sections[1].end_line == 4