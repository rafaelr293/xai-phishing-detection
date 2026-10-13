from pathlib import Path

from app.ingestion.eml_parser import parse_eml


def test_parse_eml_extracts_body_url_and_attachment(tmp_path: Path) -> None:
    source = tmp_path / "sample.eml"
    source.write_text(
        'From: sender@example.com\n'
        'To: analyst@example.com\n'
        'Subject: Verify account\n'
        'MIME-Version: 1.0\n'
        'Content-Type: multipart/mixed; boundary="b"\n\n'
        '--b\nContent-Type: text/plain\n\nVisit https://example.com/login\n'
        '--b\nContent-Type: application/pdf\n'
        'Content-Disposition: attachment; filename="invoice.pdf"\n\n'
        'not-a-real-pdf\n--b--\n',
        encoding="utf-8",
    )

    parsed = parse_eml(source, 1024 * 1024)

    assert parsed.sender == "sender@example.com"
    assert parsed.urls == ["https://example.com/login"]
    assert parsed.attachments[0].filename == "invoice.pdf"
    assert parsed.attachments[0].size_bytes > 0


def test_parse_eml_rejects_oversized_file(tmp_path: Path) -> None:
    source = tmp_path / "large.eml"
    source.write_bytes(b"x" * 20)

    try:
        parse_eml(source, 10)
    except ValueError as error:
        assert "size limit" in str(error)
    else:
        raise AssertionError("Expected oversized email to be rejected")
