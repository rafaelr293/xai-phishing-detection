import base64
import io
import zipfile
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


def test_parse_eml_lists_archive_members_without_extracting(tmp_path: Path) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("readme.txt", "safe test content")
        archive.writestr("update.js", "safe test content")
    encoded_archive = base64.b64encode(archive_buffer.getvalue()).decode("ascii")
    source = tmp_path / "archive.eml"
    source.write_text(
        'From: sender@example.com\n'
        'To: analyst@example.com\n'
        'Subject: Archive\n'
        'MIME-Version: 1.0\n'
        'Content-Type: multipart/mixed; boundary="b"\n\n'
        '--b\nContent-Type: application/zip\n'
        'Content-Disposition: attachment; filename="documents.zip"\n'
        'Content-Transfer-Encoding: base64\n\n'
        f'{encoded_archive}\n--b--\n',
        encoding="utf-8",
    )

    parsed = parse_eml(source, 1024 * 1024)

    assert parsed.attachments[0].detected_type == "zip_archive"
    assert parsed.attachments[0].archive_members == ["readme.txt", "update.js"]
    assert parsed.attachments[0].archive_malformed is False


def test_parse_eml_marks_malformed_zip_archive(tmp_path: Path) -> None:
    malformed_zip = b"PK\x03\x04" + b"not-a-valid-zip-central-directory"
    encoded_archive = base64.b64encode(malformed_zip).decode("ascii")
    source = tmp_path / "bad-archive.eml"
    source.write_text(
        'From: sender@example.com\n'
        'To: analyst@example.com\n'
        'Subject: Bad archive\n'
        'MIME-Version: 1.0\n'
        'Content-Type: multipart/mixed; boundary="b"\n\n'
        '--b\nContent-Type: application/zip\n'
        'Content-Disposition: attachment; filename="broken.zip"\n'
        'Content-Transfer-Encoding: base64\n\n'
        f'{encoded_archive}\n--b--\n',
        encoding="utf-8",
    )

    parsed = parse_eml(source, 1024 * 1024)

    assert parsed.attachments[0].detected_type == "zip_archive"
    assert parsed.attachments[0].archive_members == []
    assert parsed.attachments[0].archive_malformed is True


def test_parse_eml_sha256_matches_attachment_payload(tmp_path: Path) -> None:
    payload = b"%PDF-1.4 hello"
    encoded = base64.b64encode(payload).decode("ascii")
    source = tmp_path / "hash.eml"
    source.write_text(
        'From: sender@example.com\n'
        'To: analyst@example.com\n'
        'Subject: Hash\n'
        'MIME-Version: 1.0\n'
        'Content-Type: multipart/mixed; boundary="b"\n\n'
        '--b\nContent-Type: application/pdf\n'
        'Content-Disposition: attachment; filename="ok.pdf"\n'
        'Content-Transfer-Encoding: base64\n\n'
        f'{encoded}\n--b--\n',
        encoding="utf-8",
    )

    parsed = parse_eml(source, 1024 * 1024)

    assert parsed.attachments[0].sha256 == __import__("hashlib").sha256(payload).hexdigest()
    assert parsed.attachments[0].detected_type == "pdf"
