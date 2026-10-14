import hashlib

from app.analysis.attachment import LARGE_ATTACHMENT_BYTES, AttachmentAnalyzer
from app.domain.schemas import AttachmentData, ParsedEmail


def _email_with(*attachments: AttachmentData) -> ParsedEmail:
    return ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        attachments=list(attachments),
    )


def test_executable_signature_and_double_extension_are_high_risk() -> None:
    email = _email_with(
        AttachmentData(
            filename="invoice.pdf.exe",
            content_type="application/pdf",
            size_bytes=128,
            sha256="b" * 64,
            detected_type="windows_executable",
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 100
    assert {indicator.type for indicator in result.indicators} == {
        "dangerous_extension",
        "double_extension",
        "executable_signature",
    }


def test_safe_pdf_with_unknown_signature_is_not_marked_malicious() -> None:
    email = _email_with(
        AttachmentData(
            filename="invoice.pdf",
            content_type="application/pdf",
            size_bytes=128,
            sha256="b" * 64,
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 15
    assert result.confidence == 0.85
    assert result.indicators[0].type == "unknown_attachment_type"


def test_unrecognized_extension_reduces_confidence() -> None:
    email = _email_with(
        AttachmentData(
            filename="document.xyz",
            content_type="application/octet-stream",
            size_bytes=128,
            sha256="b" * 64,
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 15
    assert result.confidence == 0.70
    assert result.evidence["verification_failures"] == [
        {"type": "unrecognized_extension", "filename": "document.xyz", "extension": ".xyz"},
        {"type": "unknown_signature", "filename": "document.xyz"},
    ]


def test_archive_script_member_is_flagged() -> None:
    email = _email_with(
        AttachmentData(
            filename="documents.zip",
            content_type="application/zip",
            size_bytes=128,
            sha256="b" * 64,
            detected_type="zip_archive",
            archive_members=["readme.txt", "update.js"],
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 75
    assert result.indicators[0].type == "archive_contains_script"


def test_signature_mismatch_is_high_risk() -> None:
    email = _email_with(
        AttachmentData(
            filename="invoice.pdf",
            content_type="application/pdf",
            size_bytes=128,
            sha256="b" * 64,
            detected_type="windows_executable",
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert "file_signature_mismatch" in {indicator.type for indicator in result.indicators}
    assert "executable_signature" in {indicator.type for indicator in result.indicators}
    assert result.risk_score == 100


def test_empty_attachment_is_flagged_for_any_extension() -> None:
    email = _email_with(
        AttachmentData(
            filename="payload.exe",
            content_type="application/octet-stream",
            size_bytes=0,
            sha256=hashlib.sha256(b"").hexdigest(),
            detected_type="unknown",
        )
    )

    result = AttachmentAnalyzer().analyze(email)
    types = {indicator.type for indicator in result.indicators}

    assert "empty_attachment" in types
    assert "dangerous_extension" in types


def test_malformed_archive_is_flagged() -> None:
    email = _email_with(
        AttachmentData(
            filename="broken.zip",
            content_type="application/zip",
            size_bytes=32,
            sha256="c" * 64,
            detected_type="zip_archive",
            archive_members=[],
            archive_malformed=True,
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.indicators[0].type == "malformed_archive"
    assert result.risk_score == 40


def test_macro_and_script_extensions_are_flagged() -> None:
    email = _email_with(
        AttachmentData(
            filename="budget.xlsm",
            content_type="application/vnd.ms-excel.sheet.macroEnabled.12",
            size_bytes=64,
            sha256="d" * 64,
            detected_type="unknown",
        ),
        AttachmentData(
            filename="run.ps1",
            content_type="text/plain",
            size_bytes=64,
            sha256="e" * 64,
            detected_type="unknown",
        ),
    )

    result = AttachmentAnalyzer().analyze(email)
    types = {indicator.type for indicator in result.indicators}

    assert "macro_enabled_document" in types
    assert "script_attachment" in types


def test_large_attachment_indicator() -> None:
    email = _email_with(
        AttachmentData(
            filename="report.pdf",
            content_type="application/pdf",
            size_bytes=LARGE_ATTACHMENT_BYTES,
            sha256="f" * 64,
            detected_type="pdf",
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.indicators[0].type == "large_attachment"
    assert result.risk_score == 10


def test_sha256_passthrough_in_evidence_path_is_stable() -> None:
    payload = b"%PDF-1.4 test"
    digest = hashlib.sha256(payload).hexdigest()
    email = _email_with(
        AttachmentData(
            filename="ok.pdf",
            content_type="application/pdf",
            size_bytes=len(payload),
            sha256=digest,
            detected_type="pdf",
        )
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 0
    assert result.confidence == 0.95
    assert email.attachments[0].sha256 == digest
