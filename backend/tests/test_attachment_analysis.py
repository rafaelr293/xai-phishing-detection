from app.analysis.attachment import AttachmentAnalyzer
from app.domain.schemas import AttachmentData, ParsedEmail


def test_executable_signature_and_double_extension_are_high_risk() -> None:
    email = ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        attachments=[
            AttachmentData(
                filename="invoice.pdf.exe",
                content_type="application/pdf",
                size_bytes=128,
                sha256="b" * 64,
                detected_type="windows_executable",
            )
        ],
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 100
    assert {indicator.type for indicator in result.indicators} == {
        "dangerous_extension",
        "double_extension",
        "executable_signature",
    }


def test_safe_pdf_with_unknown_signature_is_not_marked_malicious() -> None:
    email = ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        attachments=[
            AttachmentData(
                filename="invoice.pdf",
                content_type="application/pdf",
                size_bytes=128,
                sha256="b" * 64,
            )
        ],
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 0
    assert result.confidence == 0.85
    assert result.indicators == []


def test_unrecognized_extension_reduces_confidence() -> None:
    email = ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        attachments=[
            AttachmentData(
                filename="document.xyz",
                content_type="application/octet-stream",
                size_bytes=128,
                sha256="b" * 64,
            )
        ],
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.confidence == 0.70
    assert result.evidence["verification_failures"] == [
        {"type": "unrecognized_extension", "filename": "document.xyz", "extension": ".xyz"},
        {"type": "unknown_signature", "filename": "document.xyz"},
    ]


def test_archive_script_member_is_flagged() -> None:
    email = ParsedEmail(
        content_hash="a" * 64,
        source_path="sample.eml",
        attachments=[
            AttachmentData(
                filename="documents.zip",
                content_type="application/zip",
                size_bytes=128,
                sha256="b" * 64,
                detected_type="zip_archive",
                archive_members=["readme.txt", "update.js"],
            )
        ],
    )

    result = AttachmentAnalyzer().analyze(email)

    assert result.risk_score == 75
    assert result.indicators[0].type == "archive_contains_script"
