# Build safe demo .eml fixtures
# No real malware. Executable samples are MZ stubs only.
# Oversized attachment is generated at prepare time (not stored in git).

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "backend" / "tests" / "fixtures" / "emails"
INBOX = ROOT / "data" / "inbox"

# Minimal valid 1x1 PNG
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)
PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
MZ_STUB = b"MZ" + b"\x00" * 64
MALFORMED_ZIP = b"PK\x03\x04" + b"not-a-valid-central-directory"


def _b64(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))


def _zip_with_members() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "safe notes")
        archive.writestr("update.js", 'console.log("demo");')
    return buffer.getvalue()


def write_eml(
    path: Path,
    *,
    subject: str,
    body: str,
    attachments: list[tuple[str, str, bytes]],
    sender: str = "demo@sentinelai.test",
) -> None:
    boundary = "kris-demo-boundary"
    lines = [
        f"From: {sender}",
        "To: analyst@company.test",
        f"Subject: {subject}",
        "MIME-Version: 1.0",
        f'Content-Type: multipart/mixed; boundary="{boundary}"',
        "",
        f"--{boundary}",
        'Content-Type: text/plain; charset="utf-8"',
        "",
        body,
    ]
    for filename, content_type, payload in attachments:
        lines.extend(
            [
                f"--{boundary}",
                f"Content-Type: {content_type}",
                f'Content-Disposition: attachment; filename="{filename}"',
                "Content-Transfer-Encoding: base64",
                "",
                _b64(payload),
            ]
        )
    lines.extend([f"--{boundary}--", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def build_fixtures(*, include_oversized: bool = True) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    write_eml(
        FIXTURES / "01_normal_pdf.eml",
        subject="Normal PDF attachment",
        body="Benign statement.\nhttps://example.com/login\n",
        attachments=[("statement.pdf", "application/pdf", PDF_BYTES)],
    )

    write_eml(
        FIXTURES / "02_normal_image.eml",
        subject="Normal PNG attachment",
        body="Logo attached.\nhttps://example.com/\n",
        attachments=[("logo.png", "image/png", PNG_BYTES)],
    )

    write_eml(
        FIXTURES / "03_executable_extension.eml",
        subject="Executable attachment",
        body="Please run the updater.\nhttps://example.com/update\n",
        attachments=[("updater.exe", "application/octet-stream", MZ_STUB)],
    )

    write_eml(
        FIXTURES / "04_double_extension.eml",
        subject="Double extension attachment",
        body="Invoice attached.\nhttps://example.com/invoice\n",
        attachments=[("invoice.pdf.exe", "application/octet-stream", MZ_STUB)],
    )

    write_eml(
        FIXTURES / "05_signature_mismatch.eml",
        subject="PDF name with executable bytes",
        body="Contract attached.\nhttps://example.org/contract\n",
        attachments=[("contract.pdf", "application/pdf", MZ_STUB)],
    )

    zip_bytes = _zip_with_members()
    write_eml(
        FIXTURES / "06_zip_archive.eml",
        subject="ZIP with nested script",
        body="Archive attached.\nhttps://example.com/files\n",
        attachments=[("documents.zip", "application/zip", zip_bytes)],
    )

    write_eml(
        FIXTURES / "07_empty_attachment.eml",
        subject="Empty attachment",
        body="Empty notes file.\nhttps://example.com/notes\n",
        attachments=[("notes.txt", "text/plain", b"")],
    )

    write_eml(
        FIXTURES / "08_malformed_zip.eml",
        subject="Malformed ZIP attachment",
        body="Broken archive.\nhttps://example.com/broken\n",
        attachments=[("broken.zip", "application/zip", MALFORMED_ZIP)],
    )

    # SHA-256 correctness helper email (known payload)
    digest = hashlib.sha256(PDF_BYTES).hexdigest()
    write_eml(
        FIXTURES / "09_sha256_known_pdf.eml",
        subject=f"SHA256 demo pdf:{digest}",
        body=f"Known PDF sha256={digest}\nhttps://example.com/hash\n",
        attachments=[("known.pdf", "application/pdf", PDF_BYTES)],
    )

def build_oversized_into_inbox() -> None:
    """5 MiB+ attachment — written only under data/inbox (gitignored)."""
    large = b"%PDF-1.4\n" + (b"A" * (5 * 1024 * 1024)) + b"\n%%EOF\n"
    write_eml(
        INBOX / "10_oversized_attachment.eml",
        subject="Oversized attachment",
        body="Large PDF attached.\nhttps://example.com/large\n",
        attachments=[("large.pdf", "application/pdf", large)],
    )


def sync_inbox() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    for path in FIXTURES.glob("*.eml"):
        target = INBOX / path.name
        target.write_bytes(path.read_bytes())
        print(f"synced {target.relative_to(ROOT)}")


if __name__ == "__main__":
    build_fixtures()
    sync_inbox()
    if True:
        build_oversized_into_inbox()
    print("Done. Scan inbox or rerun-analysis to load these emails.")
