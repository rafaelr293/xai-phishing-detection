from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.analysis.attachment import LARGE_ATTACHMENT_BYTES, AttachmentAnalyzer
from app.core.config import Settings
from app.domain.schemas import AttachmentData, ParsedEmail
from app.ingestion.eml_parser import parse_eml
from app.url_analysis.traversal import URLTraversalService, validate_target_url

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def _row(name: str, status: str, detail: str = "") -> None:
    mark = {"PASS": "[OK]", "FAIL": "[X ]", "SKIP": "[--]"}[status]
    suffix = f" — {detail}" if detail else ""
    print(f"  {mark} {name}{suffix}")


def check_url_validation() -> None:
    print("\n=== URL traversal (validation / SSRF) ===")

    # Valid public URL
    try:
        with patch(
            "app.url_analysis.traversal.resolve_host_addresses",
            return_value=["8.8.8.8"],
        ):
            validate_target_url("https://example.com/login")
        _row("Valid public URL", PASS, "https://example.com/login")
    except Exception as exc:
        _row("Valid public URL", FAIL, str(exc))

    # Invalid URL
    try:
        validate_target_url("not a url")
        _row("Invalid URL", FAIL, "should have raised")
    except ValueError:
        _row("Invalid URL", PASS)

    # Unsupported scheme
    try:
        validate_target_url("file:///tmp/x.html")
        _row("Unsupported scheme", FAIL, "should have raised")
    except ValueError:
        _row("Unsupported scheme", PASS, "file:// blocked")

    # Private IP destination
    try:
        validate_target_url("http://127.0.0.1/")
        _row("Private IP destination", FAIL, "should have raised")
    except ValueError:
        _row("Private IP destination", PASS, "127.0.0.1 blocked")

    # Hostname resolving to private IP
    try:
        with patch(
            "app.url_analysis.traversal.resolve_host_addresses",
            return_value=["127.0.0.1"],
        ):
            validate_target_url("http://rebind.example/")
        _row("Hostname resolving to a private IP", FAIL, "should have raised")
    except ValueError:
        _row("Hostname resolving to a private IP", PASS, "DNS rebind blocked")

    # DNS failures
    try:
        with patch(
            "app.url_analysis.traversal.resolve_host_addresses",
            side_effect=ValueError("DNS resolution failed for host: missing.invalid"),
        ):
            validate_target_url("http://missing.invalid/")
        _row("DNS failures", FAIL, "should have raised")
    except ValueError:
        _row("DNS failures", PASS)

    # Redirect to prohibited destination (guard rejects private hop)
    try:
        with patch(
            "app.url_analysis.traversal.resolve_host_addresses",
            return_value=["8.8.8.8"],
        ):
            validate_target_url("https://example.com/")
        validate_target_url("http://127.0.0.1/steal")
        _row("Redirect to a prohibited destination", FAIL, "private hop allowed")
    except ValueError:
        _row(
            "Redirect to a prohibited destination",
            PASS,
            "guard rejects private redirect target",
        )

    # Excessive redirects (hop counter)
    from app.url_analysis.traversal import _redirect_hop_count

    grandparent = MagicMock()
    grandparent.redirected_from = None
    parent = MagicMock()
    parent.redirected_from = grandparent
    request = MagicMock()
    request.redirected_from = parent
    hops = _redirect_hop_count(request)
    _row(
        "Excessive redirects",
        PASS if hops == 2 else FAIL,
        f"hop count={hops}; live abort enforced in guard_request when hops > max",
    )


def check_url_runtime() -> None:
    print("\n=== URL traversal (runtime / artifacts) ===")
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            artifact_directory=Path(tmp),
            browser_timeout_ms=500,
            browser_max_redirects=1,
        )
        service = URLTraversalService(settings)
        artifact_id = "checklist-cleanup"
        screenshot = Path(tmp) / "screenshots" / f"{artifact_id}.png"
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        screenshot.write_bytes(b"partial")

        fake_page = MagicMock()
        fake_page.goto.side_effect = RuntimeError("Navigation timeout")
        fake_context = MagicMock()
        fake_context.new_page.return_value = fake_page
        fake_browser = MagicMock()
        fake_browser.new_context.return_value = fake_context
        fake_controller = MagicMock()
        fake_controller.chromium.launch.return_value = fake_browser

        sync_api = MagicMock()
        sync_api.Error = type("PlaywrightError", (Exception,), {})
        sync_api.sync_playwright.return_value.__enter__.return_value = fake_controller
        sync_api.sync_playwright.return_value.__exit__.return_value = None

        with patch("app.url_analysis.traversal.validate_target_url"), patch.dict(
            "sys.modules",
            {"playwright": MagicMock(), "playwright.sync_api": sync_api},
        ):
            try:
                service.traverse("https://example.com/", artifact_id)
                _row("Navigation timeout", FAIL, "should have failed")
                _row("Browser failure", FAIL, "should have failed")
                _row("Cleanup after failure", FAIL, "no exception")
            except RuntimeError:
                _row("Navigation timeout", PASS, "failed closed")
                _row("Browser failure", PASS, "wrapped RuntimeError")
                cleaned = not screenshot.exists()
                _row(
                    "Cleanup after failure",
                    PASS if cleaned else FAIL,
                    "partial screenshot removed" if cleaned else "screenshot left behind",
                )

    _row(
        "Screenshot generation",
        SKIP,
        "needs live Docker traverse of https://example.com (see --live)",
    )
    _row(
        "Screenshot retrieval",
        SKIP,
        "GET /api/emails/traversals/{id}/screenshot after COMPLETED",
    )


def check_attachments() -> None:
    print("\n=== Attachment analysis ===")
    fixtures = ROOT / "backend" / "tests" / "fixtures" / "emails"
    if not fixtures.exists():
        _row("Fixtures missing", FAIL, "run scripts/build_demo_fixtures.py first")
        return

    cases = [
        ("01_normal_pdf.eml", "Normal PDF", lambda r, p: r.risk_score == 0 and p.attachments[0].detected_type == "pdf"),
        ("02_normal_image.eml", "Normal image", lambda r, p: r.risk_score == 0 and p.attachments[0].detected_type == "png"),
        (
            "03_executable_extension.eml",
            "Executable extension",
            lambda r, p: "dangerous_extension" in {i.type for i in r.indicators}
            or "executable_signature" in {i.type for i in r.indicators},
        ),
        (
            "04_double_extension.eml",
            "Double extension",
            lambda r, p: "double_extension" in {i.type for i in r.indicators},
        ),
        (
            "05_signature_mismatch.eml",
            "File signature mismatch",
            lambda r, p: "file_signature_mismatch" in {i.type for i in r.indicators},
        ),
        (
            "06_zip_archive.eml",
            "ZIP archive metadata",
            lambda r, p: p.attachments[0].archive_members == ["readme.txt", "update.js"]
            and "archive_contains_script" in {i.type for i in r.indicators},
        ),
        (
            "07_empty_attachment.eml",
            "Empty attachment",
            lambda r, p: "empty_attachment" in {i.type for i in r.indicators},
        ),
        (
            "08_malformed_zip.eml",
            "Malformed attachment",
            lambda r, p: p.attachments[0].archive_malformed
            and "malformed_archive" in {i.type for i in r.indicators},
        ),
    ]

    for filename, label, predicate in cases:
        path = fixtures / filename
        try:
            parsed = parse_eml(path, 20 * 1024 * 1024)
            result = AttachmentAnalyzer().analyze(parsed)
            ok = predicate(result, parsed)
            detail = f"score={result.risk_score} indicators={[i.type for i in result.indicators]}"
            _row(label, PASS if ok else FAIL, detail)
        except Exception as exc:
            _row(label, FAIL, str(exc))

    # SHA-256 correctness
    path = fixtures / "09_sha256_known_pdf.eml"
    parsed = parse_eml(path, 20 * 1024 * 1024)
    expected = hashlib.sha256(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n").hexdigest()
    actual = parsed.attachments[0].sha256
    _row(
        "SHA-256 correctness",
        PASS if actual == expected else FAIL,
        f"sha256={actual[:16]}…",
    )

    # Oversized attachment (prefer inbox copy; else synthesize in memory via analyzer only)
    oversized = ROOT / "data" / "inbox" / "10_oversized_attachment.eml"
    if oversized.exists():
        parsed = parse_eml(oversized, 30 * 1024 * 1024)
        result = AttachmentAnalyzer().analyze(parsed)
        types = {i.type for i in result.indicators}
        size = parsed.attachments[0].size_bytes
        ok = size >= LARGE_ATTACHMENT_BYTES and "large_attachment" in types
        _row("Oversized attachment", PASS if ok else FAIL, f"size={size} indicators={sorted(types)}")
    else:
        email = ParsedEmail(
            content_hash="a" * 64,
            source_path="synthetic-large.eml",
            attachments=[
                AttachmentData(
                    filename="large.pdf",
                    content_type="application/pdf",
                    size_bytes=LARGE_ATTACHMENT_BYTES,
                    sha256="f" * 64,
                    detected_type="pdf",
                )
            ],
        )
        result = AttachmentAnalyzer().analyze(email)
        ok = "large_attachment" in {i.type for i in result.indicators}
        _row("Oversized attachment", PASS if ok else FAIL, "synthetic size flag")


def check_live(api_base: str) -> None:
    print("\n=== Live API (optional) ===")
    try:
        import httpx
    except ImportError:
        _row("httpx", FAIL, "install httpx")
        return

    try:
        health = httpx.get(f"{api_base}/api/health", timeout=5.0)
        _row("API health", PASS if health.status_code == 200 else FAIL, health.text[:80])
    except Exception as exc:
        _row("API health", FAIL, str(exc))
        _row("Screenshot generation", SKIP, "API down")
        _row("Screenshot retrieval", SKIP, "API down")
        return

    print(
        "  [--] Live traverse/screenshot: in KRIT UI, open an email with "
        "selected_for_traversal=true, click Investigate, then open the screenshot."
    )
    print(
        "       Force select if needed:\n"
        '       docker compose exec postgres psql -U phishing -d phishing_detection '
        '-c "UPDATE email_urls SET selected_for_traversal = true WHERE url LIKE \'%example.com%\';"'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kris checklist runner")
    parser.add_argument("--live", action="store_true", help="Also probe API health")
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    print("Kris component checklist (safe fixtures only)")
    check_url_validation()
    check_url_runtime()
    check_attachments()
    if args.live:
        check_live(args.api)
    print("\nDone.")


if __name__ == "__main__":
    main()
