from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.url_analysis.traversal import (
    URLTraversalService,
    _redirect_hop_count,
    is_blocked_ip,
    resolve_host_addresses,
    validate_target_url,
)
import socket


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://192.168.1.20/",
        "http://[::1]/",
        "http://169.254.169.254/",
        "http://metadata.google.internal/",
        "http://user:password@example.com/",
        "file:///tmp/example.html",
        "javascript:alert(1)",
    ],
)
def test_validate_target_url_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(ValueError):
        validate_target_url(url)


def test_validate_target_url_accepts_public_http_url() -> None:
    with patch(
        "app.url_analysis.traversal.resolve_host_addresses",
        return_value=["8.8.8.8"],
    ):
        validate_target_url("https://example.com/login")


def test_validate_target_url_rejects_hostname_resolving_to_private_ip() -> None:
    with patch(
        "app.url_analysis.traversal.resolve_host_addresses",
        return_value=["127.0.0.1"],
    ):
        with pytest.raises(ValueError, match="Private"):
            validate_target_url("http://rebind.example/")


def test_validate_target_url_rejects_dns_failure() -> None:
    with patch(
        "app.url_analysis.traversal.resolve_host_addresses",
        side_effect=ValueError("DNS resolution failed for host: missing.example"),
    ):
        with pytest.raises(ValueError, match="DNS resolution failed"):
            validate_target_url("http://missing.example/")


def test_resolve_host_addresses_uses_getaddrinfo() -> None:
    fake = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.10", 0))]
    with patch("app.url_analysis.traversal.socket.getaddrinfo", return_value=fake):
        assert resolve_host_addresses("example.com") == ["203.0.113.10"]


def test_is_blocked_ip_covers_link_local_and_unspecified() -> None:
    assert is_blocked_ip("169.254.169.254")
    assert is_blocked_ip("0.0.0.0")
    assert is_blocked_ip("203.0.113.50")  # TEST-NET / reserved
    assert not is_blocked_ip("8.8.8.8")


def test_redirect_hop_count() -> None:
    grandparent = MagicMock()
    grandparent.redirected_from = None
    parent = MagicMock()
    parent.redirected_from = grandparent
    request = MagicMock()
    request.redirected_from = parent
    assert _redirect_hop_count(request) == 2


def _install_fake_playwright(fake_browser_controller: MagicMock):
    """Stub playwright modules so tests do not require the browser extra."""
    sync_api = MagicMock()
    sync_api.Error = type("PlaywrightError", (Exception,), {})
    sync_api.sync_playwright.return_value.__enter__.return_value = fake_browser_controller
    sync_api.sync_playwright.return_value.__exit__.return_value = None
    return {
        "playwright": MagicMock(),
        "playwright.sync_api": sync_api,
    }


def test_traverse_cleans_up_screenshot_after_failure(tmp_path: Path) -> None:
    settings = Settings(artifact_directory=tmp_path, browser_timeout_ms=1000, browser_max_redirects=2)
    service = URLTraversalService(settings)
    artifact_id = "cleanup-test"
    screenshot_file = tmp_path / "screenshots" / f"{artifact_id}.png"

    fake_page = MagicMock()
    fake_page.goto.side_effect = RuntimeError("Navigation timeout")
    fake_context = MagicMock()
    fake_context.new_page.return_value = fake_page
    fake_browser = MagicMock()
    fake_browser.new_context.return_value = fake_context
    fake_controller = MagicMock()
    fake_controller.chromium.launch.return_value = fake_browser

    (tmp_path / "screenshots").mkdir(parents=True, exist_ok=True)
    screenshot_file.write_bytes(b"partial")

    with patch("app.url_analysis.traversal.validate_target_url"), patch.dict(
        "sys.modules", _install_fake_playwright(fake_controller)
    ):
        with pytest.raises(RuntimeError, match="URL traversal failed"):
            service.traverse("https://example.com/", artifact_id)

    assert not screenshot_file.exists()


def test_guard_request_blocks_redirect_over_limit(tmp_path: Path) -> None:
    settings = Settings(artifact_directory=tmp_path, browser_max_redirects=1)
    service = URLTraversalService(settings)
    captured: dict = {}

    fake_page = MagicMock()
    fake_page.route.side_effect = lambda _pattern, handler: captured.update(guard=handler)
    fake_page.goto.side_effect = RuntimeError("net::ERR_FAILED")
    fake_context = MagicMock()
    fake_context.new_page.return_value = fake_page
    fake_browser = MagicMock()
    fake_browser.new_context.return_value = fake_context
    fake_controller = MagicMock()
    fake_controller.chromium.launch.return_value = fake_browser

    with patch("app.url_analysis.traversal.validate_target_url"), patch.dict(
        "sys.modules", _install_fake_playwright(fake_controller)
    ):
        with pytest.raises(RuntimeError):
            service.traverse("https://example.com/", "redirect-limit")

    grandparent = MagicMock()
    grandparent.redirected_from = None
    parent = MagicMock()
    parent.url = "https://example.com/hop1"
    parent.redirected_from = grandparent

    fake_route = MagicMock()
    fake_request = MagicMock()
    fake_request.url = "https://example.com/hop2"
    fake_request.is_navigation_request.return_value = True
    fake_request.redirected_from = parent
    fake_route.request = fake_request

    with patch("app.url_analysis.traversal.validate_target_url"):
        captured["guard"](fake_route)

    fake_route.abort.assert_called_once_with("blockedbyclient")
    fake_route.continue_.assert_not_called()
