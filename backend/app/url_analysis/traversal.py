from __future__ import annotations

import hashlib
import ipaddress
import socket
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings

BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "metadata.google.internal"}


@dataclass
class TraversalOutput:
    requested_url: str
    final_url: str | None = None
    http_status: int | None = None
    page_title: str | None = None
    redirect_chain: list[dict] = field(default_factory=list)
    blocked_requests: list[str] = field(default_factory=list)
    screenshot_path: str | None = None
    screenshot_sha256: str | None = None


def is_blocked_ip(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    return (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
    )


def resolve_host_addresses(hostname: str) -> list[str]:
    try:
        results = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for host: {hostname}") from exc

    addresses: list[str] = []
    seen: set[str] = set()
    for result in results:
        address = result[4][0]
        if address not in seen:
            seen.add(address)
            addresses.append(address)
    if not addresses:
        raise ValueError(f"DNS resolution returned no addresses for host: {hostname}")
    return addresses


def is_private_host(hostname: str, *, resolve_dns: bool = True) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized in BLOCKED_HOSTNAMES or normalized.endswith(".local"):
        return True
    try:
        return is_blocked_ip(str(ipaddress.ip_address(normalized)))
    except ValueError:
        pass
    if not resolve_dns:
        return False
    return any(is_blocked_ip(address) for address in resolve_host_addresses(normalized))


def validate_target_url(url: str, *, resolve_dns: bool = True) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP(S) URLs can be traversed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if is_private_host(parsed.hostname, resolve_dns=resolve_dns):
        raise ValueError("Private, loopback, link-local, or reserved hosts are blocked")


def _redirect_hop_count(request, *, safety_limit: int = 100) -> int:
    hops = 0
    current = request.redirected_from
    while current is not None and hops <= safety_limit:
        hops += 1
        current = current.redirected_from
    return hops


def _cleanup_screenshot(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class URLTraversalService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def traverse(self, url: str, artifact_id: str) -> TraversalOutput:
        validate_target_url(url)
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is required in the browser traversal container") from exc

        output = TraversalOutput(requested_url=url)
        screenshot_directory = self.settings.artifact_directory / "screenshots"
        screenshot_directory.mkdir(parents=True, exist_ok=True)
        screenshot_file = screenshot_directory / f"{artifact_id}.png"
        redirect_limit_exceeded = False

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            context = browser.new_context(
                accept_downloads=False,
                service_workers="block",
                viewport={
                    "width": self.settings.browser_viewport_width,
                    "height": self.settings.browser_viewport_height,
                },
            )
            page = context.new_page()

            def guard_request(route) -> None:
                nonlocal redirect_limit_exceeded
                request = route.request
                request_url = request.url
                try:
                    validate_target_url(request_url)
                except ValueError:
                    output.blocked_requests.append(request_url)
                    route.abort("blockedbyclient")
                    return

                if request.is_navigation_request():
                    hops = _redirect_hop_count(request)
                    if hops > self.settings.browser_max_redirects:
                        redirect_limit_exceeded = True
                        output.blocked_requests.append(request_url)
                        route.abort("blockedbyclient")
                        return
                    if request.redirected_from is not None:
                        output.redirect_chain.append(
                            {"from": request.redirected_from.url, "to": request_url}
                        )

                route.continue_()

            page.route("**/*", guard_request)
            try:
                response = page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.settings.browser_timeout_ms,
                )
                if redirect_limit_exceeded or len(output.redirect_chain) > self.settings.browser_max_redirects:
                    raise RuntimeError("Redirect limit exceeded")
                output.final_url = page.url
                output.http_status = response.status if response else None
                output.page_title = page.title()
                page.screenshot(path=str(screenshot_file), full_page=True)
                output.screenshot_path = str(screenshot_file.relative_to(self.settings.artifact_directory))
                output.screenshot_sha256 = hashlib.sha256(screenshot_file.read_bytes()).hexdigest()
            except (PlaywrightError, OSError, RuntimeError, ValueError) as exc:
                _cleanup_screenshot(screenshot_file)
                output.screenshot_path = None
                output.screenshot_sha256 = None
                raise RuntimeError(f"URL traversal failed: {exc}") from exc
            finally:
                context.close()
                browser.close()
        return output
