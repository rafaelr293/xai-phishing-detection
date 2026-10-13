from __future__ import annotations

import hashlib
import ipaddress
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


def validate_target_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP(S) URLs can be traversed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    if is_private_host(parsed.hostname):
        raise ValueError("Private, loopback, link-local, or reserved hosts are blocked")


def is_private_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized in BLOCKED_HOSTNAMES or normalized.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_multicast


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
                request_url = route.request.url
                try:
                    validate_target_url(request_url)
                except ValueError:
                    output.blocked_requests.append(request_url)
                    route.abort("blockedbyclient")
                    return
                route.continue_()

            def record_redirect(request) -> None:
                if request.is_navigation_request() and request.redirected_from:
                    output.redirect_chain.append({"from": request.redirected_from.url, "to": request.url})

            page.route("**/*", guard_request)
            page.on("request", record_redirect)
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=self.settings.browser_timeout_ms)
                if len(output.redirect_chain) > self.settings.browser_max_redirects:
                    raise RuntimeError("Redirect limit exceeded")
                output.final_url = page.url
                output.http_status = response.status if response else None
                output.page_title = page.title()
                page.screenshot(path=str(screenshot_file), full_page=True)
                output.screenshot_path = str(screenshot_file.relative_to(self.settings.artifact_directory))
                output.screenshot_sha256 = hashlib.sha256(screenshot_file.read_bytes()).hexdigest()
            except (PlaywrightError, OSError) as exc:
                raise RuntimeError(f"URL traversal failed: {exc}") from exc
            finally:
                context.close()
                browser.close()
        return output
