from __future__ import annotations

from typing import Any

import httpx


class VisualWorkerClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def predict(self, traversal_id: str, screenshot_path: str, url: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/predict",
            json={
                "traversal_id": traversal_id,
                "screenshot_path": screenshot_path,
                "url": url,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
