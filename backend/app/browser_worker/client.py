from __future__ import annotations

from typing import Any

import httpx


class BrowserWorkerClient:
    def __init__(self, base_url: str, timeout_seconds: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def traverse(self, requested_url: str, traversal_id: str) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/traverse",
            json={"requested_url": requested_url, "traversal_id": traversal_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
