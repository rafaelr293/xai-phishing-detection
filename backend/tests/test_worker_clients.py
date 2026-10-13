from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.browser_worker.client import BrowserWorkerClient
from app.visual_worker_client import VisualWorkerClient


def test_browser_worker_client_posts_traversal_payload() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "requested_url": "https://example.com/",
        "final_url": "https://example.com/",
        "http_status": 200,
        "page_title": "Example",
        "redirect_chain": [],
        "blocked_requests": [],
        "screenshot_path": "screenshots/abc.png",
        "screenshot_sha256": "a" * 64,
    }

    with patch("app.browser_worker.client.httpx.post", return_value=response) as post:
        result = BrowserWorkerClient("http://browser-worker:8010").traverse(
            "https://example.com/",
            "11111111-1111-1111-1111-111111111111",
        )

    post.assert_called_once()
    assert post.call_args.args[0] == "http://browser-worker:8010/traverse"
    assert result["screenshot_path"] == "screenshots/abc.png"


def test_visual_worker_client_posts_predict_payload() -> None:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "detector": "visual_analysis",
        "status": "COMPLETED",
        "risk_score": 40,
        "confidence": 0.8,
        "indicators": [],
        "evidence": {"screenshot_count": 1},
    }

    with patch("app.visual_worker_client.httpx.post", return_value=response) as post:
        result = VisualWorkerClient("http://visual-worker:8010").predict(
            "11111111-1111-1111-1111-111111111111",
            "screenshots/abc.png",
            "https://example.com/",
        )

    assert post.call_args.kwargs["json"]["screenshot_path"] == "screenshots/abc.png"
    assert result["status"] == "COMPLETED"


def test_browser_worker_client_raises_on_http_error() -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "blocked",
        request=MagicMock(),
        response=MagicMock(status_code=422),
    )

    with patch("app.browser_worker.client.httpx.post", return_value=response):
        with pytest.raises(httpx.HTTPStatusError):
            BrowserWorkerClient("http://browser-worker:8010").traverse(
                "http://127.0.0.1/",
                "11111111-1111-1111-1111-111111111111",
            )
