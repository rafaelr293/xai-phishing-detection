import pytest

from app.url_analysis.traversal import validate_target_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://192.168.1.20/",
        "http://[::1]/",
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
    validate_target_url("https://example.com/login")
