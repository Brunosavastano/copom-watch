from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a public data source cannot be fetched and no cache is available."""


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: int = 30,
        retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_json(self, url: str, cache_name: str, params: dict[str, Any] | None = None) -> Any:
        cache_path = self.cache_dir / f"{cache_name}.json"
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return data
            except Exception as exc:  # noqa: BLE001 - source errors are logged and retried.
                last_error = exc
                LOGGER.warning("Fetch failed (%s/%s) for %s: %s", attempt, self.retries, url, exc)
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        if cache_path.exists():
            LOGGER.warning("Using cached response for %s after fetch failure.", url)
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise FetchError(f"Could not fetch {url} and no cache exists at {cache_path}") from last_error
