"""Small streaming HTTP abstraction used by the resumable installer."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import BinaryIO, Protocol


class TransportError(RuntimeError):
    """A release asset could not be fetched."""


@dataclass(slots=True)
class Response:
    status: int
    headers: Mapping[str, str]
    body: BinaryIO

    def close(self) -> None:
        self.body.close()

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class Transport(Protocol):
    def open(self, url: str, offset: int = 0) -> Response:
        """Open *url*, requesting bytes beginning at *offset*."""


class UrllibTransport:
    """HTTP(S) transport with explicit Range requests and no hidden writes."""

    def __init__(self, *, timeout: float = 60.0, user_agent: str = "lexicon-data/1.0") -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def open(self, url: str, offset: int = 0) -> Response:
        if not url.startswith(("https://", "http://")):
            raise TransportError("release URLs must use HTTP(S)")
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": self.user_agent,
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            opened = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise TransportError(f"HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"unable to fetch {url}: {exc.reason}") from exc
        return Response(
            status=int(getattr(opened, "status", opened.getcode())),
            headers={key.lower(): value for key, value in opened.headers.items()},
            body=opened,
        )


def read_limited(response: Response, limit: int) -> bytes:
    """Read a small response while rejecting unexpectedly large payloads."""

    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = response.body.read(min(64 * 1024, limit + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise TransportError(f"response exceeds {limit} bytes")
