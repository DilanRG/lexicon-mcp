"""Process-local network denial used by offline acceptance and semantic workers."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch


class NetworkDisabledError(RuntimeError):
    """Raised when offline runtime code attempts network access."""


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise NetworkDisabledError("network access is disabled for the offline lexicon runtime")


def install_network_guard() -> None:
    """Permanently deny new network connections in the current worker process."""

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked


@contextmanager
def deny_network() -> Iterator[None]:
    """Temporarily prove that a block of runtime calls remains fully offline."""

    with (
        patch.object(socket.socket, "connect", _blocked),
        patch.object(socket.socket, "connect_ex", _blocked),
        patch.object(socket, "create_connection", _blocked),
        patch.object(socket, "getaddrinfo", _blocked),
    ):
        yield
