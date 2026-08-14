from __future__ import annotations

import socket

import pytest

from lexicon_mcp.runtime.offline import NetworkDisabledError, deny_network


def test_network_denial_guard_blocks_connection_apis_and_restores_them() -> None:
    original = socket.create_connection
    with deny_network():
        with pytest.raises(NetworkDisabledError):
            socket.create_connection(("example.invalid", 443))
        with pytest.raises(NetworkDisabledError):
            socket.getaddrinfo("example.invalid", 443)
    assert socket.create_connection is original
