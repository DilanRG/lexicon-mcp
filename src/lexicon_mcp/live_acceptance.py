"""Auditable ten-cycle acceptance for the Windows MCPO/Open WebUI stack.

The command-line adapter requires an explicit live-execution flag.  The orchestration
is dependency-injected so ordinary tests and fixture replays cannot stop local services.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar, Protocol, cast

import httpx

from .data.manifest import parse_manifest

RESTART_CYCLES = 10
EXPECTED_LEXICON_TOOLS = frozenset(
    {
        "dictionary_lookup",
        "dictionary_synonyms",
        "dictionary_translate",
        "dictionary_relations",
        "dictionary_semantic_neighbors",
        "dictionary_wordplay",
    }
)
ADMIN_TERMS = frozenset({"install", "status", "verify", "repair", "rollback"})
HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class AcceptanceFailure(RuntimeError):
    """A live acceptance invariant was not satisfied."""


class HostRequestError(RuntimeError):
    """An HTTP request could not reach its target."""


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    root: Path = Path(r"E:\AI")
    project: Path = Path(r"E:\AI\lexicon-mcp")
    data_root: Path = Path(r"E:\AI\data\lexicon-mcp")
    project_venv: Path = Path(r"E:\AI\lexicon-mcp\.venv")
    start_script: Path = Path(r"E:\AI\scripts\start.ps1")
    stop_script: Path = Path(r"E:\AI\scripts\stop.ps1")
    mcpo_pid_file: Path = Path(r"E:\AI\run\mcpo.pid")
    router_health_url: str = "http://127.0.0.1:8080/health"
    router_models_url: str = "http://127.0.0.1:8080/v1/models"
    webui_health_url: str = "http://127.0.0.1:18000/health"
    mcpo_openapi_url: str = "http://127.0.0.1:18010/openapi.json"
    lexicon_base_url: str = "http://127.0.0.1:18010/lexicon"
    calculator_base_url: str = "http://127.0.0.1:18010/calculator"
    startup_timeout_seconds: float = 240.0
    shutdown_timeout_seconds: float = 90.0
    request_timeout_seconds: float = 15.0
    poll_interval_seconds: float = 0.5
    mcpo_command_marker: str = "mcpo"
    required_child_markers: tuple[str, ...] = ("lexicon-mcp", "calculator.py")

    @property
    def lexicon_openapi_url(self) -> str:
        return f"{self.lexicon_base_url.rstrip('/')}/openapi.json"

    @property
    def calculator_openapi_url(self) -> str:
        return f"{self.calculator_base_url.rstrip('/')}/openapi.json"

    @property
    def lookup_url(self) -> str:
        return f"{self.lexicon_base_url.rstrip('/')}/dictionary_lookup"

    def lexicon_tool_url(self, name: str) -> str:
        if name not in EXPECTED_LEXICON_TOOLS:
            raise ValueError(f"unknown Lexicon tool: {name}")
        return f"{self.lexicon_base_url.rstrip('/')}/{name}"

    @property
    def calculator_url(self) -> str:
        return f"{self.calculator_base_url.rstrip('/')}/calculate"


@dataclass(frozen=True, slots=True)
class Installation:
    version: str
    path: Path
    manifest_sha256: str
    transformation_commit: str
    manifest_size: int


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    pid: int
    parent_pid: int
    creation_time: str
    name: str
    executable: str | None
    command_line: str | None

    @property
    def identity(self) -> str:
        return f"{self.pid}:{self.creation_time}"

    def evidence(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "parent_pid": self.parent_pid,
            "creation_time": self.creation_time,
            "identity": self.identity,
            "name": self.name,
            "executable": self.executable,
            "command_line": self.command_line,
        }


@dataclass(frozen=True, slots=True)
class ProcessTree:
    root: ProcessRecord
    descendants: tuple[ProcessRecord, ...]
    marker_pids: Mapping[str, tuple[int, ...]]

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({self.root.pid, *(item.pid for item in self.descendants)})

    def evidence(self) -> dict[str, Any]:
        return {
            "root": self.root.evidence(),
            "descendants": [item.evidence() for item in self.descendants],
            "required_child_markers": {
                key: list(value) for key, value in sorted(self.marker_pids.items())
            },
        }


@dataclass(frozen=True, slots=True)
class FileRecord:
    relative_path: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    root: Path
    file_count: int
    total_bytes: int
    metadata_sha256: str
    content_sha256: str | None
    records: tuple[FileRecord, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "metadata_sha256": self.metadata_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class DirectoryChange:
    root: str
    action: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class HttpObservation:
    status_code: int
    body: Any
    body_sha256: str


@dataclass(frozen=True, slots=True)
class CommandObservation:
    script: str
    exit_code: int
    duration_seconds: float
    stdout_sha256: str
    stderr_sha256: str
    stdout_tail: str
    stderr_tail: str


class ChangeMonitor(Protocol):
    def changes(self) -> tuple[DirectoryChange, ...]: ...

    def errors(self) -> tuple[str, ...]: ...

    def close(self) -> None: ...


class AcceptanceHost(Protocol):
    is_live: bool

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def installation(self, data_root: Path) -> Installation: ...

    def inventory(self, root: Path, *, content: bool) -> InventorySnapshot: ...

    def begin_change_monitor(self, roots: Sequence[Path]) -> ChangeMonitor: ...

    def process_table(self) -> tuple[ProcessRecord, ...]: ...

    def read_pid(self, path: Path) -> int: ...

    def run_script(self, path: Path, *, timeout_seconds: float) -> CommandObservation: ...

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float,
    ) -> HttpObservation: ...

    def exclusive_open(self, paths: Sequence[Path]) -> int: ...


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _display_tail(value: bytes, limit: int = 4_000) -> str:
    return value[-limit:].decode("utf-8", errors="replace")


class EvidenceWriter:
    """Append-only, flushed JSONL evidence."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self._stream = path.open("xb")
        self._sequence = 0
        self._closed = False

    def emit(
        self,
        event: str,
        *,
        status: str = "ok",
        cycle: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("evidence writer is closed")
        self._sequence += 1
        value: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "sequence": self._sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "status": status,
        }
        if cycle is not None:
            value["cycle"] = cycle
        if details is not None:
            value["details"] = dict(details)
        line = _canonical_json(value) + b"\n"
        self._stream.write(line)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        print(f"[{self._sequence:03d}] {status.upper()} {event}", flush=True)

    def close(self) -> str:
        if not self._closed:
            self._stream.close()
            self._closed = True
        return _sha256_file(self.path)


def _inventory_records(root: Path, *, content: bool) -> InventorySnapshot:
    if not root.is_dir():
        raise AcceptanceFailure(f"inventory root is not a directory: {root}")
    records: list[FileRecord] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort(key=str.casefold)
        file_names.sort(key=str.casefold)
        directory_path = Path(directory)
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                raise AcceptanceFailure(f"inventory does not permit symbolic links: {path}")
            stat = path.stat(follow_symlinks=False)
            relative = path.relative_to(root).as_posix()
            records.append(
                FileRecord(
                    relative_path=relative,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    device=stat.st_dev,
                    inode=stat.st_ino,
                    sha256=_sha256_file(path) if content else None,
                )
            )
    records.sort(key=lambda item: item.relative_path.casefold())
    metadata_digest = hashlib.sha256()
    content_digest = hashlib.sha256() if content else None
    for item in records:
        metadata_digest.update(
            _canonical_json(
                {
                    "path": item.relative_path,
                    "size": item.size,
                    "mtime_ns": item.mtime_ns,
                    "device": item.device,
                    "inode": item.inode,
                }
            )
        )
        metadata_digest.update(b"\n")
        if content_digest is not None:
            content_digest.update(
                _canonical_json(
                    {"path": item.relative_path, "size": item.size, "sha256": item.sha256}
                )
            )
            content_digest.update(b"\n")
    return InventorySnapshot(
        root=root,
        file_count=len(records),
        total_bytes=sum(item.size for item in records),
        metadata_sha256=metadata_digest.hexdigest(),
        content_sha256=content_digest.hexdigest() if content_digest is not None else None,
        records=tuple(records),
    )


def inventory_diff(
    before: InventorySnapshot, after: InventorySnapshot, *, limit: int = 100
) -> dict[str, list[str]]:
    before_items = {item.relative_path: item for item in before.records}
    after_items = {item.relative_path: item for item in after.records}
    added = sorted(set(after_items) - set(before_items), key=str.casefold)
    removed = sorted(set(before_items) - set(after_items), key=str.casefold)
    changed = sorted(
        (
            path
            for path in set(before_items).intersection(after_items)
            if before_items[path] != after_items[path]
        ),
        key=str.casefold,
    )
    return {
        "added": added[:limit],
        "removed": removed[:limit],
        "changed": changed[:limit],
    }


def _descendants(root_pid: int, processes: Sequence[ProcessRecord]) -> tuple[ProcessRecord, ...]:
    by_parent: dict[int, list[ProcessRecord]] = {}
    for process in processes:
        by_parent.setdefault(process.parent_pid, []).append(process)
    found: list[ProcessRecord] = []
    pending = [root_pid]
    seen = {root_pid}
    while pending:
        parent = pending.pop()
        for child in by_parent.get(parent, []):
            if child.pid in seen:
                continue
            seen.add(child.pid)
            found.append(child)
            pending.append(child.pid)
    return tuple(sorted(found, key=lambda item: item.pid))


def _process_has_marker(process: ProcessRecord, marker: str) -> bool:
    normalized = marker.casefold()
    executable_names = {normalized, f"{normalized}.exe"}
    for value in (process.name, process.executable):
        if value and PureWindowsPath(value).name.casefold() in executable_names:
            return True
    command_line = process.command_line
    if not command_line:
        return False
    if any(character.isspace() for character in marker):
        return normalized in command_line.casefold()
    try:
        tokens = shlex.split(command_line, posix=False)
    except ValueError:
        tokens = command_line.split()
    marker_has_extension = "." in PureWindowsPath(marker).name
    for token in tokens:
        stripped = token.strip("\"'")
        if stripped.casefold() == normalized:
            return True
        name = PureWindowsPath(stripped).name.casefold()
        if name == f"{normalized}.exe" or (marker_has_extension and name == normalized):
            return True
    return False


def capture_mcpo_tree(host: AcceptanceHost, config: AcceptanceConfig) -> ProcessTree:
    pid = host.read_pid(config.mcpo_pid_file)
    processes = host.process_table()
    by_pid = {item.pid: item for item in processes}
    root = by_pid.get(pid)
    if root is None:
        raise AcceptanceFailure(f"MCPO PID file points to absent process {pid}")
    if not _process_has_marker(root, config.mcpo_command_marker):
        raise AcceptanceFailure(
            f"MCPO root process {pid} does not contain marker {config.mcpo_command_marker!r}"
        )
    descendants = _descendants(pid, processes)
    marker_pids: dict[str, tuple[int, ...]] = {}
    for marker in config.required_child_markers:
        matches = tuple(item.pid for item in descendants if _process_has_marker(item, marker))
        if not matches:
            raise AcceptanceFailure(
                f"MCPO tree has no descendant containing required marker {marker!r}"
            )
        marker_pids[marker] = matches
    return ProcessTree(root, descendants, marker_pids)


def _assert_no_orphan_children(
    processes: Sequence[ProcessRecord],
    config: AcceptanceConfig,
    tree: ProcessTree | None,
) -> None:
    allowed = tree.pids if tree is not None else frozenset()
    orphans: list[dict[str, Any]] = []
    for process in processes:
        markers = [
            marker
            for marker in config.required_child_markers
            if _process_has_marker(process, marker)
        ]
        if markers and process.pid not in allowed:
            orphans.append({"process": process.evidence(), "markers": markers})
    if orphans:
        raise AcceptanceFailure(f"orphan MCP child processes detected: {orphans!r}")


def openapi_operations(schema: Any) -> frozenset[tuple[str, str]]:
    if not isinstance(schema, dict) or not isinstance(schema.get("paths"), dict):
        raise AcceptanceFailure("OpenAPI response has no paths object")
    operations: set[tuple[str, str]] = set()
    for path, path_item in schema["paths"].items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            raise AcceptanceFailure("OpenAPI paths are malformed")
        for method in path_item:
            normalized = str(method).casefold()
            if normalized in HTTP_METHODS:
                operations.add((path, normalized))
    return frozenset(operations)


def validate_lexicon_openapi(schema: Any) -> tuple[str, ...]:
    operations = openapi_operations(schema)
    expected = frozenset((f"/{name}", "post") for name in EXPECTED_LEXICON_TOOLS)
    if operations != expected:
        raise AcceptanceFailure(
            "Lexicon OpenAPI operations differ from the exact six-tool contract: "
            f"expected={sorted(expected)!r}, actual={sorted(operations)!r}"
        )
    serialized = json.dumps(schema, ensure_ascii=False).casefold()
    exposed_admin = sorted(term for term in ADMIN_TERMS if f"/{term}" in serialized)
    if exposed_admin:
        raise AcceptanceFailure(
            f"Lexicon OpenAPI contains administration operation names: {exposed_admin}"
        )
    return tuple(sorted(path.removeprefix("/") for path, _ in operations))


def validate_calculator_openapi(schema: Any) -> None:
    operations = openapi_operations(schema)
    if ("/calculate", "post") not in operations:
        raise AcceptanceFailure("Calculator OpenAPI has no POST /calculate operation")


def _active_models(body: Any) -> list[Any]:
    if isinstance(body, dict) and "active_models" in body:
        value = body["active_models"]
        if not isinstance(value, list):
            raise AcceptanceFailure("router active_models must be an array")
        return value
    entries: Any = body.get("data") if isinstance(body, dict) else body
    if not isinstance(entries, list):
        raise AcceptanceFailure(
            "router models response must expose active_models or a model data array"
        )
    active: list[Any] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AcceptanceFailure(f"router model entry {index} is not an object")
        status = entry.get("status")
        state = status.get("value") if isinstance(status, dict) else status
        if not isinstance(state, str):
            raise AcceptanceFailure(f"router model entry {index} has no status.value")
        if state.casefold() != "unloaded":
            active.append(entry.get("id") or entry.get("name") or f"entry-{index}")
    return active


def _require_status(observation: HttpObservation, label: str) -> None:
    if not 200 <= observation.status_code < 300:
        raise AcceptanceFailure(f"{label} returned HTTP {observation.status_code}")


def _require_webui_health(observation: HttpObservation) -> None:
    _require_status(observation, "Open WebUI health")
    if isinstance(observation.body, dict) and observation.body.get("status") is not True:
        raise AcceptanceFailure("Open WebUI health JSON did not report status=true")


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceFailure(f"{label} must be non-empty text")
    return value


def _require_provenance(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceFailure(f"{label} provenance must be an object")
    _require_text(value.get("source"), f"{label} provenance.source")
    _require_text(value.get("license"), f"{label} provenance.license")
    return cast(dict[str, Any], value)


def _normalized_term(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _structured_tool_response(
    observation: HttpObservation,
    tool: str,
    installation: Installation,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require_status(observation, f"Lexicon {tool}")
    value = observation.body
    if not isinstance(value, dict) or value.get("type") != tool:
        raise AcceptanceFailure(f"{tool} returned the wrong structured type")
    if value.get("dataset_version") != installation.version:
        raise AcceptanceFailure(f"{tool} returned the wrong dataset version")
    if not isinstance(value.get("query"), dict):
        raise AcceptanceFailure(f"{tool} returned no structured query object")
    results = value.get("results")
    count = value.get("count")
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        raise AcceptanceFailure(f"{tool} results must be an array of objects")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(results):
        raise AcceptanceFailure(f"{tool} count does not match its results")
    return cast(dict[str, Any], value), cast(list[dict[str, Any]], results)


def _validate_lookup_sense(value: Mapping[str, Any], *, language: str, label: str) -> str:
    sense_id = _require_text(value.get("sense_id"), f"{label} sense_id")
    if value.get("language") != language or value.get("sense_scope") not in {
        "sense",
        "unsensed",
    }:
        raise AcceptanceFailure(f"{label} has the wrong language or sense scope")
    _require_text(value.get("word"), f"{label} word")
    _require_text(value.get("part_of_speech"), f"{label} part_of_speech")
    _require_text(value.get("gloss"), f"{label} gloss")
    examples = value.get("examples")
    pronunciations = value.get("pronunciations")
    if not isinstance(examples, list) or any(not isinstance(item, str) for item in examples):
        raise AcceptanceFailure(f"{label} examples must be an array of text")
    if not isinstance(pronunciations, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("ipa"), str)
        or not item["ipa"].strip()
        for item in pronunciations
    ):
        raise AcceptanceFailure(f"{label} pronunciations must contain IPA objects")
    if "etymology" not in value or (
        value["etymology"] is not None and not isinstance(value["etymology"], str)
    ):
        raise AcceptanceFailure(f"{label} etymology must be text or null")
    translations = value.get("translations")
    if not isinstance(translations, list) or any(
        not isinstance(item, dict) for item in translations
    ):
        raise AcceptanceFailure(f"{label} translations must be an array of objects")
    truncated_fields = value.get("truncated_fields")
    if (
        not isinstance(truncated_fields, list)
        or any(
            item not in {"examples", "pronunciations", "translations"} for item in truncated_fields
        )
        or len(set(truncated_fields)) != len(truncated_fields)
    ):
        raise AcceptanceFailure(f"{label} truncated_fields is malformed")
    _require_provenance(value.get("provenance"), label)
    return sense_id


def _retry(
    host: AcceptanceHost,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    action: Callable[[], Any],
    label: str,
) -> Any:
    deadline = host.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while True:
        try:
            return action()
        except (AcceptanceFailure, HostRequestError, OSError) as exc:
            last_error = exc
        if host.monotonic() >= deadline:
            raise AcceptanceFailure(f"timed out waiting for {label}: {last_error}") from last_error
        host.sleep(poll_seconds)


def _assert_inventory_unchanged(
    host: AcceptanceHost,
    baseline: InventorySnapshot,
    *,
    content: bool = False,
) -> InventorySnapshot:
    current = host.inventory(baseline.root, content=content)
    expected_digest = baseline.content_sha256 if content else baseline.metadata_sha256
    actual_digest = current.content_sha256 if content else current.metadata_sha256
    if actual_digest != expected_digest:
        raise AcceptanceFailure(
            f"inventory changed under {baseline.root}: "
            f"{json.dumps(inventory_diff(baseline, current), ensure_ascii=False)}"
        )
    return current


def _forbidden_dataset_transients(snapshot: InventorySnapshot) -> tuple[str, ...]:
    forbidden: list[str] = []
    for record in snapshot.records:
        name = record.relative_path.casefold()
        if (
            name.endswith((".partial", ".tmp", "-wal", "-shm", ".wal", ".shm"))
            or ".partial/" in name
            or ".partial\\" in name
        ):
            forbidden.append(record.relative_path)
    return tuple(forbidden)


class LiveAcceptanceRunner:
    def __init__(
        self,
        host: AcceptanceHost,
        config: AcceptanceConfig,
        evidence: EvidenceWriter,
    ) -> None:
        self.host = host
        self.config = config
        self.evidence = evidence
        self.cycles: list[dict[str, Any]] = []
        self.six_tool_acceptance: dict[str, Any] | None = None
        self.failure_recovery: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
    ) -> HttpObservation:
        return self.host.request(
            method,
            url,
            payload=payload,
            timeout_seconds=self.config.request_timeout_seconds,
        )

    def _ready_snapshot(self) -> dict[str, Any]:
        router = self._request("GET", self.config.router_health_url)
        _require_status(router, "router health")
        webui = self._request("GET", self.config.webui_health_url)
        _require_webui_health(webui)
        mcpo = self._request("GET", self.config.mcpo_openapi_url)
        _require_status(mcpo, "MCPO OpenAPI")
        if not isinstance(mcpo.body, dict):
            raise AcceptanceFailure("MCPO OpenAPI response is not an object")
        lexicon = self._request("GET", self.config.lexicon_openapi_url)
        _require_status(lexicon, "Lexicon OpenAPI")
        tools = validate_lexicon_openapi(lexicon.body)
        calculator = self._request("GET", self.config.calculator_openapi_url)
        _require_status(calculator, "Calculator OpenAPI")
        validate_calculator_openapi(calculator.body)
        models = self._request("GET", self.config.router_models_url)
        _require_status(models, "router models")
        active_models = _active_models(models.body)
        if active_models:
            raise AcceptanceFailure(f"router has active models: {active_models!r}")
        return {
            "router_health_sha256": router.body_sha256,
            "webui_health_sha256": webui.body_sha256,
            "mcpo_openapi_sha256": mcpo.body_sha256,
            "lexicon_openapi_sha256": lexicon.body_sha256,
            "calculator_openapi_sha256": calculator.body_sha256,
            "lexicon_operations": list(tools),
            "active_models": active_models,
        }

    def _wait_ready(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _retry(
                self.host,
                timeout_seconds=self.config.startup_timeout_seconds,
                poll_seconds=self.config.poll_interval_seconds,
                action=self._ready_snapshot,
                label="router, Open WebUI, MCPO, Lexicon, and Calculator readiness",
            ),
        )

    def _wait_down(self) -> None:
        urls = (
            self.config.router_health_url,
            self.config.webui_health_url,
            self.config.mcpo_openapi_url,
            self.config.lexicon_openapi_url,
        )

        def down() -> bool:
            still_up: list[str] = []
            for url in urls:
                try:
                    observation = self._request("GET", url)
                except HostRequestError:
                    continue
                if 200 <= observation.status_code < 300:
                    still_up.append(url)
            if still_up:
                raise AcceptanceFailure(f"endpoints still reachable after stop: {still_up}")
            return True

        _retry(
            self.host,
            timeout_seconds=self.config.shutdown_timeout_seconds,
            poll_seconds=self.config.poll_interval_seconds,
            action=down,
            label="all stack endpoints to stop",
        )

    def _wait_old_tree_exit(self, tree: ProcessTree) -> None:
        expected_absent = tree.pids

        def exited() -> bool:
            processes = self.host.process_table()
            present = sorted(expected_absent.intersection(item.pid for item in processes))
            late_descendants = [item.pid for item in _descendants(tree.root.pid, processes)]
            remaining = sorted(set(present).union(late_descendants))
            if remaining:
                raise AcceptanceFailure(f"old MCPO process tree still alive: {remaining}")
            return True

        _retry(
            self.host,
            timeout_seconds=self.config.shutdown_timeout_seconds,
            poll_seconds=self.config.poll_interval_seconds,
            action=exited,
            label="old MCPO root and descendants to exit",
        )

    def _invoke_lexicon_tool(
        self,
        tool: str,
        payload: Mapping[str, Any],
        installation: Installation,
    ) -> tuple[HttpObservation, dict[str, Any], list[dict[str, Any]]]:
        observation = self._request("POST", self.config.lexicon_tool_url(tool), payload)
        value, results = _structured_tool_response(observation, tool, installation)
        return observation, value, results

    def _invoke_cycle_tools(self, installation: Installation) -> dict[str, Any]:
        lookup, value, results = self._invoke_lexicon_tool(
            "dictionary_lookup",
            {"word": "cat", "language": "en", "limit": 8},
            installation,
        )
        if not results:
            raise AcceptanceFailure("dictionary_lookup(cat/en) returned no senses")
        if (
            value["query"].get("normalized_word") != "cat"
            or value["query"].get("language") != "en"
            or value["query"].get("limit") != 8
            or value["query"].get("examples_limit") != 8
            or value["query"].get("pronunciations_limit") != 8
            or value["query"].get("translations_limit") != 20
        ):
            raise AcceptanceFailure("dictionary_lookup(cat/en) returned a wrong query scope")
        for index, item in enumerate(results):
            _validate_lookup_sense(item, language="en", label=f"cat sense {index}")

        calculator = self._request("POST", self.config.calculator_url, {"expression": "6 * 7"})
        _require_status(calculator, "Calculator calculate")
        if calculator.body not in (42, 42.0, "42"):
            raise AcceptanceFailure(f"Calculator returned {calculator.body!r} instead of 42")
        return {
            "dictionary_lookup": {
                "response_sha256": lookup.body_sha256,
                "count": len(results),
                "first_sense_id": results[0].get("sense_id"),
                "dataset_version": value.get("dataset_version"),
            },
            "calculator": {
                "response_sha256": calculator.body_sha256,
                "result": calculator.body,
            },
        }

    def _invoke_bank_translation(
        self,
        installation: Installation,
        *,
        sense_id: str,
        expected_gloss: str,
        required_term: str,
    ) -> dict[str, Any]:
        payload = {
            "word": "bank",
            "source_language": "en",
            "target_language": "de",
            "sense_id": sense_id,
            "limit": 20,
            "max_senses": 100,
        }
        observation, value, groups = self._invoke_lexicon_tool(
            "dictionary_translate", payload, installation
        )
        if (
            value["query"].get("sense_id") != sense_id
            or value["query"].get("source_language") != "en"
            or value["query"].get("target_language") != "de"
            or value["query"].get("max_senses") != 100
        ):
            raise AcceptanceFailure("dictionary_translate returned a wrong query scope")
        if not groups:
            raise AcceptanceFailure(
                f"dictionary_translate returned no result for bank sense {sense_id}"
            )
        translated_terms: set[str] = set()
        actual_candidate_count = 0
        for group in groups:
            if (
                group.get("sense_id") != sense_id
                or group.get("sense_scope") != "sense"
                or group.get("source_language") != "en"
                or str(group.get("gloss") or "").strip().casefold() != expected_gloss.casefold()
            ):
                raise AcceptanceFailure(
                    "dictionary_translate crossed or discarded the requested bank sense"
                )
            _require_provenance(group.get("provenance"), "translation group")
            candidates = group.get("translations")
            if (
                not isinstance(candidates, list)
                or not candidates
                or any(not isinstance(item, dict) for item in candidates)
            ):
                raise AcceptanceFailure("dictionary_translate group is malformed")
            for candidate in cast(list[dict[str, Any]], candidates):
                actual_candidate_count += 1
                if (
                    candidate.get("sense_id") != sense_id
                    or candidate.get("sense_scope") != "sense"
                    or candidate.get("language") != "de"
                ):
                    raise AcceptanceFailure(
                        "dictionary_translate returned an unscoped or wrong-language item"
                    )
                term = _require_text(candidate.get("term"), "translation term")
                translated_terms.add(_normalized_term(term))
                _require_provenance(candidate.get("provenance"), f"translation candidate {term}")
        candidate_count = value.get("candidate_count")
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count != actual_candidate_count
            or candidate_count > 20
        ):
            raise AcceptanceFailure("dictionary_translate candidate_count violates its total limit")
        if _normalized_term(required_term) not in translated_terms:
            raise AcceptanceFailure(
                f"dictionary_translate did not map bank sense {sense_id} to {required_term}"
            )
        return {
            "response_sha256": observation.body_sha256,
            "count": len(groups),
            "candidate_count": candidate_count,
            "requested_sense_id": sense_id,
            "expected_gloss": expected_gloss,
            "required_translation": required_term,
            "returned_terms": sorted(translated_terms),
            "source_sense_preserved": True,
        }

    def _invoke_all_lexicon_tools(self, installation: Installation) -> dict[str, Any]:
        lookup, lookup_value, senses = self._invoke_lexicon_tool(
            "dictionary_lookup",
            {
                "word": "bank",
                "language": "en",
                "limit": 100,
                "examples_limit": 8,
                "pronunciations_limit": 8,
                "translations_limit": 3,
            },
            installation,
        )
        if len(senses) < 2:
            raise AcceptanceFailure("dictionary_lookup(bank/en) did not separate senses")
        if (
            lookup_value["query"].get("normalized_word") != "bank"
            or lookup_value["query"].get("language") != "en"
            or lookup_value["query"].get("limit") != 100
            or lookup_value["query"].get("examples_limit") != 8
            or lookup_value["query"].get("pronunciations_limit") != 8
            or lookup_value["query"].get("translations_limit") != 3
        ):
            raise AcceptanceFailure("dictionary_lookup(bank/en) returned a wrong query scope")
        river_senses: list[dict[str, Any]] = []
        financial_senses: list[dict[str, Any]] = []
        embedded_translation_count = 0
        translation_truncation_sense_ids: list[str] = []
        for index, sense in enumerate(senses):
            sense_id = _validate_lookup_sense(sense, language="en", label=f"bank sense {index}")
            if sense.get("sense_scope") != "sense":
                raise AcceptanceFailure("dictionary_lookup(bank/en) returned an unscoped sense")
            translations = sense.get("translations")
            assert isinstance(translations, list)
            truncated_fields = cast(list[str], sense["truncated_fields"])
            if "translations" in truncated_fields:
                translation_truncation_sense_ids.append(sense_id)
            for translation in cast(list[dict[str, Any]], translations):
                embedded_translation_count += 1
                if translation.get("sense_id") != sense_id:
                    raise AcceptanceFailure(
                        "dictionary_lookup(bank/en) attached a translation to the wrong sense"
                    )
                if translation.get("sense_scope") != "sense":
                    raise AcceptanceFailure(
                        "dictionary_lookup(bank/en) returned an unscoped translation"
                    )
                term = _require_text(translation.get("term"), f"bank sense {sense_id} translation")
                _require_text(
                    translation.get("language"),
                    f"bank sense {sense_id} translation language",
                )
                _require_provenance(
                    translation.get("provenance"),
                    f"bank sense {sense_id} translation {term}",
                )
            gloss = str(sense.get("gloss") or "").strip().casefold()
            provenance = cast(dict[str, Any], sense["provenance"])
            if (
                gloss == "edge of river or lake"
                and provenance.get("source") == "Wiktionary via Wiktextract"
                and sense.get("part_of_speech") == "noun"
            ):
                river_senses.append(sense)
            if (
                gloss == "institution"
                and provenance.get("source") == "Wiktionary via Wiktextract"
                and sense.get("part_of_speech") == "noun"
            ):
                financial_senses.append(sense)
        if embedded_translation_count > 3:
            raise AcceptanceFailure("dictionary_lookup exceeded its total translations_limit")
        if not translation_truncation_sense_ids:
            raise AcceptanceFailure(
                "dictionary_lookup did not report expected translation truncation"
            )
        if len(river_senses) != 1:
            raise AcceptanceFailure(
                "dictionary_lookup(bank/en) did not yield one source-scoped river sense"
            )
        if len(financial_senses) != 1:
            raise AcceptanceFailure(
                "dictionary_lookup(bank/en) did not yield one source-scoped financial sense"
            )
        river_sense = river_senses[0]
        river_sense_id = cast(str, river_sense["sense_id"])
        financial_sense = financial_senses[0]
        financial_sense_id = cast(str, financial_sense["sense_id"])
        if financial_sense_id == river_sense_id:
            raise AcceptanceFailure("bank financial and river senses were not distinct")

        synonyms, synonym_value, synonym_groups = self._invoke_lexicon_tool(
            "dictionary_synonyms",
            {
                "word": "important",
                "language": "en",
                "limit": 20,
                "max_senses": 20,
                "unsensed_limit": 5,
            },
            installation,
        )
        if (
            synonym_value["query"].get("normalized_word") != "important"
            or synonym_value["query"].get("language") != "en"
            or synonym_value["query"].get("max_senses") != 20
            or synonym_value["query"].get("unsensed_limit") != 5
        ):
            raise AcceptanceFailure("dictionary_synonyms returned a wrong query scope")
        synonym_terms: set[str] = set()
        actual_synonym_candidate_count = 0
        for group_index, group in enumerate(synonym_groups):
            if group.get("language") != "en" or group.get("sense_scope") not in {
                "sense",
                "unsensed",
            }:
                raise AcceptanceFailure(
                    "dictionary_synonyms returned a wrong-language or unscoped group"
                )
            _require_provenance(group.get("provenance"), f"synonym group {group_index}")
            candidates = group.get("synonyms")
            if not isinstance(candidates, list) or any(
                not isinstance(item, dict) for item in candidates
            ):
                raise AcceptanceFailure("dictionary_synonyms group is malformed")
            for candidate in cast(list[dict[str, Any]], candidates):
                actual_synonym_candidate_count += 1
                if candidate.get("language") != "en" or candidate.get("sense_scope") not in {
                    "sense",
                    "unsensed",
                }:
                    raise AcceptanceFailure(
                        "dictionary_synonyms returned a wrong-language or unscoped candidate"
                    )
                term = _require_text(candidate.get("term"), "synonym candidate term")
                if _normalized_term(term) == "important":
                    raise AcceptanceFailure("dictionary_synonyms echoed its query")
                synonym_terms.add(_normalized_term(term))
                _require_provenance(candidate.get("provenance"), f"synonym candidate {term}")
        synonym_candidate_count = synonym_value.get("candidate_count")
        if (
            isinstance(synonym_candidate_count, bool)
            or not isinstance(synonym_candidate_count, int)
            or synonym_candidate_count != actual_synonym_candidate_count
            or synonym_candidate_count > 20
        ):
            raise AcceptanceFailure("dictionary_synonyms candidate_count violates its total limit")
        if "significant" not in synonym_terms:
            raise AcceptanceFailure("dictionary_synonyms(important/en) did not return significant")

        river_translation = self._invoke_bank_translation(
            installation,
            sense_id=river_sense_id,
            expected_gloss="edge of river or lake",
            required_term="Ufer",
        )
        financial_translation = self._invoke_bank_translation(
            installation,
            sense_id=financial_sense_id,
            expected_gloss="institution",
            required_term="Bank",
        )

        relations, relation_value, relation_results = self._invoke_lexicon_tool(
            "dictionary_relations",
            {
                "word": "poodle",
                "language": "en",
                "relation": "hypernym",
                "target_language": "en",
                "limit": 100,
                "max_depth": 1,
                "transitive_limit": 0,
            },
            installation,
        )
        if (
            relation_value["query"].get("normalized_word") != "poodle"
            or relation_value["query"].get("relation") != "hypernym"
            or relation_value["query"].get("target_language") != "en"
            or relation_value["query"].get("max_depth") != 1
            or relation_value["query"].get("transitive_limit") != 0
        ):
            raise AcceptanceFailure("dictionary_relations returned a wrong query scope")
        dog_relation: dict[str, Any] | None = None
        for item in relation_results:
            if item.get("target_language") != "en":
                raise AcceptanceFailure("dictionary_relations violated language scope")
            _require_text(item.get("source_term"), "relation source_term")
            _require_text(item.get("source_language"), "relation source_language")
            _require_text(item.get("target_term"), "relation target_term")
            if item.get("sense_scope") not in {"sense", "unsensed"}:
                raise AcceptanceFailure("dictionary_relations returned invalid sense scope")
            _require_provenance(item.get("provenance"), "relation result")
            path = item.get("path")
            distance = item.get("distance")
            if (
                isinstance(distance, bool)
                or not isinstance(distance, int)
                or distance != 1
                or not isinstance(path, list)
                or len(path) != distance
            ):
                raise AcceptanceFailure("dictionary_relations returned a malformed path")
            for edge in path:
                if (
                    not isinstance(edge, dict)
                    or edge.get("relation") != "hypernym"
                    or edge.get("direction") not in {"outbound", "inbound"}
                ):
                    raise AcceptanceFailure(
                        "dictionary_relations returned a malformed directed edge"
                    )
                _require_text(edge.get("source_language"), "relation edge source language")
                _require_text(edge.get("target_language"), "relation edge target language")
                _require_provenance(edge.get("provenance"), "relation path edge")
            if (
                _normalized_term(str(item.get("target_term") or "")) == "dog"
                and item.get("relation") == "hypernym"
                and item.get("direction") == "outbound"
                and item.get("relation_scope") == "direct"
                and distance == 1
            ):
                dog_relation = item
        if dog_relation is None:
            raise AcceptanceFailure(
                "dictionary_relations(poodle/hypernym) did not return direct outbound dog"
            )

        semantic, semantic_value, semantic_results = self._invoke_lexicon_tool(
            "dictionary_semantic_neighbors",
            {
                "word": "cat",
                "source_language": "en",
                "target_language": "de",
                "limit": 20,
            },
            installation,
        )
        if semantic_value.get("available") is not True or not semantic_results:
            raise AcceptanceFailure("dictionary_semantic_neighbors is unavailable or empty")
        if (
            semantic_value["query"].get("source_language") != "en"
            or semantic_value["query"].get("target_language") != "de"
        ):
            raise AcceptanceFailure("dictionary_semantic_neighbors returned a wrong query scope")
        semantic_identities: set[tuple[str, str]] = set()
        semantic_scores: list[float] = []
        for item in semantic_results:
            term = _require_text(item.get("term"), "semantic neighbour term")
            _require_text(item.get("concept"), f"semantic neighbour {term} concept")
            semantic_id = item.get("semantic_id")
            if isinstance(semantic_id, bool) or not isinstance(semantic_id, int):
                raise AcceptanceFailure("semantic neighbour semantic_id is not an integer")
            if item.get("language") != "de" or item.get("sense_scope") != "unsensed":
                raise AcceptanceFailure(
                    "dictionary_semantic_neighbors violated language or sense scope"
                )
            score = item.get("similarity")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise AcceptanceFailure("semantic similarity is not numeric")
            score = float(score)
            if not math.isfinite(score) or not -1.0 <= score <= 1.0:
                raise AcceptanceFailure("semantic similarity is not finite cosine similarity")
            identity = ("de", _normalized_term(term))
            if identity in semantic_identities:
                raise AcceptanceFailure("dictionary_semantic_neighbors returned a duplicate")
            semantic_identities.add(identity)
            semantic_scores.append(score)
            _require_provenance(item.get("provenance"), f"semantic neighbour {term}")

        wordplay, wordplay_value, wordplay_results = self._invoke_lexicon_tool(
            "dictionary_wordplay",
            {"mode": "rhyme", "text": "cat", "limit": 100},
            installation,
        )
        if (
            wordplay_value["query"].get("mode") != "rhyme"
            or wordplay_value["query"].get("normalized_text") != "cat"
        ):
            raise AcceptanceFailure("dictionary_wordplay returned a wrong query scope")
        wordplay_terms: set[str] = set()
        for item in wordplay_results:
            term = _require_text(item.get("term"), "wordplay term")
            _require_text(item.get("phonemes"), f"wordplay candidate {term} phonemes")
            normalized = _normalized_term(term)
            if (
                item.get("language") != "en"
                or item.get("mode") != "rhyme"
                or item.get("sense_scope") != "unsensed"
            ):
                raise AcceptanceFailure("dictionary_wordplay returned a malformed result")
            if normalized == "cat":
                raise AcceptanceFailure("dictionary_wordplay echoed its query")
            if normalized in wordplay_terms:
                raise AcceptanceFailure("dictionary_wordplay returned a duplicate")
            wordplay_terms.add(normalized)
            _require_provenance(item.get("provenance"), f"wordplay candidate {term}")
        if "bat" not in wordplay_terms:
            raise AcceptanceFailure("dictionary_wordplay(rhyme/cat) did not return bat")

        suite: dict[str, Any] = {
            "schema_version": 1,
            "execution_cycle": 1,
            "transport": "mcpo-http" if self.host.is_live else "fixture-replay",
            "live_evidence": self.host.is_live,
            "tool_names": sorted(EXPECTED_LEXICON_TOOLS),
            "tools": {
                "dictionary_lookup": {
                    "response_sha256": lookup.body_sha256,
                    "count": lookup_value["count"],
                    "translations_limit": 3,
                    "embedded_translation_count": embedded_translation_count,
                    "translation_truncation_sense_ids": sorted(translation_truncation_sense_ids),
                    "river_sense_id": river_sense_id,
                    "financial_sense_id": financial_sense_id,
                },
                "dictionary_synonyms": {
                    "response_sha256": synonyms.body_sha256,
                    "count": len(synonym_groups),
                    "candidate_count": synonym_candidate_count,
                    "required_candidate": "significant",
                },
                "dictionary_translate": {
                    "call_count": 2,
                    "river": river_translation,
                    "financial": financial_translation,
                },
                "dictionary_relations": {
                    "response_sha256": relations.body_sha256,
                    "count": len(relation_results),
                    "required_edge": {
                        "source": "poodle",
                        "relation": "hypernym",
                        "target": "dog",
                        "direction": "outbound",
                        "distance": 1,
                    },
                },
                "dictionary_semantic_neighbors": {
                    "response_sha256": semantic.body_sha256,
                    "count": len(semantic_results),
                    "target_language": "de",
                    "finite_similarity_min": min(semantic_scores),
                    "finite_similarity_max": max(semantic_scores),
                },
                "dictionary_wordplay": {
                    "response_sha256": wordplay.body_sha256,
                    "count": len(wordplay_results),
                    "required_candidate": "bat",
                    "query_excluded": True,
                },
            },
            "cross_tool_sense_flow": {
                "word": "bank",
                "lookup_language": "en",
                "target_language": "de",
                "river": {
                    "selected_gloss": river_sense.get("gloss"),
                    "selected_sense_id": river_sense_id,
                    "translate_request_sense_id": river_translation["requested_sense_id"],
                    "required_translation": "Ufer",
                    "source_sense_preserved": True,
                },
                "financial": {
                    "selected_gloss": financial_sense.get("gloss"),
                    "selected_sense_id": financial_sense_id,
                    "translate_request_sense_id": financial_translation["requested_sense_id"],
                    "required_translation": "Bank",
                    "source_sense_preserved": True,
                },
            },
        }
        suite["details_sha256"] = _sha256_bytes(_canonical_json(suite))
        return suite

    def _assert_no_changes(self, monitor: ChangeMonitor) -> None:
        errors = monitor.errors()
        changes = monitor.changes()
        if errors:
            raise AcceptanceFailure(f"directory change monitor failed: {list(errors)!r}")
        if changes:
            raise AcceptanceFailure(
                "dataset or project venv changed during acceptance: "
                f"{[asdict(item) for item in changes[:100]]!r}"
            )

    def _restore_stack_after_failure(
        self,
        original: BaseException,
        *,
        cycle: int | None,
    ) -> None:
        """Best-effort restoration without replacing the acceptance failure."""

        details: dict[str, Any] = {
            "cycle": cycle,
            "original_error_type": type(original).__name__,
            "original_error": str(original),
        }
        try:
            try:
                ready = self._ready_snapshot()
                tree = capture_mcpo_tree(self.host, self.config)
                _assert_no_orphan_children(self.host.process_table(), self.config, tree)
                details.update(
                    {
                        "status": "restored",
                        "method": "already_ready",
                        "health": ready,
                        "mcpo_tree": tree.evidence(),
                    }
                )
            except BaseException as probe_error:
                details["initial_probe_error"] = {
                    "error_type": type(probe_error).__name__,
                    "error": str(probe_error),
                }
                stopped = self.host.run_script(
                    self.config.stop_script,
                    timeout_seconds=self.config.shutdown_timeout_seconds,
                )
                details["recovery_stop_command"] = asdict(stopped)
                self._wait_down()
                started = self.host.run_script(
                    self.config.start_script,
                    timeout_seconds=self.config.startup_timeout_seconds,
                )
                details["recovery_start_command"] = asdict(started)
                ready = self._wait_ready()
                tree = capture_mcpo_tree(self.host, self.config)
                _assert_no_orphan_children(self.host.process_table(), self.config, tree)
                details.update(
                    {
                        "status": "restored",
                        "method": "stop_start",
                        "health": ready,
                        "mcpo_tree": tree.evidence(),
                    }
                )
        except BaseException as recovery_error:
            details.update(
                {
                    "status": "failed",
                    "recovery_error_type": type(recovery_error).__name__,
                    "recovery_error": str(recovery_error),
                }
            )
            original.add_note(
                "Live-stack restoration also failed: "
                f"{type(recovery_error).__name__}: {recovery_error}"
            )
        self.failure_recovery = details
        try:
            self.evidence.emit(
                "failure_stack_recovery",
                status="ok" if details["status"] == "restored" else "failed",
                cycle=cycle,
                details=details,
            )
        except BaseException as evidence_error:
            original.add_note(
                "Recording live-stack restoration evidence also failed: "
                f"{type(evidence_error).__name__}: {evidence_error}"
            )

    def run(self) -> dict[str, Any]:
        started = self.host.now()
        installation = self.host.installation(self.config.data_root)
        if not _SHA256_RE.fullmatch(installation.manifest_sha256):
            raise AcceptanceFailure("installed manifest has an invalid SHA-256")
        if not _COMMIT_RE.fullmatch(installation.transformation_commit):
            raise AcceptanceFailure("installed manifest has an invalid transformation commit")
        monitor = self.host.begin_change_monitor((self.config.data_root, self.config.project_venv))
        recovery_required = False
        current_cycle: int | None = None
        primary_failure: BaseException | None = None
        try:
            self.evidence.emit(
                "baseline_inventory_started",
                details={
                    "data_root": str(self.config.data_root),
                    "active_dataset": str(installation.path),
                    "project_venv": str(self.config.project_venv),
                    "content_hashes": True,
                },
            )
            data_baseline = self.host.inventory(self.config.data_root, content=False)
            venv_baseline = self.host.inventory(self.config.project_venv, content=False)
            active_content_baseline = self.host.inventory(installation.path, content=True)
            venv_content_baseline = self.host.inventory(self.config.project_venv, content=True)
            transients = _forbidden_dataset_transients(data_baseline)
            if transients:
                raise AcceptanceFailure(
                    f"dataset contains forbidden transient artifacts: {list(transients)!r}"
                )
            self._assert_no_changes(monitor)
            self.evidence.emit(
                "baseline_inventory",
                details={
                    "installation": {
                        "dataset_version": installation.version,
                        "active_path": str(installation.path),
                        "manifest_sha256": installation.manifest_sha256,
                        "manifest_size": installation.manifest_size,
                        "transformation_commit": installation.transformation_commit,
                    },
                    "data_root_metadata": data_baseline.summary(),
                    "active_dataset_content": active_content_baseline.summary(),
                    "project_venv_metadata": venv_baseline.summary(),
                    "project_venv_content": venv_content_baseline.summary(),
                },
            )

            ready = self._wait_ready()
            old_tree = capture_mcpo_tree(self.host, self.config)
            _assert_no_orphan_children(self.host.process_table(), self.config, old_tree)
            self.evidence.emit(
                "preflight",
                details={"health": ready, "mcpo_tree": old_tree.evidence()},
            )

            for cycle in range(1, RESTART_CYCLES + 1):
                current_cycle = cycle
                old_tree = capture_mcpo_tree(self.host, self.config)
                self.evidence.emit(
                    "cycle_started", cycle=cycle, details={"old_mcpo_tree": old_tree.evidence()}
                )
                # A stop attempt can partially mutate the live stack even when
                # the script times out or returns non-zero. Keep recovery armed
                # until a fully ready replacement tree has been verified.
                recovery_required = True
                stopped = self.host.run_script(
                    self.config.stop_script,
                    timeout_seconds=self.config.shutdown_timeout_seconds,
                )
                if stopped.exit_code != 0:
                    raise AcceptanceFailure(
                        f"stop script failed in cycle {cycle}: {stopped.stderr_tail}"
                    )
                self._wait_old_tree_exit(old_tree)
                self._wait_down()
                _assert_no_orphan_children(self.host.process_table(), self.config, None)
                exclusive_count = self.host.exclusive_open(
                    tuple(
                        installation.path / record.relative_path
                        for record in active_content_baseline.records
                    )
                )
                if exclusive_count != active_content_baseline.file_count:
                    raise AcceptanceFailure("exclusive-open artifact count was incomplete")
                data_stopped = _assert_inventory_unchanged(self.host, data_baseline)
                venv_stopped = _assert_inventory_unchanged(self.host, venv_baseline)
                self._assert_no_changes(monitor)
                self.evidence.emit(
                    "stack_stopped",
                    cycle=cycle,
                    details={
                        "command": asdict(stopped),
                        "old_tree_exit_verified": True,
                        "exclusive_artifact_opens": exclusive_count,
                        "data_inventory": data_stopped.summary(),
                        "venv_inventory": venv_stopped.summary(),
                    },
                )

                started_command = self.host.run_script(
                    self.config.start_script,
                    timeout_seconds=self.config.startup_timeout_seconds,
                )
                if started_command.exit_code != 0:
                    raise AcceptanceFailure(
                        f"start script failed in cycle {cycle}: {started_command.stderr_tail}"
                    )
                ready = self._wait_ready()
                new_tree = capture_mcpo_tree(self.host, self.config)
                _assert_no_orphan_children(self.host.process_table(), self.config, new_tree)
                if new_tree.root.identity == old_tree.root.identity:
                    raise AcceptanceFailure("MCPO root identity did not change across restart")
                old_identities = {
                    old_tree.root.identity,
                    *(item.identity for item in old_tree.descendants),
                }
                current_identities = {item.identity for item in self.host.process_table()}
                if old_identities.intersection(current_identities):
                    raise AcceptanceFailure("an old MCPO process identity survived restart")
                recovery_required = False
                if cycle == 1:
                    self.six_tool_acceptance = self._invoke_all_lexicon_tools(installation)
                    self.evidence.emit(
                        "six_tool_acceptance_completed",
                        cycle=cycle,
                        details=self.six_tool_acceptance,
                    )
                calls = self._invoke_cycle_tools(installation)
                models = self._request("GET", self.config.router_models_url)
                _require_status(models, "router models after tool calls")
                active_models = _active_models(models.body)
                if active_models:
                    raise AcceptanceFailure(
                        f"cycle {cycle} finished with active models: {active_models!r}"
                    )
                data_started = _assert_inventory_unchanged(self.host, data_baseline)
                venv_started = _assert_inventory_unchanged(self.host, venv_baseline)
                self._assert_no_changes(monitor)
                cycle_evidence = {
                    "cycle": cycle,
                    "stop_command": asdict(stopped),
                    "start_command": asdict(started_command),
                    "old_mcpo_tree": old_tree.evidence(),
                    "new_mcpo_tree": new_tree.evidence(),
                    "old_tree_exit_verified": True,
                    "exclusive_artifact_opens": exclusive_count,
                    "health": ready,
                    "tool_calls": calls,
                    "six_tool_acceptance_executed": cycle == 1,
                    "active_models": active_models,
                    "data_inventory": data_started.summary(),
                    "venv_inventory": venv_started.summary(),
                }
                self.cycles.append(cycle_evidence)
                self.evidence.emit("cycle_completed", cycle=cycle, details=cycle_evidence)

            final_data = _assert_inventory_unchanged(self.host, data_baseline)
            final_venv = _assert_inventory_unchanged(self.host, venv_baseline)
            self.evidence.emit(
                "final_content_verification_started",
                details={
                    "active_dataset_bytes": active_content_baseline.total_bytes,
                    "project_venv_bytes": venv_content_baseline.total_bytes,
                },
            )
            final_active_content = _assert_inventory_unchanged(
                self.host, active_content_baseline, content=True
            )
            final_venv_content = _assert_inventory_unchanged(
                self.host, venv_content_baseline, content=True
            )
            self._assert_no_changes(monitor)
            final_ready = self._wait_ready()
            active_models = cast(list[Any], final_ready["active_models"])
            if active_models:
                raise AcceptanceFailure(
                    f"acceptance finished with active models: {active_models!r}"
                )
            final_tree = capture_mcpo_tree(self.host, self.config)
            _assert_no_orphan_children(self.host.process_table(), self.config, final_tree)
            self._assert_no_changes(monitor)
            if len(self.cycles) != RESTART_CYCLES:
                raise AcceptanceFailure("acceptance did not complete exactly ten cycles")
            if self.six_tool_acceptance is None:
                raise AcceptanceFailure("six-tool Lexicon acceptance was not executed")
            finished = self.host.now()
            report = {
                "schema_version": 1,
                "dataset_version": installation.version,
                "manifest_sha256": installation.manifest_sha256,
                "transformation_commit": installation.transformation_commit,
                "live_stack_ok": self.host.is_live,
                "restart_cycles": RESTART_CYCLES if self.host.is_live else 0,
                "simulated_restart_cycles": 0 if self.host.is_live else RESTART_CYCLES,
                "completed_restart_cycles": len(self.cycles),
                "active_models": active_models,
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "duration_seconds": max(0.0, (finished - started).total_seconds()),
                "platform": platform.platform(),
                "python": sys.version,
                "paths": {
                    "root": str(self.config.root),
                    "project": str(self.config.project),
                    "data_root": str(self.config.data_root),
                    "project_venv": str(self.config.project_venv),
                    "start_script": str(self.config.start_script),
                    "stop_script": str(self.config.stop_script),
                },
                "checks": {
                    "exact_lexicon_operations": sorted(EXPECTED_LEXICON_TOOLS),
                    "administration_operations_absent": True,
                    "router_health": True,
                    "open_webui_health": True,
                    "mcpo_health": True,
                    "all_six_lexicon_tools_validated": True,
                    "all_six_lexicon_tools_via_live_mcpo": self.host.is_live,
                    "bank_river_sense_translation_flow": True,
                    "semantic_neighbors_finite_and_language_filtered": True,
                    "wordplay_query_excluded": True,
                    "lexicon_lookup_each_cycle": True,
                    "calculator_each_cycle": True,
                    "old_mcpo_trees_exited": True,
                    "orphan_mcpo_children_absent": True,
                    "exclusive_artifact_opens_while_stopped": True,
                    "directory_change_events": [],
                    "inventories_unchanged": True,
                    "final_active_models_empty": True,
                },
                "live_stack_scope": {
                    "mcpo_http_tools": self.host.is_live,
                    "open_webui_health": self.host.is_live,
                    "open_webui_ordinary_prompts": False,
                    "note": (
                        "This runner validates MCPO HTTP tools and Open WebUI health. "
                        "Ordinary-prompt tool selection in the Open WebUI UI requires "
                        "separate observed evidence and is not claimed here."
                    ),
                },
                "six_tool_acceptance": self.six_tool_acceptance,
                "inventories": {
                    "data_root_baseline": data_baseline.summary(),
                    "data_root_final": final_data.summary(),
                    "active_dataset_baseline": active_content_baseline.summary(),
                    "active_dataset_final": final_active_content.summary(),
                    "project_venv_baseline": venv_content_baseline.summary(),
                    "project_venv_final": final_venv_content.summary(),
                    "project_venv_metadata_final": final_venv.summary(),
                },
                "final_health": final_ready,
                "final_mcpo_tree": final_tree.evidence(),
                "cycles": self.cycles,
            }
            self.evidence.emit(
                "acceptance_completed",
                details={
                    "live_stack_ok": report["live_stack_ok"],
                    "restart_cycles": report["restart_cycles"],
                    "simulated_restart_cycles": report["simulated_restart_cycles"],
                    "active_models": active_models,
                    "six_tool_details_sha256": self.six_tool_acceptance["details_sha256"],
                    "open_webui_ordinary_prompts": False,
                },
            )
            return report
        except BaseException as exc:
            primary_failure = exc
            if recovery_required:
                self._restore_stack_after_failure(exc, cycle=current_cycle)
            raise
        finally:
            try:
                monitor.close()
                self._assert_no_changes(monitor)
            except BaseException as monitor_error:
                if primary_failure is None:
                    raise
                primary_failure.add_note(
                    "Directory-monitor finalization also failed: "
                    f"{type(monitor_error).__name__}: {monitor_error}"
                )


def _read_installation(data_root: Path) -> Installation:
    pointer_path = data_root / "current.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceFailure(
            f"cannot read active dataset pointer {pointer_path}: {exc}"
        ) from exc
    if not isinstance(pointer, dict):
        raise AcceptanceFailure("active dataset pointer is not an object")
    version = pointer.get("version")
    relative_path = pointer.get("path")
    if not isinstance(version, str) or not isinstance(relative_path, str):
        raise AcceptanceFailure("active dataset pointer has no valid version/path")
    candidate = Path(relative_path)
    active_path = (candidate if candidate.is_absolute() else data_root / candidate).resolve()
    resolved_root = data_root.resolve()
    if not active_path.is_relative_to(resolved_root) or not active_path.is_dir():
        raise AcceptanceFailure("active dataset path is missing or escapes the data root")
    manifest_path = active_path / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = parse_manifest(raw)
    except (OSError, ValueError) as exc:
        raise AcceptanceFailure(f"cannot parse installed manifest {manifest_path}: {exc}") from exc
    if manifest.dataset_version != version:
        raise AcceptanceFailure("current.json and installed manifest versions differ")
    return Installation(
        version=version,
        path=active_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        transformation_commit=manifest.transformation_commit,
        manifest_size=len(raw),
    )


def _decode_notify_buffer(buffer: bytes, byte_count: int) -> tuple[tuple[int, str], ...]:
    actions: list[tuple[int, str]] = []
    offset = 0
    while offset < byte_count:
        if byte_count - offset < 12:
            raise AcceptanceFailure("truncated FILE_NOTIFY_INFORMATION record")
        next_offset = int.from_bytes(buffer[offset : offset + 4], "little")
        action = int.from_bytes(buffer[offset + 4 : offset + 8], "little")
        name_length = int.from_bytes(buffer[offset + 8 : offset + 12], "little")
        end = offset + 12 + name_length
        if name_length % 2 or end > byte_count:
            raise AcceptanceFailure("malformed FILE_NOTIFY_INFORMATION name")
        name = buffer[offset + 12 : end].decode("utf-16-le")
        actions.append((action, name))
        if next_offset == 0:
            break
        if next_offset < 12 or offset + next_offset > byte_count:
            raise AcceptanceFailure("malformed FILE_NOTIFY_INFORMATION offset")
        offset += next_offset
    return tuple(actions)


class _NullChangeMonitor:
    def changes(self) -> tuple[DirectoryChange, ...]:
        return ()

    def errors(self) -> tuple[str, ...]:
        return ()

    def close(self) -> None:
        return None


class _WindowsOverlapped(ctypes.Structure):
    """ctypes layout of the Windows OVERLAPPED structure."""

    _fields_ = (
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", ctypes.c_ulong),
        ("OffsetHigh", ctypes.c_ulong),
        ("hEvent", ctypes.c_void_p),
    )


class _WindowsChangeMonitor:
    _ACTION_NAMES: ClassVar[dict[int, str]] = {
        1: "added",
        2: "removed",
        3: "modified",
        4: "renamed_from",
        5: "renamed_to",
    }
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    def __init__(self, roots: Sequence[Path]) -> None:
        if os.name != "nt":
            raise AcceptanceFailure("Windows change monitoring is available only on Windows")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateFileW.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )
        self._kernel32.CreateFileW.restype = ctypes.c_void_p
        self._kernel32.ReadDirectoryChangesW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(_WindowsOverlapped),
            ctypes.c_void_p,
        )
        self._kernel32.ReadDirectoryChangesW.restype = ctypes.c_int
        self._kernel32.CreateEventW.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_wchar_p,
        )
        self._kernel32.CreateEventW.restype = ctypes.c_void_p
        self._kernel32.ResetEvent.argtypes = (ctypes.c_void_p,)
        self._kernel32.ResetEvent.restype = ctypes.c_int
        self._kernel32.WaitForSingleObject.argtypes = (
            ctypes.c_void_p,
            ctypes.c_ulong,
        )
        self._kernel32.WaitForSingleObject.restype = ctypes.c_ulong
        self._kernel32.GetOverlappedResult.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsOverlapped),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_int,
        )
        self._kernel32.GetOverlappedResult.restype = ctypes.c_int
        self._kernel32.CancelIoEx.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self._kernel32.CancelIoEx.restype = ctypes.c_int
        self._kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._kernel32.CloseHandle.restype = ctypes.c_int
        self._handles: list[int] = []
        self._threads: list[threading.Thread] = []
        self._events: list[DirectoryChange] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._shutdown_cancelled_handles: set[int] = set()
        for root in roots:
            handle = self._kernel32.CreateFileW(
                str(root),
                0x0001,  # FILE_LIST_DIRECTORY
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x40000000,  # BACKUP_SEMANTICS | OVERLAPPED
                None,
            )
            if handle == self._INVALID_HANDLE:
                error = ctypes.get_last_error()
                self.close()
                raise AcceptanceFailure(f"cannot monitor directory {root}: Windows error {error}")
            numeric_handle = int(handle)
            self._handles.append(numeric_handle)
            ready = threading.Event()
            thread = threading.Thread(
                target=self._watch,
                args=(root, numeric_handle, ready),
                daemon=True,
                name=f"lexicon-accept-watch-{len(self._threads)}",
            )
            self._threads.append(thread)
            thread.start()
            if not ready.wait(10):
                self.close()
                raise AcceptanceFailure(f"directory watcher did not start for {root}")
            startup_errors = self.errors()
            if startup_errors:
                self.close()
                raise AcceptanceFailure(
                    f"directory watcher did not arm for {root}: {startup_errors[-1]}"
                )

    def _watch(self, root: Path, handle: int, ready: threading.Event) -> None:
        buffer = ctypes.create_string_buffer(64 * 1024)
        event = self._kernel32.CreateEventW(None, True, False, None)
        if not event:
            error = ctypes.get_last_error()
            with self._lock:
                self._errors.append(f"{root}: cannot create watcher event: Windows error {error}")
            ready.set()
            return
        event_handle = int(event)
        try:
            while not self._stop.is_set():
                if not self._kernel32.ResetEvent(ctypes.c_void_p(event_handle)):
                    error = ctypes.get_last_error()
                    with self._lock:
                        self._errors.append(
                            f"{root}: cannot reset watcher event: Windows error {error}"
                        )
                    return
                overlapped = _WindowsOverlapped()
                overlapped.hEvent = event_handle
                success = self._kernel32.ReadDirectoryChangesW(
                    ctypes.c_void_p(handle),
                    buffer,
                    len(buffer),
                    True,
                    0x00000001 | 0x00000002 | 0x00000004 | 0x00000008 | 0x00000010 | 0x00000040,
                    None,
                    ctypes.byref(overlapped),
                    None,
                )
                if not success:
                    error = ctypes.get_last_error()
                    if error != 997:  # ERROR_IO_PENDING is normal overlapped I/O
                        if self._stop.is_set() and error in {6, 995}:
                            return
                        with self._lock:
                            self._errors.append(f"{root}: Windows error {error}")
                        return
                # This is the readiness boundary: the notification request is
                # now owned by the kernel, so an immediate caller write cannot
                # race ahead of the first ReadDirectoryChangesW operation.
                ready.set()
                wait_result = self._kernel32.WaitForSingleObject(
                    ctypes.c_void_p(event_handle), 0xFFFFFFFF
                )
                if wait_result != 0:  # WAIT_OBJECT_0
                    error = ctypes.get_last_error()
                    with self._lock:
                        self._errors.append(
                            f"{root}: watcher wait failed ({wait_result}): Windows error {error}"
                        )
                    return
                bytes_returned = ctypes.c_ulong()
                completed = self._kernel32.GetOverlappedResult(
                    ctypes.c_void_p(handle),
                    ctypes.byref(overlapped),
                    ctypes.byref(bytes_returned),
                    False,
                )
                if not completed:
                    error = ctypes.get_last_error()
                    if self._stop.is_set() and error in {6, 995}:
                        return
                    with self._lock:
                        self._errors.append(f"{root}: Windows error {error}")
                    return
                count = int(bytes_returned.value)
                if count == 0:
                    with self._lock:
                        shutdown_cancellation = (
                            self._stop.is_set() and handle in self._shutdown_cancelled_handles
                        )
                        if not shutdown_cancellation:
                            self._errors.append(f"{root}: change buffer overflow")
                    return
                try:
                    decoded = _decode_notify_buffer(buffer.raw, count)
                except AcceptanceFailure as exc:
                    with self._lock:
                        self._errors.append(f"{root}: {exc}")
                    return
                with self._lock:
                    for action, relative_path in decoded:
                        if len(self._events) >= 1_000:
                            break
                        self._events.append(
                            DirectoryChange(
                                root=str(root),
                                action=self._ACTION_NAMES.get(action, f"action_{action}"),
                                relative_path=relative_path,
                            )
                        )
        finally:
            ready.set()
            self._kernel32.CloseHandle(ctypes.c_void_p(event_handle))

    def changes(self) -> tuple[DirectoryChange, ...]:
        with self._lock:
            return tuple(self._events)

    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def close(self) -> None:
        self._stop.set()
        if hasattr(self, "_kernel32"):
            for handle in getattr(self, "_handles", []):
                # Hold the lock until a successful cancellation is recorded.
                # The worker can otherwise observe the resulting successful
                # zero-byte completion before close() marks it as intentional.
                with self._lock:
                    if self._kernel32.CancelIoEx(ctypes.c_void_p(handle), None):
                        self._shutdown_cancelled_handles.add(handle)
            for thread in getattr(self, "_threads", []):
                thread.join(timeout=5)
            for handle in getattr(self, "_handles", []):
                self._kernel32.CloseHandle(ctypes.c_void_p(handle))
        self._handles = []
        self._threads = []


class WindowsHost:
    """Real Windows implementation.  Construct it only after explicit CLI consent."""

    is_live = True

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def installation(self, data_root: Path) -> Installation:
        return _read_installation(data_root)

    def inventory(self, root: Path, *, content: bool) -> InventorySnapshot:
        return _inventory_records(root, content=content)

    def begin_change_monitor(self, roots: Sequence[Path]) -> ChangeMonitor:
        return _WindowsChangeMonitor(roots)

    def process_table(self) -> tuple[ProcessRecord, ...]:
        command = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,ParentProcessId,CreationDate,Name,"
            "ExecutablePath,CommandLine | "
            "ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise AcceptanceFailure(
                f"cannot inventory Windows processes: {_display_tail(result.stderr)}"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AcceptanceFailure("PowerShell returned invalid process JSON") from exc
        rows = value if isinstance(value, list) else [value]
        processes: list[ProcessRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = row.get("ProcessId")
            parent = row.get("ParentProcessId")
            if isinstance(pid, bool) or not isinstance(pid, int):
                continue
            if isinstance(parent, bool) or not isinstance(parent, int):
                continue
            processes.append(
                ProcessRecord(
                    pid=pid,
                    parent_pid=parent,
                    creation_time=str(row.get("CreationDate") or "unknown"),
                    name=str(row.get("Name") or ""),
                    executable=(
                        str(row["ExecutablePath"])
                        if isinstance(row.get("ExecutablePath"), str)
                        else None
                    ),
                    command_line=(
                        str(row["CommandLine"]) if isinstance(row.get("CommandLine"), str) else None
                    ),
                )
            )
        return tuple(processes)

    def read_pid(self, path: Path) -> int:
        try:
            value = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError) as exc:
            raise AcceptanceFailure(f"cannot read valid PID from {path}") from exc
        if value <= 0:
            raise AcceptanceFailure(f"PID in {path} is not positive")
        return value

    def run_script(self, path: Path, *, timeout_seconds: float) -> CommandObservation:
        if not path.is_file():
            raise AcceptanceFailure(f"stack script does not exist: {path}")
        started = time.monotonic()
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(path),
                ],
                cwd=path.parent,
                check=False,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise AcceptanceFailure(f"stack script timed out: {path}") from exc
        duration = time.monotonic() - started
        return CommandObservation(
            script=str(path),
            exit_code=result.returncode,
            duration_seconds=duration,
            stdout_sha256=_sha256_bytes(result.stdout),
            stderr_sha256=_sha256_bytes(result.stderr),
            stdout_tail=_display_tail(result.stdout),
            stderr_tail=_display_tail(result.stderr),
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float,
    ) -> HttpObservation:
        try:
            response = httpx.request(
                method,
                url,
                json=dict(payload) if payload is not None else None,
                timeout=timeout_seconds,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise HostRequestError(f"{method} {url}: {exc}") from exc
        raw = response.content
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return HttpObservation(response.status_code, body, _sha256_bytes(raw))

    def exclusive_open(self, paths: Sequence[Path]) -> int:
        if os.name != "nt":
            raise AcceptanceFailure("exclusive artifact opens require Windows")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        invalid = ctypes.c_void_p(-1).value
        opened = 0
        for path in paths:
            handle = kernel32.CreateFileW(
                str(path),
                0x80000000,  # GENERIC_READ
                0,  # no sharing: proves no leaked reader/map handle remains
                None,
                3,  # OPEN_EXISTING
                0x80,  # FILE_ATTRIBUTE_NORMAL
                None,
            )
            if handle == invalid:
                error = ctypes.get_last_error()
                raise AcceptanceFailure(
                    f"artifact cannot be opened exclusively: {path} (Windows error {error})"
                )
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            opened += 1
        return opened


class FixtureHost:
    """Deterministic ten-cycle host used only by ``--dry-run-fixture`` and tests."""

    is_live = False

    def __init__(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != 1:
            raise AcceptanceFailure("dry-run fixture schema_version must be 1")
        dataset = value.get("dataset")
        if not isinstance(dataset, dict):
            raise AcceptanceFailure("dry-run fixture requires dataset object")
        self._installation = Installation(
            version=str(dataset["version"]),
            path=Path(str(dataset["path"])),
            manifest_sha256=str(dataset["manifest_sha256"]),
            transformation_commit=str(dataset["transformation_commit"]),
            manifest_size=int(dataset.get("manifest_size", 1)),
        )
        self._value = dict(value)
        self._clock = 0.0
        self._generation = 0
        self._running = True
        self._script_calls: list[str] = []
        self._request_calls: list[dict[str, Any]] = []
        self._monitor = _NullChangeMonitor()
        self._inventory_drift_cycle = value.get("inventory_drift_cycle")
        self._orphan_cycle = value.get("orphan_cycle")
        self._extra_orphan = value.get("extra_orphan") is True
        self._late_active_model = value.get("late_active_model") is True
        self._content_inventory_calls = 0

    @classmethod
    def from_path(cls, path: Path) -> FixtureHost:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AcceptanceFailure(f"cannot read dry-run fixture {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise AcceptanceFailure("dry-run fixture root must be an object")
        return cls(value)

    @property
    def script_calls(self) -> tuple[str, ...]:
        return tuple(self._script_calls)

    @property
    def request_calls(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._request_calls)

    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=self._clock)

    def monotonic(self) -> float:
        return self._clock

    def sleep(self, seconds: float) -> None:
        self._clock += max(seconds, 0.001)

    def installation(self, data_root: Path) -> Installation:
        del data_root
        return self._installation

    def _records(self, root: Path, *, content: bool) -> tuple[FileRecord, ...]:
        marker = "active" if root == self._installation.path else root.name.casefold()
        drift = self._inventory_drift_cycle == self._generation and self._running
        return (
            FileRecord(
                relative_path=f"{marker}.artifact" + (".tmp" if drift else ""),
                size=100 + int(bool(drift)),
                mtime_ns=1 + int(bool(drift)),
                device=1,
                inode=1,
                sha256=("a" * 64) if content else None,
            ),
        )

    def inventory(self, root: Path, *, content: bool) -> InventorySnapshot:
        if content:
            self._content_inventory_calls += 1
        records = self._records(root, content=content)
        metadata = _sha256_bytes(
            _canonical_json(
                [
                    (
                        item.relative_path,
                        item.size,
                        item.mtime_ns,
                        item.device,
                        item.inode,
                    )
                    for item in records
                ]
            )
        )
        content_digest = (
            _sha256_bytes(
                _canonical_json([(item.relative_path, item.size, item.sha256) for item in records])
            )
            if content
            else None
        )
        return InventorySnapshot(
            root,
            len(records),
            sum(item.size for item in records),
            metadata,
            content_digest,
            records,
        )

    def begin_change_monitor(self, roots: Sequence[Path]) -> ChangeMonitor:
        del roots
        return self._monitor

    def _root_pid(self) -> int:
        return 10_000 + self._generation * 100

    def process_table(self) -> tuple[ProcessRecord, ...]:
        if not self._running:
            if self._orphan_cycle == self._generation:
                root = self._root_pid()
                return (ProcessRecord(root + 1, root, "orphan", "python.exe", None, "lexicon-mcp"),)
            return ()
        root = self._root_pid()
        processes = [
            ProcessRecord(
                4_242,
                1,
                "runner",
                "python.exe",
                r"E:\AI\lexicon-mcp\.venv\Scripts\python.exe",
                (
                    r"python E:\AI\lexicon-mcp\scripts\run_live_acceptance.py "
                    r"--project E:\AI\lexicon-mcp"
                ),
            ),
            ProcessRecord(root, 500, f"root-{self._generation}", "uvx.exe", None, "uvx mcpo"),
            ProcessRecord(
                root + 1,
                root,
                f"lexicon-{self._generation}",
                "python.exe",
                None,
                "python lexicon-mcp",
            ),
            ProcessRecord(
                root + 2,
                root,
                f"calculator-{self._generation}",
                "python.exe",
                None,
                "python calculator.py",
            ),
        ]
        if self._extra_orphan:
            processes.append(
                ProcessRecord(
                    99_999,
                    1,
                    "orphan",
                    "python.exe",
                    None,
                    "python lexicon-mcp",
                )
            )
        return tuple(processes)

    def read_pid(self, path: Path) -> int:
        del path
        if not self._running:
            raise AcceptanceFailure("fixture MCPO is stopped")
        return self._root_pid()

    def run_script(self, path: Path, *, timeout_seconds: float) -> CommandObservation:
        del timeout_seconds
        name = PureWindowsPath(str(path)).name.casefold()
        self._script_calls.append(name)
        if "stop" in name:
            self._running = False
        elif "start" in name:
            self._generation += 1
            self._running = True
        else:
            raise AcceptanceFailure(f"fixture does not recognize stack script {path}")
        self._clock += 1
        empty = _sha256_bytes(b"")
        return CommandObservation(str(path), 0, 1.0, empty, empty, "", "")

    def _lexicon_schema(self) -> dict[str, Any]:
        tools = sorted(EXPECTED_LEXICON_TOOLS)
        if self._value.get("include_admin_operation"):
            tools.append("repair")
        return {
            "openapi": "3.1.0",
            "paths": {f"/{name}": {"post": {"operationId": name}} for name in tools},
        }

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float,
    ) -> HttpObservation:
        del timeout_seconds
        if not self._running:
            raise HostRequestError(f"fixture endpoint is stopped: {url}")
        request_payload = dict(payload) if payload is not None else None
        self._request_calls.append(
            {"method": method.upper(), "url": url, "payload": request_payload}
        )
        provenance = {
            "source": "fixture source",
            "license": "fixture license",
            "url": "https://example.invalid/fixture",
        }
        wiktionary_provenance = {
            "source": "Wiktionary via Wiktextract",
            "license": "CC-BY-SA-4.0 and GFDL-1.3-or-later",
            "url": "https://kaikki.org/",
        }
        results: list[dict[str, Any]]
        if url.endswith("/models"):
            body: Any = {
                "active_models": (
                    ["late-model"]
                    if self._late_active_model and self._content_inventory_calls >= 4
                    else []
                )
            }
        elif url.endswith("/lexicon/openapi.json"):
            body = self._lexicon_schema()
        elif url.endswith("/calculator/openapi.json"):
            body = {"openapi": "3.1.0", "paths": {"/calculate": {"post": {}}}}
        elif url.endswith("/dictionary_lookup"):
            word = str((request_payload or {}).get("word", ""))
            if word == "bank":
                results = [
                    {
                        "sense_id": "wikt:labeled:bank-finance",
                        "sense_scope": "sense",
                        "word": "bank",
                        "language": "en",
                        "part_of_speech": "noun",
                        "gloss": "institution",
                        "examples": [],
                        "pronunciations": [{"ipa": "bank", "region": None}],
                        "etymology": None,
                        "translations": [
                            {
                                "term": "banque",
                                "language": "fr",
                                "sense_id": (
                                    "wikt:labeled:bank-river"
                                    if self._value.get("lookup_translation_crosses_sense")
                                    else "wikt:labeled:bank-finance"
                                ),
                                "sense_scope": "sense",
                                "provenance": wiktionary_provenance,
                            },
                            {
                                "term": "banco",
                                "language": "es",
                                "sense_id": "wikt:labeled:bank-finance",
                                "sense_scope": "sense",
                                "provenance": wiktionary_provenance,
                            },
                        ],
                        "truncated_fields": (
                            []
                            if self._value.get("lookup_missing_translation_truncation")
                            else ["translations"]
                        ),
                        "provenance": wiktionary_provenance,
                    },
                    {
                        "sense_id": "wikt:labeled:bank-river",
                        "sense_scope": "sense",
                        "word": "bank",
                        "language": "en",
                        "part_of_speech": "noun",
                        "gloss": "edge of river or lake",
                        "examples": [],
                        "pronunciations": [{"ipa": "bank", "region": None}],
                        "etymology": None,
                        "translations": [
                            {
                                "term": "rive",
                                "language": "fr",
                                "sense_id": "wikt:labeled:bank-river",
                                "sense_scope": "sense",
                                "provenance": wiktionary_provenance,
                            }
                        ],
                        "truncated_fields": (
                            []
                            if self._value.get("lookup_missing_translation_truncation")
                            else ["translations"]
                        ),
                        "provenance": wiktionary_provenance,
                    },
                ]
            else:
                results = [
                    {
                        "sense_id": "oewn:cat-n",
                        "sense_scope": "sense",
                        "word": "cat",
                        "language": "en",
                        "part_of_speech": "noun",
                        "gloss": "a cat",
                        "examples": [],
                        "pronunciations": [{"ipa": "kat", "region": None}],
                        "etymology": None,
                        "translations": [],
                        "truncated_fields": [],
                        "provenance": provenance,
                    }
                ]
            body = {
                "type": "dictionary_lookup",
                "dataset_version": self._installation.version,
                "query": {
                    **(request_payload or {}),
                    "normalized_word": word,
                    "part_of_speech": None,
                    "limit": int((request_payload or {}).get("limit", 8)),
                    "examples_limit": int((request_payload or {}).get("examples_limit", 8)),
                    "pronunciations_limit": int(
                        (request_payload or {}).get("pronunciations_limit", 8)
                    ),
                    "translations_limit": int(
                        (request_payload or {}).get("translations_limit", 20)
                    ),
                },
                "count": len(results),
                "results": results,
            }
        elif url.endswith("/dictionary_synonyms"):
            candidate = "notable" if self._value.get("missing_synonym_anchor") else "significant"
            results = [
                {
                    "sense_id": None,
                    "sense_scope": "unsensed",
                    "word": "important",
                    "language": "en",
                    "part_of_speech": None,
                    "gloss": None,
                    "synonyms": [
                        {
                            "term": candidate,
                            "language": "en",
                            "part_of_speech": None,
                            "sense_id": None,
                            "sense_scope": "unsensed",
                            "provenance": provenance,
                        }
                    ],
                    "provenance": provenance,
                }
            ]
            body = {
                "type": "dictionary_synonyms",
                "dataset_version": self._installation.version,
                "query": {
                    **(request_payload or {}),
                    "normalized_word": "important",
                    "sense_id": None,
                    "part_of_speech": None,
                },
                "count": len(results),
                "candidate_count": sum(len(group["synonyms"]) for group in results),
                "results": results,
            }
        elif url.endswith("/dictionary_translate"):
            requested_sense = str((request_payload or {}).get("sense_id") or "")
            river_id = "wikt:labeled:bank-river"
            financial_id = "wikt:labeled:bank-finance"
            if self._value.get("translation_crosses_sense"):
                returned_sense = financial_id if requested_sense == river_id else river_id
            else:
                returned_sense = requested_sense
            is_river = returned_sense == river_id
            translation_term = (
                "Flussrand"
                if is_river and self._value.get("missing_river_translation")
                else (
                    "Geldhaus"
                    if not is_river and self._value.get("missing_financial_translation")
                    else "Ufer"
                    if is_river
                    else "Bank"
                )
            )
            results = [
                {
                    "sense_id": returned_sense,
                    "sense_scope": "sense",
                    "word": "bank",
                    "source_language": "en",
                    "part_of_speech": "noun",
                    "gloss": "edge of river or lake" if is_river else "institution",
                    "translations": [
                        {
                            "term": translation_term,
                            "language": "de",
                            "sense_id": returned_sense,
                            "sense_scope": "sense",
                            "provenance": wiktionary_provenance,
                        }
                    ],
                    "provenance": wiktionary_provenance,
                }
            ]
            body = {
                "type": "dictionary_translate",
                "dataset_version": self._installation.version,
                "query": request_payload or {},
                "count": len(results),
                "candidate_count": sum(len(group["translations"]) for group in results),
                "results": results,
            }
        elif url.endswith("/dictionary_relations"):
            results = [
                {
                    "source_term": "poodle",
                    "source_language": "en",
                    "source_sense_id": None,
                    "sense_scope": "unsensed",
                    "relation": "hypernym",
                    "target_term": "dog",
                    "target_language": "en",
                    "target_sense_id": None,
                    "direction": "outbound",
                    "relation_scope": "direct",
                    "distance": 1,
                    "path": [
                        {
                            "source_term": "poodle",
                            "source_language": "en",
                            "source_sense_id": None,
                            "relation": "hypernym",
                            "target_term": "dog",
                            "target_language": "en",
                            "target_sense_id": None,
                            "direction": "outbound",
                            "provenance": provenance,
                        }
                    ],
                    "provenance": provenance,
                }
            ]
            body = {
                "type": "dictionary_relations",
                "dataset_version": self._installation.version,
                "query": {
                    **(request_payload or {}),
                    "normalized_word": "poodle",
                },
                "count": len(results),
                "results": results,
            }
        elif url.endswith("/dictionary_semantic_neighbors"):
            language = "en" if self._value.get("semantic_wrong_language") else "de"
            results = [
                {
                    "semantic_id": 42,
                    "concept": f"/c/{language}/katze",
                    "term": "Katze",
                    "language": language,
                    "similarity": 0.8125,
                    "sense_scope": "unsensed",
                    "provenance": provenance,
                }
            ]
            body = {
                "type": "dictionary_semantic_neighbors",
                "dataset_version": self._installation.version,
                "query": request_payload or {},
                "count": len(results),
                "results": results,
                "available": True,
            }
        elif url.endswith("/dictionary_wordplay"):
            term = "cat" if self._value.get("wordplay_echo_query") else "bat"
            results = [
                {
                    "term": term,
                    "language": "en",
                    "mode": "rhyme",
                    "phonemes": "B AE1 T",
                    "sense_scope": "unsensed",
                    "provenance": provenance,
                }
            ]
            body = {
                "type": "dictionary_wordplay",
                "dataset_version": self._installation.version,
                "query": {
                    **(request_payload or {}),
                    "normalized_text": "cat",
                    "language": "en",
                },
                "count": len(results),
                "results": results,
            }
        elif url.endswith("/calculate"):
            body = 42
        elif url.endswith("/openapi.json"):
            body = {"openapi": "3.1.0", "paths": {}}
        elif url.endswith("/health") and ":18000" in url:
            body = {"status": True}
        elif url.endswith("/health"):
            body = {"status": "ok"}
        else:
            raise HostRequestError(f"fixture has no endpoint {url}")
        raw = _canonical_json(body)
        return HttpObservation(200, body, _sha256_bytes(raw))

    def exclusive_open(self, paths: Sequence[Path]) -> int:
        if self._running:
            raise AcceptanceFailure("fixture exclusive opens require stopped state")
        return len(paths)


def _merge_base_report(report: dict[str, Any], path: Path | None) -> dict[str, Any]:
    if path is None:
        return report
    try:
        base = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceFailure(f"cannot read base acceptance report {path}: {exc}") from exc
    if not isinstance(base, dict):
        raise AcceptanceFailure("base acceptance report must be an object")
    for key in ("dataset_version", "manifest_sha256", "transformation_commit"):
        if base.get(key) != report.get(key):
            raise AcceptanceFailure(f"base acceptance report {key} does not match live run")
    merged = dict(base)
    merged.update(report)
    return merged


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AcceptanceFailure(f"refusing to overwrite acceptance report: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json(dict(report)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_from_args(args: argparse.Namespace) -> AcceptanceConfig:
    return AcceptanceConfig(
        root=args.root.resolve(),
        project=args.project.resolve(),
        data_root=args.data_root.resolve(),
        project_venv=args.project_venv.resolve(),
        start_script=args.start_script.resolve(),
        stop_script=args.stop_script.resolve(),
        mcpo_pid_file=args.mcpo_pid_file.resolve(),
        router_health_url=args.router_health_url,
        router_models_url=args.router_models_url,
        webui_health_url=args.webui_health_url,
        mcpo_openapi_url=args.mcpo_openapi_url,
        lexicon_base_url=args.lexicon_base_url,
        calculator_base_url=args.calculator_base_url,
        startup_timeout_seconds=args.startup_timeout,
        shutdown_timeout_seconds=args.shutdown_timeout,
        request_timeout_seconds=args.request_timeout,
        poll_interval_seconds=args.poll_interval,
        mcpo_command_marker=args.mcpo_command_marker,
        required_child_markers=tuple(args.required_child_marker),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an auditable, exactly-ten-cycle Lexicon live-stack acceptance."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--execute-live",
        action="store_true",
        help="explicitly authorize ten real stop/start cycles",
    )
    mode.add_argument(
        "--dry-run-fixture",
        type=Path,
        help="replay a deterministic fixture; never touches live services",
    )
    parser.add_argument("--root", type=Path, default=Path(r"E:\AI"))
    parser.add_argument("--project", type=Path, default=Path(r"E:\AI\lexicon-mcp"))
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\AI\data\lexicon-mcp"))
    parser.add_argument("--project-venv", type=Path, default=Path(r"E:\AI\lexicon-mcp\.venv"))
    parser.add_argument("--start-script", type=Path, default=Path(r"E:\AI\scripts\start.ps1"))
    parser.add_argument("--stop-script", type=Path, default=Path(r"E:\AI\scripts\stop.ps1"))
    parser.add_argument("--mcpo-pid-file", type=Path, default=Path(r"E:\AI\run\mcpo.pid"))
    parser.add_argument("--router-health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--router-models-url", default="http://127.0.0.1:8080/v1/models")
    parser.add_argument("--webui-health-url", default="http://127.0.0.1:18000/health")
    parser.add_argument("--mcpo-openapi-url", default="http://127.0.0.1:18010/openapi.json")
    parser.add_argument("--lexicon-base-url", default="http://127.0.0.1:18010/lexicon")
    parser.add_argument("--calculator-base-url", default="http://127.0.0.1:18010/calculator")
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--shutdown-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--mcpo-command-marker", default="mcpo")
    parser.add_argument(
        "--required-child-marker",
        action="append",
        default=["lexicon-mcp", "calculator.py"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\AI\state\lexicon-mcp-build\acceptance"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--base-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute_live and os.name != "nt":
        raise SystemExit("--execute-live is supported only on Windows")
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise SystemExit("--run-id must contain only letters, numbers, dot, underscore, or dash")
    output_dir = args.output_dir.resolve()
    jsonl_path = output_dir / f"live-acceptance-{run_id}.jsonl"
    report_path = output_dir / f"live-acceptance-{run_id}.json"
    if report_path.exists() or jsonl_path.exists():
        raise SystemExit("refusing to overwrite existing acceptance evidence")
    config = _config_from_args(args)
    host: AcceptanceHost
    if args.execute_live:
        host = WindowsHost()
    else:
        host = FixtureHost.from_path(args.dry_run_fixture.resolve())
    evidence = EvidenceWriter(jsonl_path, run_id=run_id)
    runner = LiveAcceptanceRunner(host, config, evidence)
    report: dict[str, Any]
    exit_code = 0
    try:
        report = runner.run()
        report = _merge_base_report(report, args.base_report)
    except BaseException as exc:
        exit_code = 1
        evidence.emit(
            "acceptance_failed",
            status="failed",
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        report = {
            "schema_version": 1,
            "live_stack_ok": False,
            "restart_cycles": 0,
            "completed_restart_cycles": len(runner.cycles),
            "active_models": None,
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "six_tool_acceptance": runner.six_tool_acceptance,
            "failure_recovery": runner.failure_recovery,
            "cycles": runner.cycles,
        }
    events_sha256 = evidence.close()
    report["run_id"] = run_id
    report["events_jsonl"] = str(jsonl_path)
    report["events_sha256"] = events_sha256
    report["dry_run"] = not host.is_live
    _write_report(report_path, report)
    print(json.dumps({"report": str(report_path), "ok": exit_code == 0}), flush=True)
    return exit_code


__all__ = [
    "EXPECTED_LEXICON_TOOLS",
    "RESTART_CYCLES",
    "AcceptanceConfig",
    "AcceptanceFailure",
    "EvidenceWriter",
    "FixtureHost",
    "InventorySnapshot",
    "LiveAcceptanceRunner",
    "ProcessRecord",
    "WindowsHost",
    "capture_mcpo_tree",
    "inventory_diff",
    "main",
    "openapi_operations",
    "validate_lexicon_openapi",
]
