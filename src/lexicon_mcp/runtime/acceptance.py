"""Protocol-independent release acceptance and isolated performance helpers."""

from __future__ import annotations

import math
import multiprocessing
import os
import sqlite3
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from .evidence import compact_evidence_json
from .locator import ActiveDataset, DatasetLocator
from .offline import deny_network
from .service import LexiconService

MIB = 1024 * 1024


class AcceptanceDatasetUnavailable(RuntimeError):
    """No activated full corpus is present on this host."""


def load_acceptance_dataset() -> ActiveDataset:
    """Resolve the explicit data root or the standard E:\\AI full-corpus root.

    An explicitly configured but invalid root is a failure. The conventional
    Windows root being absent means this host simply cannot run full-corpus gates.
    """

    configured = os.environ.get("LEXICON_DATA_DIR")
    if configured:
        dataset = DatasetLocator(configured).active()
    else:
        if os.name != "nt":
            raise AcceptanceDatasetUnavailable(
                "set LEXICON_DATA_DIR to an activated full lexicon dataset"
            )
        default = Path(r"E:\AI\data\lexicon-mcp")
        if not (default / "current.json").is_file():
            raise AcceptanceDatasetUnavailable(
                f"no activated full lexicon dataset at {default}"
            )
        dataset = DatasetLocator(default).active()
    if dataset.manifest.get("profile") != "full":
        raise RuntimeError("release acceptance requires a manifest with profile='full'")
    return dataset


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no measurements")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be greater than 0 and at most 1")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def process_private_bytes(pid: int | None = None) -> int:
    """Return private/unique bytes rather than mapped-file working set."""

    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - acceptance extra guarantees it
        raise RuntimeError("psutil is required for performance acceptance") from exc
    process = psutil.Process(pid or os.getpid())
    basic: Any = process.memory_info()
    private = getattr(basic, "private", None)
    if private is not None:
        return int(private)
    full: Any = process.memory_full_info()
    unique = getattr(full, "uss", None)
    return int(unique if unique is not None else full.rss)


def process_working_set_bytes(pid: int | None = None) -> int:
    """Return the process resident set (the Windows total working set)."""

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - acceptance extra guarantees it
        raise RuntimeError("psutil is required for performance acceptance") from exc
    process = psutil.Process(pid or os.getpid())
    return max(0, int(process.memory_info().rss))


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Acceptance timings plus independent process-memory observations.

    The mapped-artifact RSS diagnostic is not additive with the worker working
    set. In particular, Windows can report the entire mmap region for a file
    even when only part of that region is in the process working set.
    """

    lexical_p95_ms: float
    semantic_cold_ms: float
    semantic_warm_p95_ms: float
    idle_private_bytes: int
    semantic_worker_peak_private_bytes: int
    semantic_worker_peak_working_set_bytes: int
    semantic_worker_peak_mapped_artifact_rss_bytes: int
    lexical_samples: int
    semantic_warm_samples: int
    semantic_seed: str
    semantic_language: str
    # One entry per wordplay kind: first-call latency on this connection and
    # warm p95 across limit=1,20,100 samples.
    wordplay_cold_ms: dict[str, float] = field(default_factory=dict)
    wordplay_warm_p95_ms: dict[str, float] = field(default_factory=dict)
    wordplay_samples: int = 0

    def to_evidence(self) -> dict[str, object]:
        """Return every measurement with non-overlapping memory labels."""

        return {
            "latency_ms": {
                "lexical_p95": self.lexical_p95_ms,
                "semantic_cold": self.semantic_cold_ms,
                "semantic_warm_p95": self.semantic_warm_p95_ms,
                "wordplay_cold_by_kind": self.wordplay_cold_ms,
                "wordplay_warm_p95_by_kind": self.wordplay_warm_p95_ms,
            },
            "memory_bytes": {
                "idle_process_private": self.idle_private_bytes,
                "semantic_worker_peak_mapped_artifact_rss_diagnostic": (
                    self.semantic_worker_peak_mapped_artifact_rss_bytes
                ),
                "semantic_worker_peak_private": (
                    self.semantic_worker_peak_private_bytes
                ),
                "semantic_worker_peak_total_working_set": (
                    self.semantic_worker_peak_working_set_bytes
                ),
            },
            "report": "performance_acceptance",
            "samples": {
                "lexical": self.lexical_samples,
                "semantic_warm": self.semantic_warm_samples,
                "wordplay_warm": self.wordplay_samples,
            },
            "semantic_query": {
                "language": self.semantic_language,
                "seed": self.semantic_seed,
            },
        }

    def to_json(self) -> str:
        """Return one deterministic compact JSON evidence record."""

        return compact_evidence_json(self.to_evidence())


def _semantic_seed(directory: Path) -> tuple[str, str]:
    mapping = directory / "mapping.sqlite3"
    connection = sqlite3.connect(
        f"file:{mapping.as_posix()}?mode=ro&immutable=1", uri=True
    )
    try:
        row = connection.execute(
            """
            SELECT term.term, term.language
            FROM semantic_terms AS semantic
            JOIN lexical_terms AS term ON term.term_id = semantic.term_id
            WHERE term.normalized_term = 'cat' AND term.language = 'en'
            ORDER BY semantic.semantic_id LIMIT 1
            """
        ).fetchone()
        if row is None:
            row = connection.execute(
                """
                SELECT term.term, term.language
                FROM semantic_terms AS semantic
                JOIN lexical_terms AS term ON term.term_id = semantic.term_id
                GROUP BY term.normalized_term, term.language
                HAVING COUNT(*) = 1
                ORDER BY CASE term.language WHEN 'en' THEN 0 ELSE 1 END,
                         semantic.semantic_id
                LIMIT 1
                """
            ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError("semantic mapping contains no benchmark seed")
    return str(row[0]), str(row[1])


def process_mapped_artifact_rss_bytes(
    pid: int, semantic_directory: str | Path
) -> int:
    """Sum OS-reported RSS for mappings of semantic dataset artifacts.

    This is a mapping diagnostic, not a process working-set measurement. Some
    Windows versions report the whole mapped region as RSS here.
    """

    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - acceptance extra guarantees it
        raise RuntimeError("psutil is required for performance acceptance") from exc

    root = os.path.normcase(os.path.realpath(os.fspath(semantic_directory)))
    total = 0
    for mapping in psutil.Process(pid).memory_maps(grouped=False):
        raw_path = str(getattr(mapping, "path", ""))
        if not raw_path or raw_path.startswith("["):
            continue
        mapped_path = os.path.normcase(os.path.realpath(raw_path))
        try:
            if os.path.commonpath((root, mapped_path)) != root:
                continue
        except ValueError:
            continue
        name = os.path.basename(mapped_path)
        if not name.endswith(
            (".usearch", ".f16", ".sqlite", ".sqlite3", ".db", "-wal", "-shm")
        ):
            continue
        total += max(0, int(getattr(mapping, "rss", 0)))
    return total


def _semantic_artifact_rss_by_name(pid: int) -> int:
    """Handle Windows mappings whose path predates atomic directory promotion."""

    import psutil

    total = 0
    for mapping in psutil.Process(pid).memory_maps(grouped=False):
        name = os.path.basename(str(getattr(mapping, "path", ""))).casefold()
        if name.endswith(
            (".usearch", ".f16", ".sqlite", ".sqlite3", ".db", "-wal", "-shm")
        ):
            total += max(0, int(getattr(mapping, "rss", 0)))
    return total


def _sample_semantic_children(
    parent: Any,
    private_peak: list[int],
    working_set_peak: list[int],
    mapped_peak: list[int],
    semantic_directory: Path,
) -> None:
    try:
        import psutil
    except ImportError:
        return
    try:
        children = parent.children(recursive=True)
    except (psutil.Error, OSError):
        return
    for child in children:
        try:
            private_peak[0] = max(private_peak[0], process_private_bytes(child.pid))
            working_set_peak[0] = max(
                working_set_peak[0], process_working_set_bytes(child.pid)
            )
            mapped = process_mapped_artifact_rss_bytes(child.pid, semantic_directory)
            if mapped == 0:
                # USearch mmap views retain the pre-promotion staging path on
                # Windows even after the containing directory is atomically
                # renamed. Match the same artifact filenames as a fallback;
                # the worker opens only this dataset's semantic index/vector.
                mapped = _semantic_artifact_rss_by_name(child.pid)
            mapped_peak[0] = max(mapped_peak[0], mapped)
        except (psutil.Error, OSError):
            continue


def _monitor_semantic_children(
    stop: threading.Event,
    private_peak: list[int],
    working_set_peak: list[int],
    mapped_peak: list[int],
    semantic_directory: Path,
) -> None:
    try:
        import psutil
    except ImportError:
        return
    parent = psutil.Process(os.getpid())
    while not stop.wait(0.02):
        _sample_semantic_children(
            parent, private_peak, working_set_peak, mapped_peak, semantic_directory
        )
    # Capture persistent mmap views once more after the final warm query.
    _sample_semantic_children(
        parent, private_peak, working_set_peak, mapped_peak, semantic_directory
    )


def _performance_worker(
    channel: Connection,
    database: str,
    dataset_version: str,
    semantic_directory: str,
    lexical_iterations: int,
    semantic_iterations: int,
) -> None:
    service: LexiconService | None = None
    try:
        with deny_network():
            service = LexiconService(
                database,
                dataset_version,
                semantic_directory=semantic_directory,
            )
            idle_private = process_private_bytes()
            lexical_calls = (
                lambda: service.dictionary_lookup("bank", "en"),
                lambda: service.dictionary_synonyms("important", "en"),
                lambda: service.dictionary_translate("bank", "en", "de"),
                lambda: service.dictionary_relations("dog", "hypernym", "en"),
                lambda: service.dictionary_relations("thing", "hyponym", "en"),
                lambda: service.dictionary_relations("object", "hyponym", "en"),
                lambda: service.dictionary_relations("animal", "hyponym", "en"),
                lambda: service.dictionary_relations("person", "hyponym", "en"),
                lambda: service.dictionary_wordplay("rhyme", "cat"),
            )
            lexical_timings: list[float] = []
            for index in range(lexical_iterations):
                started = time.perf_counter()
                lexical_calls[index % len(lexical_calls)]()
                lexical_timings.append((time.perf_counter() - started) * 1000.0)

            # One cold (first-call on this connection) plus warm samples at
            # limit=1,20,100 per wordplay kind, including a high-fanout
            # anagram probe and a two-alternative spoonerism pairing.
            wordplay_queries: list[tuple[str, str, str | None, int]] = [
                ("anagram", "listen", None, 1),
                ("anagram", "listen", None, 20),
                ("anagram", "stare", None, 100),
                ("palindrome", "level", None, 1),
                ("palindrome", "level", None, 20),
                ("palindrome", "level", None, 100),
                ("spoonerism", "light rain", None, 1),
                ("spoonerism", "light rain", None, 20),
                ("spoonerism", "light rain", None, 100),
                ("pun", "sea", None, 1),
                ("pun", "sea", "the sea was calm", 20),
                ("pun", "sea", None, 100),
            ]
            wordplay_cold: dict[str, float] = {}
            wordplay_warm: dict[str, list[float]] = {}
            for kind, text, context, limit in wordplay_queries:
                started = time.perf_counter()
                service.wordplay(text, kind, context, limit)
                elapsed = (time.perf_counter() - started) * 1000.0
                if kind not in wordplay_cold or elapsed > wordplay_cold[kind]:
                    wordplay_cold[kind] = elapsed
                timings = wordplay_warm.setdefault(kind, [])
                for _repeat in range(5):
                    started = time.perf_counter()
                    service.wordplay(text, kind, context, limit)
                    timings.append((time.perf_counter() - started) * 1000.0)
            wordplay_p95 = {
                kind: percentile(timings, 0.95) for kind, timings in wordplay_warm.items()
            }
            wordplay_samples = sum(len(t) for t in wordplay_warm.values())

            seed, language = _semantic_seed(Path(semantic_directory))
            stop = threading.Event()
            child_private_peak = [0]
            child_working_set_peak = [0]
            child_mapped_peak = [0]
            monitor = threading.Thread(
                target=_monitor_semantic_children,
                args=(
                    stop,
                    child_private_peak,
                    child_working_set_peak,
                    child_mapped_peak,
                    Path(semantic_directory).resolve(),
                ),
                name="semantic-memory-monitor",
                daemon=True,
            )
            monitor.start()
            started = time.perf_counter()
            cold = service.dictionary_semantic_neighbors(seed, language, language, 20)
            cold_ms = (time.perf_counter() - started) * 1000.0
            if not cold["available"] or not cold["results"]:
                raise RuntimeError("semantic cold query returned no neighbours")
            warm_timings: list[float] = []
            for _ in range(semantic_iterations):
                started = time.perf_counter()
                warm = service.dictionary_semantic_neighbors(seed, language, language, 20)
                warm_timings.append((time.perf_counter() - started) * 1000.0)
                if not warm["results"]:
                    raise RuntimeError("semantic warm query returned no neighbours")
            stop.set()
            monitor.join(timeout=2)
            report = PerformanceReport(
                lexical_p95_ms=percentile(lexical_timings, 0.95),
                semantic_cold_ms=cold_ms,
                semantic_warm_p95_ms=percentile(warm_timings, 0.95),
                idle_private_bytes=idle_private,
                semantic_worker_peak_private_bytes=child_private_peak[0],
                semantic_worker_peak_working_set_bytes=child_working_set_peak[0],
                semantic_worker_peak_mapped_artifact_rss_bytes=child_mapped_peak[0],
                lexical_samples=len(lexical_timings),
                semantic_warm_samples=len(warm_timings),
                semantic_seed=seed,
                semantic_language=language,
                wordplay_cold_ms=wordplay_cold,
                wordplay_warm_p95_ms=wordplay_p95,
                wordplay_samples=wordplay_samples,
            )
            channel.send({"ok": True, "report": asdict(report)})
    except BaseException:
        channel.send({"ok": False, "error": traceback.format_exc()})
    finally:
        if service is not None:
            service.close()
        channel.close()


def run_isolated_performance(
    dataset: ActiveDataset,
    *,
    lexical_iterations: int = 100,
    semantic_iterations: int = 20,
    timeout_seconds: float = 180.0,
) -> PerformanceReport:
    """Benchmark in a clean spawned process so pytest imports do not inflate memory."""

    if lexical_iterations < 5 or semantic_iterations < 2:
        raise ValueError("performance acceptance needs at least 5 lexical and 2 semantic samples")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_performance_worker,
        args=(
            send,
            str(dataset.lexical_database),
            dataset.version,
            str(dataset.semantic_directory),
            lexical_iterations,
            semantic_iterations,
        ),
        name="lexicon-performance-acceptance",
    )
    process.start()
    send.close()
    if not receive.poll(timeout_seconds):
        process.terminate()
        process.join(timeout=10)
        raise TimeoutError(f"performance acceptance exceeded {timeout_seconds:g} seconds")
    payload: dict[str, Any] = receive.recv()
    receive.close()
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        raise RuntimeError("performance acceptance worker did not shut down")
    if process.exitcode != 0 and payload.get("ok"):
        raise RuntimeError(f"performance worker exited with code {process.exitcode}")
    if not payload.get("ok"):
        raise RuntimeError(f"performance worker failed:\n{payload.get('error', 'unknown error')}")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise RuntimeError("performance worker returned no report")
    return PerformanceReport(**report)
