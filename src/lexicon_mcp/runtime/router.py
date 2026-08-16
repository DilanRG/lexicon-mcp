"""Route queries to the pack that serves a language, and say so when none does.

The router is the runtime's view of an activation.  It answers two questions:
which database serves a given capability and language, and -- when none does --
precisely why.  That second answer matters as much as the first: an MCP client
cannot tell "this word does not exist" from "you never installed that language"
unless the server distinguishes them, and a model reading an empty result will
confidently assume the former.

Pack connections open on first use and are held in a small LRU.  A full install
is on the order of 55 packs, but a session typically touches one to three
languages, so opening every installed pack up front would pay for cache and file
handles nothing is going to read.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..data.activation import Activation, ActivationComponent
from ..data.manifest import ManifestError, normalize_language
from ..data.store import ComponentStore

# Why a capability is unavailable for a language. These strings are a contract:
# the CLI and the MCP tool responses both surface them verbatim.
INSTALLED = "installed"
LANGUAGE_NOT_INSTALLED = "language_not_installed"
CAPABILITY_NOT_INSTALLED = "capability_not_installed"
NOT_AVAILABLE_UPSTREAM = "not_available_upstream"
UNKNOWN_LANGUAGE = "unknown_language"

DEFAULT_MAX_OPEN_PACKS = 8


class RouterError(RuntimeError):
    """The activation cannot be served."""


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether a capability can serve a language here, and why not."""

    capability: str
    language: str
    installed: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "language": self.language,
            "installed": self.installed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    """What the dataset carries for a language, independent of this install."""

    language: str
    term_count: int
    entry_count: int
    sense_count: int
    translation_count: int
    relation_count: int
    has_semantic: bool
    has_pronunciation: bool
    has_wordplay: bool

    def offers(self, capability: str) -> bool:
        if capability == "lexical":
            return True
        return {
            "semantic": self.has_semantic,
            "pronunciation": self.has_pronunciation,
            "wordplay": self.has_wordplay,
        }.get(capability, False)


class PackRouter:
    """Open pack databases on demand and report capability coverage."""

    def __init__(
        self,
        activation: Activation,
        store: ComponentStore,
        *,
        max_open_packs: int = DEFAULT_MAX_OPEN_PACKS,
    ) -> None:
        if max_open_packs < 1:
            raise RouterError("max_open_packs must be at least 1")
        self.activation = activation
        self.store = store
        self.max_open_packs = max_open_packs
        self._lock = threading.RLock()
        self._open: OrderedDict[str, sqlite3.Connection] = OrderedDict()
        self._coverage: dict[str, LanguageCoverage] | None = None

    # ------------------------------------------------------------- coverage

    @property
    def coverage(self) -> dict[str, LanguageCoverage]:
        """The dataset's language catalogue, read once from the core pack.

        This is what separates "not installed" from "the corpus never had it".
        """

        with self._lock:
            if self._coverage is None:
                self._coverage = self._load_coverage()
            return self._coverage

    def _load_coverage(self) -> dict[str, LanguageCoverage]:
        core = self._core_component()
        connection = self._connect(core)
        rows = connection.execute(
            "SELECT language, term_count, entry_count, sense_count,"
            " translation_count, relation_count, has_semantic,"
            " has_pronunciation, has_wordplay FROM language_catalogue"
        ).fetchall()
        return {
            str(row[0]): LanguageCoverage(
                language=str(row[0]),
                term_count=int(row[1]),
                entry_count=int(row[2]),
                sense_count=int(row[3]),
                translation_count=int(row[4]),
                relation_count=int(row[5]),
                has_semantic=bool(row[6]),
                has_pronunciation=bool(row[7]),
                has_wordplay=bool(row[8]),
            )
            for row in rows
        }

    def _core_component(self) -> ActivationComponent:
        for pack in self.activation.packs:
            if pack.capability == "core":
                return self.activation.component(pack.components[0])
        raise RouterError("activation has no core pack; the catalogue is unreadable")

    # -------------------------------------------------------------- routing

    def availability(self, capability: str, language: str) -> Availability:
        """Whether *capability* can serve *language* on this install."""

        try:
            tag = normalize_language(language)
        except ManifestError:
            return Availability(capability, language, False, UNKNOWN_LANGUAGE)
        if self.activation.component_for(capability, tag) is not None:
            return Availability(capability, tag, True, INSTALLED)

        known = self.coverage.get(tag)
        if known is None:
            return Availability(capability, tag, False, UNKNOWN_LANGUAGE)
        if not known.offers(capability):
            # The corpus itself has nothing here; installing more cannot help.
            return Availability(capability, tag, False, NOT_AVAILABLE_UPSTREAM)
        if self.activation.component_for("lexical", tag) is None:
            return Availability(capability, tag, False, LANGUAGE_NOT_INSTALLED)
        return Availability(capability, tag, False, CAPABILITY_NOT_INSTALLED)

    def installed_languages(self, capability: str) -> tuple[str, ...]:
        return self.activation.installed_languages(capability)

    def connection_for(self, capability: str, language: str) -> sqlite3.Connection | None:
        """The pack serving *language*, or None when it is not installed.

        None is a routing answer, not an error. Callers translate it through
        :meth:`availability` into a response that names the reason.
        """

        component = self.activation.component_for(capability, normalize_language(language))
        if component is None:
            return None
        return self._connect(component)

    # ----------------------------------------------------------- connections

    def _connect(self, component: ActivationComponent) -> sqlite3.Connection:
        with self._lock:
            existing = self._open.get(component.sha256)
            if existing is not None:
                self._open.move_to_end(component.sha256)
                return existing
            path = self.store.open_path(component.sha256)
            connection = self._open_read_only(path)
            self._open[component.sha256] = connection
            while len(self._open) > self.max_open_packs:
                _digest, evicted = self._open.popitem(last=False)
                evicted.close()
            return connection

    @staticmethod
    def _open_read_only(path: Path) -> sqlite3.Connection:
        # Store objects are immutable by construction -- their name is their
        # content hash -- so immutable=1 is accurate and skips locking entirely.
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        # Callers index rows by column name, exactly as they do against a
        # monolith, so a pack connection must be configured the same way.
        connection.row_factory = sqlite3.Row
        return connection

    @property
    def open_pack_count(self) -> int:
        with self._lock:
            return len(self._open)

    def close(self) -> None:
        with self._lock:
            while self._open:
                _digest, connection = self._open.popitem()
                connection.close()
            self._coverage = None

    def __enter__(self) -> PackRouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
