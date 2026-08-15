"""Content-addressed storage for verified release components.

Components are stored by digest rather than in a per-version directory tree, so
a component shared by two installs is held once, adding a language downloads
only what is genuinely new, and rollback is a pointer swap rather than a refetch.

The published digest stays the trust boundary: nothing enters the store without
hashing to the digest the manifest declared for it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

from .integrity import sha256_file
from .manifest import is_sha256


class StoreError(RuntimeError):
    """A component could not be admitted to or read from the store."""


def _validated_digest(digest: str) -> str:
    if not is_sha256(digest):
        raise StoreError(f"not a lowercase SHA-256 digest: {digest!r}")
    return digest


class ComponentStore:
    """A directory of immutable, digest-named component files.

    Layout is ``<root>/sha256/<first two hex characters>/<digest>``.  The fan-out
    keeps any single directory small enough to stay listable when a full install
    holds a component per language.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def objects_root(self) -> Path:
        return self.root / "sha256"

    def path_for(self, digest: str) -> Path:
        digest = _validated_digest(digest)
        return self.objects_root / digest[:2] / digest

    def contains(self, digest: str) -> bool:
        """Cheap presence check: the file exists and is a regular file.

        Presence is not proof of integrity.  Callers that need certainty before
        activating call :meth:`verify`; the installer does exactly that.
        """

        path = self.path_for(digest)
        return path.is_file() and not path.is_symlink()

    def verify(self, digest: str) -> bool:
        path = self.path_for(digest)
        try:
            if not path.is_file() or path.is_symlink():
                return False
            return sha256_file(path) == digest
        except OSError:
            return False

    def open_path(self, digest: str) -> Path:
        """Return the stored path, or fail loudly if it is absent."""

        path = self.path_for(digest)
        if not path.is_file() or path.is_symlink():
            raise StoreError(f"component {digest} is not in the store")
        return path

    def adopt(self, staged: Path, digest: str) -> Path:
        """Move a staged file into the store under its verified digest.

        The file is hashed before it is admitted, so a corrupt download can
        never become a stored component.  Admission is a rename, which is atomic
        on the same volume, so a reader never observes a half-written object.
        """

        digest = _validated_digest(digest)
        staged = Path(staged)
        if not staged.is_file() or staged.is_symlink():
            raise StoreError(f"staged component is missing or unsafe: {staged}")
        actual = sha256_file(staged)
        if actual != digest:
            raise StoreError(
                f"staged component hashes to {actual}, expected {digest}"
            )
        destination = self.path_for(digest)
        if destination.is_file():
            # Already stored, and identical by construction: digests match.
            staged.unlink(missing_ok=True)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)
        return destination

    def iter_digests(self) -> Iterator[str]:
        objects = self.objects_root
        if not objects.is_dir():
            return
        for prefix in sorted(objects.iterdir()):
            if not prefix.is_dir() or prefix.is_symlink():
                continue
            for item in sorted(prefix.iterdir()):
                if not item.is_file() or item.is_symlink():
                    continue
                if is_sha256(item.name) and item.name[:2] == prefix.name:
                    yield item.name

    def prune(self, keep: set[str]) -> tuple[str, ...]:
        """Delete stored components no live activation references.

        Callers must pass the union over *every* retained activation, not just
        the active one, or rollback targets would lose their components.
        """

        for digest in keep:
            _validated_digest(digest)
        removed: list[str] = []
        for digest in list(self.iter_digests()):
            if digest in keep:
                continue
            try:
                self.path_for(digest).unlink()
            except OSError as exc:
                raise StoreError(f"cannot prune component {digest}: {exc}") from exc
            removed.append(digest)
        self._drop_empty_prefixes()
        return tuple(removed)

    def total_bytes(self) -> int:
        return sum(self.path_for(digest).stat().st_size for digest in self.iter_digests())

    def _drop_empty_prefixes(self) -> None:
        objects = self.objects_root
        if not objects.is_dir():
            return
        for prefix in sorted(objects.iterdir()):
            if prefix.is_dir() and not prefix.is_symlink() and not any(prefix.iterdir()):
                shutil.rmtree(prefix, ignore_errors=True)
