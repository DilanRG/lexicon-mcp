"""Cross-platform advisory installation lock."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


class LockBusyError(RuntimeError):
    """Another dataset mutation currently owns the installation lock."""


class InstallationLock:
    """Hold a one-byte non-blocking advisory lock until the file is closed.

    Unlike a lock implemented with ``O_EXCL``, an OS advisory lock is released
    after crashes and cannot leave a stale sentinel that needs unsafe PID
    guessing.  The implementation works with Windows ``msvcrt`` and POSIX
    ``flock``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: object | None = None

    def __enter__(self) -> InstallationLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        try:
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                file.write(b"\0")
                file.flush()
            file.seek(0)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise LockBusyError("another lexicon-data operation is running") from exc
            else:
                import fcntl  # type: ignore[attr-defined,unused-ignore]

                try:
                    fcntl.flock(  # type: ignore[attr-defined,unused-ignore]
                        file.fileno(),
                        fcntl.LOCK_EX  # type: ignore[attr-defined,unused-ignore]
                        | fcntl.LOCK_NB,  # type: ignore[attr-defined,unused-ignore]
                    )
                except OSError as exc:
                    raise LockBusyError("another lexicon-data operation is running") from exc
        except BaseException:
            file.close()
            raise
        self._file = file
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        file = self._file
        self._file = None
        if file is None:
            return
        try:
            file.seek(0)  # type: ignore[attr-defined]
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl  # type: ignore[attr-defined,unused-ignore]

                fcntl.flock(  # type: ignore[attr-defined,unused-ignore]
                    file.fileno(),  # type: ignore[attr-defined]
                    fcntl.LOCK_UN,  # type: ignore[attr-defined,unused-ignore]
                )
        finally:
            file.close()  # type: ignore[attr-defined]
