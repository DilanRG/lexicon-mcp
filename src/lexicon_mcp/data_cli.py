"""Command-line dataset administration for Lexicon MCP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from lexicon_mcp.data.activation import ActivationError
from lexicon_mcp.data.component_lifecycle import ComponentLifecycle
from lexicon_mcp.data.integrity import IntegrityError
from lexicon_mcp.data.lifecycle import (
    DatasetLifecycle,
    DownloadError,
    LifecycleError,
    SpaceError,
    VerificationError,
)
from lexicon_mcp.data.locking import LockBusyError
from lexicon_mcp.data.manifest import ManifestError
from lexicon_mcp.data.selection import REQUESTABLE_CAPABILITIES, SelectionError
from lexicon_mcp.data.store import StoreError

DEFAULT_MANIFEST = (
    "https://github.com/DilanRG/lexicon-mcp/releases/download/{version}/manifest.json"
)
DATASET_PROFILES = ("full", "english")


def _language_list(value: str) -> list[str]:
    """Parse a comma-separated language list, rejecting empty entries."""

    tags = [item.strip() for item in value.split(",")]
    if not tags or any(not tag for tag in tags):
        raise argparse.ArgumentTypeError("language list must be comma-separated tags")
    return tags


def _capability_list(value: str) -> list[str]:
    capabilities = [item.strip() for item in value.split(",")]
    unknown = [item for item in capabilities if item not in REQUESTABLE_CAPABILITIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown capabilities {unknown}; expected from {list(REQUESTABLE_CAPABILITIES)}"
        )
    return capabilities


def _manifest_source(value: str | None, *, version: str, profile: str) -> str:
    template = value or os.environ.get("LEXICON_MANIFEST_URL") or DEFAULT_MANIFEST
    if "{" not in template:
        # A plain URL or filesystem path is used verbatim.  Windows paths and
        # mirror directories must never be reinterpreted as format fields.
        return template
    try:
        return template.format(version=version, profile=profile)
    except (KeyError, IndexError, ValueError) as exc:
        raise LifecycleError(f"invalid manifest URL template: {exc}") from exc


def _resolved_source(args: argparse.Namespace) -> str | None:
    """Return the requested source, honoring the deprecated alias."""

    source: str | None = getattr(args, "source", None)
    legacy: str | None = getattr(args, "manifest_url", None)
    if source and legacy:
        raise LifecycleError("--from and --manifest-url cannot be combined; use --from")
    if legacy:
        print(
            "warning: --manifest-url is deprecated and will be removed in 2.0.0; use --from",
            file=sys.stderr,
        )
        return legacy
    return source


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the schema-2 language and capability selectors."""

    languages = parser.add_mutually_exclusive_group()
    languages.add_argument(
        "--languages",
        type=_language_list,
        help="comma-separated language tags to install, e.g. en,fr,de",
    )
    languages.add_argument(
        "--all-languages",
        action="store_true",
        help="install every language the release carries",
    )
    parser.add_argument(
        "--capabilities",
        type=_capability_list,
        default=["lexical"],
        help=(
            "comma-separated capabilities to install "
            f"(default: lexical; available: {','.join(REQUESTABLE_CAPABILITIES)})"
        ),
    )
    parser.add_argument(
        "--allow-missing-languages",
        action="store_true",
        help="report languages the release does not carry instead of failing",
    )


def _requested_languages(args: argparse.Namespace) -> list[str] | None:
    """Return the requested tags, or None meaning every language."""

    if getattr(args, "all_languages", False):
        return None
    return getattr(args, "languages", None)


def _add_source_arguments(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--from", dest="source", help=help_text)
    parser.add_argument(
        "--manifest-url",
        dest="manifest_url",
        help="deprecated alias for --from (removed in 2.0.0)",
    )


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lexicon-data",
        description="Install and verify immutable offline Lexicon MCP datasets.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="dataset root (defaults to LEXICON_DATA_DIR or the platform data directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    install = commands.add_parser("install", help="install and atomically activate a release")
    install.add_argument(
        "--profile",
        choices=DATASET_PROFILES,
        help="schema 1 releases only; schema 2 selects languages and capabilities",
    )
    install.add_argument("--version", required=True)
    _add_selection_arguments(install)
    _add_source_arguments(
        install,
        help_text=(
            "install source: a mirror directory produced by 'fetch', a manifest.json "
            "path, an HTTP(S) URL, or a template containing {version} and {profile}. "
            "A local source installs with no network access at all."
        ),
    )

    fetch = commands.add_parser(
        "fetch",
        help="mirror a release to a local directory without installing it",
    )
    fetch.add_argument("--profile", choices=DATASET_PROFILES, required=True)
    fetch.add_argument("--version", required=True)
    fetch.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="destination directory; install it later with 'install --from <dest>'",
    )
    _add_source_arguments(
        fetch,
        help_text="release source (defaults to the published immutable release)",
    )

    add_language = commands.add_parser(
        "add-language",
        help="widen the active selection, fetching only what is not already held",
    )
    add_language.add_argument("--version", required=True)
    add_language.add_argument("--languages", type=_language_list, required=True)
    add_language.add_argument("--capabilities", type=_capability_list)
    add_language.add_argument("--allow-missing-languages", action="store_true")
    _add_source_arguments(add_language, help_text="release source")

    remove_language = commands.add_parser(
        "remove-language",
        help="narrow the active selection; components stay for instant rollback",
    )
    remove_language.add_argument("--version", required=True)
    remove_language.add_argument("--languages", type=_language_list, required=True)
    remove_language.add_argument("--capabilities", type=_capability_list)
    _add_source_arguments(remove_language, help_text="release source")

    activate = commands.add_parser(
        "activate", help="switch to a previously installed selection"
    )
    activate.add_argument("--activation", required=True, help="activation id from status")

    forget = commands.add_parser(
        "forget", help="delete a retained activation record, freeing it for prune"
    )
    forget.add_argument("--activation", required=True)

    commands.add_parser(
        "prune", help="delete stored components no retained activation references"
    )

    commands.add_parser("status", help="show active and retained versions")

    verify = commands.add_parser("verify", help="verify all artifacts in an installed version")
    verify.add_argument("--version", help="installed version (defaults to active)")

    repair = commands.add_parser("repair", help="redownload only damaged components")
    repair.add_argument("--version", help="installed version (defaults to active)")
    repair.add_argument(
        "--profile",
        choices=DATASET_PROFILES,
        help="installed profile (normally detected from activation/manifest metadata)",
    )
    _add_source_arguments(
        repair,
        help_text="optional pinned source if the installed manifest is damaged",
    )

    rollback = commands.add_parser("rollback", help="activate the retained previous version")
    rollback.add_argument("--version", help="explicit retained version (defaults to previous)")
    return parser


def _is_component_install(data_dir: Path | None, lifecycle: DatasetLifecycle) -> bool:
    """True when the data root holds a schema-2 activation pointer."""

    pointer = (Path(data_dir) if data_dir is not None else lifecycle.root) / "current.json"
    if not pointer.is_file():
        # Nothing installed yet: report through the schema-2 view, which is what
        # any new install will produce.
        return True
    try:
        value = json.loads(pointer.read_bytes())
    except (OSError, ValueError):
        return False
    return isinstance(value, dict) and value.get("schema_version") == 2


def _repair_profile(
    lifecycle: DatasetLifecycle, *, version: str | None, requested: str | None
) -> str:
    if requested is not None:
        return requested
    status = lifecycle.status()
    current = status.get("current")
    resolved_version = version
    if resolved_version is None and isinstance(current, dict):
        current_version = current.get("version")
        if isinstance(current_version, str):
            resolved_version = current_version
        current_profile = current.get("profile")
        if isinstance(current_profile, str) and current_profile in DATASET_PROFILES:
            return current_profile
    for installed in status.get("installed_versions", []):
        if not isinstance(installed, dict) or installed.get("version") != resolved_version:
            continue
        profile = installed.get("profile")
        if isinstance(profile, str) and profile in DATASET_PROFILES:
            return profile
    # Activation records created before profile support can only refer to the
    # original full release profile.
    return "full"


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    lifecycle = DatasetLifecycle(args.data_dir)
    if args.command == "install":
        source = _manifest_source(
            _resolved_source(args),
            version=args.version,
            profile=args.profile or "full",
        )
        manifest = lifecycle.load_manifest(source)
        if manifest.schema_version >= 2:
            if args.profile is not None:
                raise LifecycleError(
                    "--profile applies to schema 1 releases only; "
                    "select --languages or --all-languages instead"
                )
            if _requested_languages(args) is None and not args.all_languages:
                raise LifecycleError(
                    "specify --languages, or --all-languages to install everything"
                )
            components = ComponentLifecycle(args.data_dir, fetcher=lifecycle)
            # Pass the source, not the parsed manifest: a local source resolves
            # its sibling assets from the manifest's own path, and installing
            # from a preloaded object would silently reach for the network.
            return (
                components.install(
                    source,
                    languages=_requested_languages(args),
                    capabilities=args.capabilities,
                    strict=not args.allow_missing_languages,
                ),
                0,
            )
        if args.profile is None:
            raise LifecycleError("schema 1 releases require --profile")
        if args.languages or args.all_languages:
            raise LifecycleError(
                "language selection requires a schema 2 release; "
                f"{args.version} is schema {manifest.schema_version}"
            )
        return (
            lifecycle.install(source, profile=args.profile, version=args.version),
            0,
        )
    if args.command in {"add-language", "remove-language"}:
        source = _manifest_source(
            _resolved_source(args), version=args.version, profile="full"
        )
        components = ComponentLifecycle(args.data_dir, fetcher=lifecycle)
        if args.command == "add-language":
            return (
                components.add_languages(
                    source,
                    languages=args.languages,
                    capabilities=args.capabilities,
                    strict=not args.allow_missing_languages,
                ),
                0,
            )
        return (
            components.remove_languages(
                source, languages=args.languages, capabilities=args.capabilities
            ),
            0,
        )
    if args.command == "activate":
        return ComponentLifecycle(args.data_dir, fetcher=lifecycle).activate(
            args.activation
        ), 0
    if args.command == "forget":
        return ComponentLifecycle(args.data_dir, fetcher=lifecycle).forget(
            args.activation
        ), 0
    if args.command == "prune":
        return ComponentLifecycle(args.data_dir, fetcher=lifecycle).prune(), 0
    if args.command == "fetch":
        source = _manifest_source(
            _resolved_source(args),
            version=args.version,
            profile=args.profile,
        )
        return (
            lifecycle.fetch(
                source,
                profile=args.profile,
                version=args.version,
                dest=args.dest,
            ),
            0,
        )
    if args.command == "status":
        if _is_component_install(args.data_dir, lifecycle):
            result = ComponentLifecycle(args.data_dir, fetcher=lifecycle).status()
        else:
            result = lifecycle.status()
        return result, 1 if result.get("current_error") else 0
    if args.command == "verify":
        if _is_component_install(args.data_dir, lifecycle):
            result = ComponentLifecycle(args.data_dir, fetcher=lifecycle).verify(args.version)
        else:
            result = lifecycle.verify(args.version)
        return result, 0 if result["ok"] else 1
    if args.command == "repair":
        requested_source = _resolved_source(args)
        repair_source: str | None = None
        if requested_source:
            version: str | None = args.version
            if version is None:
                status = lifecycle.status()
                current = status.get("current")
                if not isinstance(current, dict) or not isinstance(current.get("version"), str):
                    raise LifecycleError("cannot resolve active version for repair manifest")
                version = current["version"]
            profile = _repair_profile(
                lifecycle,
                version=version,
                requested=args.profile,
            )
            repair_source = _manifest_source(requested_source, version=version, profile=profile)
        return lifecycle.repair(version=args.version, manifest_source=repair_source), 0
    if args.command == "rollback":
        return lifecycle.rollback(args.version), 0
    raise LifecycleError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result, code = run(args)
    except (
        ActivationError,
        DownloadError,
        IntegrityError,
        LifecycleError,
        LockBusyError,
        ManifestError,
        SelectionError,
        SpaceError,
        StoreError,
        VerificationError,
    ) as exc:
        _emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__}, stream=sys.stderr)
        return 2
    _emit(result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
