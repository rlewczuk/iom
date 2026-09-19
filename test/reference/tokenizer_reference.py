#!/usr/bin/env python3
"""Offline provenance checks for the TinyLlama tokenizer reference.

This module deliberately has no tokenizer dependency.  It validates the
caller-selected distribution and the reference environment before any later
reference tool is allowed to load or measure tokenizer behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


# Keep these values as the one authority shared by later reference tools.
PINNED_RUNTIME: dict[str, str] = {
    "implementation": "CPython",
    "version": "3.11.16",
}

# Keep these values as the one authority shared by later reference tools.
PINNED_PACKAGES: dict[str, str] = {
    "jinja2": "3.1.2",
    "sentencepiece": "0.1.99",
    "tokenizers": "0.14.1",
    "transformers": "4.35.0",
}

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

_MANIFEST_ARTIFACT_FIELDS = ("name", "sha256", "size")
_SHA256_LENGTH = hashlib.sha256().digest_size * 2


class ProvenanceError(RuntimeError):
    """A contextual, expected failure while establishing reference identity."""


def _actual_runtime() -> dict[str, str]:
    implementation = getattr(sys.implementation, "name", "<unknown>")
    if implementation != "cpython":
        raise ProvenanceError(
            "reference runtime requires CPython; expected CPython, "
            f"actual {implementation}"
        )

    version_info = sys.version_info
    try:
        version = ".".join(
            str(component)
            for component in (
                version_info.major,
                version_info.minor,
                version_info.micro,
            )
        )
    except (AttributeError, TypeError) as error:
        raise ProvenanceError(
            "reference runtime has no usable CPython major.minor.patch version; "
            f"actual {version_info!r}"
        ) from error

    return {"implementation": "CPython", "version": version}


def check_runtime(expected: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the active CPython identity and compare all applicable pins."""

    actual = _actual_runtime()
    if actual != PINNED_RUNTIME:
        raise ProvenanceError(
            "reference runtime mismatch: expected CPython "
            f"{PINNED_RUNTIME['version']}, actual "
            f"{actual['implementation']} {actual['version']}"
        )
    if expected is None:
        return actual

    if not isinstance(expected, Mapping):
        raise ProvenanceError(
            "reference runtime pin is malformed; expected an object, "
            f"actual {type(expected).__name__}"
        )

    expected_implementation = expected.get("implementation")
    expected_version = expected.get("version")
    if expected_implementation != actual["implementation"]:
        raise ProvenanceError(
            "reference runtime mismatch: expected implementation "
            f"{expected_implementation!r}, actual {actual['implementation']!r}"
        )
    if expected_version != actual["version"]:
        raise ProvenanceError(
            "reference runtime mismatch: expected CPython "
            f"{expected_version!r}, actual {actual['version']!r}"
        )
    return actual


def check_packages(
    expected: Mapping[str, str] = PINNED_PACKAGES,
) -> dict[str, str]:
    """Read package metadata without importing any reference package."""

    if not isinstance(expected, Mapping):
        raise ProvenanceError(
            "reference package pin is malformed; expected an object, "
            f"actual {type(expected).__name__}"
        )

    actual: dict[str, str] = {}
    for package_name in sorted(PINNED_PACKAGES):
        expected_version = expected.get(package_name)
        if expected_version is None:
            raise ProvenanceError(
                "reference package pin is incomplete: missing "
                f"{package_name!r}"
            )
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            actual_version = "<missing>"
        except Exception as error:
            raise ProvenanceError(
                "cannot inspect reference package "
                f"{package_name!r}: expected {expected_version!r}; "
                f"metadata lookup failed: {error}"
            ) from error

        actual[package_name] = actual_version
        if actual_version != expected_version:
            raise ProvenanceError(
                "reference package mismatch for "
                f"{package_name!r}: expected {expected_version!r}, "
                f"actual {actual_version!r}"
            )

    return actual


def check_environment(
    expected_runtime: Mapping[str, Any] | None = None,
    expected_packages: Mapping[str, str] = PINNED_PACKAGES,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate runtime and package pins before inspecting reference behavior."""

    runtime = check_runtime(expected_runtime)
    packages = check_packages(expected_packages)
    return runtime, packages


def _read_regular_artifact(path: Path, name: str) -> dict[str, Any]:
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise ProvenanceError(
            f"cannot inspect required artifact {name!r} at {path}: {error}"
        ) from error

    if not stat.S_ISREG(file_stat.st_mode):
        if stat.S_ISLNK(file_stat.st_mode):
            actual_kind = "a symbolic link"
        elif stat.S_ISDIR(file_stat.st_mode):
            actual_kind = "a directory"
        else:
            actual_kind = "a non-regular file"
        raise ProvenanceError(
            f"required artifact {name!r} at {path} must be a regular file; "
            f"actual {actual_kind}"
        )

    digest = hashlib.sha256()
    byte_count = 0
    try:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, os.O_RDONLY | no_follow)
        with os.fdopen(descriptor, "rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ProvenanceError(
                    f"required artifact {name!r} at {path} changed to a "
                    "non-regular file while opening"
                )
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
            final_stat = os.fstat(stream.fileno())
    except ProvenanceError:
        raise
    except OSError as error:
        raise ProvenanceError(
            f"cannot read required artifact {name!r} at {path}: {error}"
        ) from error

    if byte_count != final_stat.st_size:
        raise ProvenanceError(
            f"required artifact {name!r} at {path} changed while reading; "
            f"expected {final_stat.st_size} bytes, read {byte_count}"
        )

    return {
        "name": name,
        "sha256": digest.hexdigest(),
        "size": byte_count,
    }


def _validate_artifact_records(
    artifacts: Any,
    source: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping):
        raise ProvenanceError(
            f"{source} artifacts field is malformed; expected an object, "
            f"actual {type(artifacts).__name__}"
        )

    actual_names = set(artifacts)
    required_names = set(REQUIRED_ARTIFACTS)
    if actual_names != required_names:
        missing = sorted(required_names - actual_names)
        extra = sorted(actual_names - required_names)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if extra:
            details.append(f"unexpected {extra!r}")
        raise ProvenanceError(
            f"{source} artifacts must contain exactly the five required files; "
            + "; ".join(details)
        )

    normalized: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_ARTIFACTS:
        record = artifacts[name]
        if not isinstance(record, Mapping):
            raise ProvenanceError(
                f"{source} artifact {name!r} is malformed; expected an object, "
                f"actual {type(record).__name__}"
            )
        for field in _MANIFEST_ARTIFACT_FIELDS:
            if field not in record:
                raise ProvenanceError(
                    f"{source} artifact {name!r} is missing field {field!r}"
                )

        if record["name"] != name:
            raise ProvenanceError(
                f"{source} artifact key {name!r} has name "
                f"{record['name']!r}; expected {name!r}"
            )
        size = record["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProvenanceError(
                f"{source} artifact {name!r} has invalid size {size!r}; "
                "expected a non-negative integer"
            )
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProvenanceError(
                f"{source} artifact {name!r} has invalid sha256 {digest!r}; "
                "expected a lowercase SHA-256 digest"
            )
        normalized[name] = {
            "name": name,
            "sha256": digest,
            "size": size,
        }
    return normalized


def collect_artifacts(
    model_dir: os.PathLike[str] | str,
    expected: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash the five caller-owned files without discovering another model."""

    directory = Path(model_dir)
    try:
        directory_stat = directory.stat()
    except OSError as error:
        raise ProvenanceError(
            f"cannot inspect caller-supplied model directory {directory}: {error}"
        ) from error
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ProvenanceError(
            f"caller-supplied model path {directory} must be a directory; "
            "actual non-directory"
        )

    records: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_ARTIFACTS:
        records[name] = _read_regular_artifact(directory / name, name)

    if expected is not None:
        expected_records = _validate_artifact_records(expected, "expected")
        for name in REQUIRED_ARTIFACTS:
            expected_record = expected_records[name]
            actual_record = records[name]
            if actual_record["size"] != expected_record["size"]:
                raise ProvenanceError(
                    f"artifact identity mismatch for {name!r}: expected size "
                    f"{expected_record['size']}, actual {actual_record['size']}"
                )
            if actual_record["sha256"] != expected_record["sha256"]:
                raise ProvenanceError(
                    f"artifact identity mismatch for {name!r}: expected SHA-256 "
                    f"{expected_record['sha256']}, actual "
                    f"{actual_record['sha256']}"
                )

    return records


def _load_json(path: Path, source: str) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProvenanceError(f"cannot read {source} {path}: {error}") from error
    try:
        text = raw.decode("utf-8")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"{source} {path} is not valid UTF-8 JSON: {error}") from error


def _validate_generation(generation: Any, source: str) -> dict[str, Any]:
    if not isinstance(generation, Mapping):
        raise ProvenanceError(
            f"{source} generation field is malformed; expected an object, "
            f"actual {type(generation).__name__}"
        )
    command = generation.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(argument, str) or not argument for argument in command)
    ):
        raise ProvenanceError(
            f"{source} generation command is malformed; expected a non-empty "
            "list of strings"
        )
    return {"command": list(command)}


def validate_manifest(manifest: Any, source: str = "manifest") -> dict[str, Any]:
    """Validate and normalize the stable manifest schema."""

    if not isinstance(manifest, Mapping):
        raise ProvenanceError(
            f"{source} is malformed; expected an object, "
            f"actual {type(manifest).__name__}"
        )

    for field in ("artifacts", "generation", "packages", "runtime"):
        if field not in manifest:
            raise ProvenanceError(f"{source} is missing required field {field!r}")

    runtime = manifest["runtime"]
    if not isinstance(runtime, Mapping):
        raise ProvenanceError(
            f"{source} runtime field is malformed; expected an object, "
            f"actual {type(runtime).__name__}"
        )
    if runtime.get("implementation") != "CPython":
        raise ProvenanceError(
            f"{source} runtime implementation must be 'CPython'; "
            f"actual {runtime.get('implementation')!r}"
        )
    version = runtime.get("version")
    if (
        not isinstance(version, str)
        or len(version.split(".")) != 3
        or any(not component.isdigit() for component in version.split("."))
    ):
        raise ProvenanceError(
            f"{source} runtime version is malformed; expected major.minor.patch, "
            f"actual {version!r}"
        )

    packages = manifest["packages"]
    if not isinstance(packages, Mapping):
        raise ProvenanceError(
            f"{source} packages field is malformed; expected an object, "
            f"actual {type(packages).__name__}"
        )
    if set(packages) != set(PINNED_PACKAGES):
        raise ProvenanceError(
            f"{source} packages must contain exactly the pinned package set; "
            f"actual {sorted(packages)!r}"
        )
    normalized_packages: dict[str, str] = {}
    for package_name in sorted(PINNED_PACKAGES):
        package_version = packages[package_name]
        if not isinstance(package_version, str) or not package_version:
            raise ProvenanceError(
                f"{source} package {package_name!r} has invalid version "
                f"{package_version!r}"
            )
        if package_version != PINNED_PACKAGES[package_name]:
            raise ProvenanceError(
                f"{source} package {package_name!r} is not pinned to "
                f"{PINNED_PACKAGES[package_name]!r}; actual {package_version!r}"
            )
        normalized_packages[package_name] = package_version

    artifacts = _validate_artifact_records(manifest["artifacts"], source)
    generation = _validate_generation(manifest["generation"], source)
    return {
        "artifacts": artifacts,
        "generation": generation,
        "packages": normalized_packages,
        "runtime": {
            "implementation": "CPython",
            "version": version,
        },
    }


def load_manifest(path: os.PathLike[str] | str, source: str = "manifest") -> dict[str, Any]:
    """Load and validate an existing expected or destination manifest."""

    manifest_path = Path(path)
    return validate_manifest(_load_json(manifest_path, source), f"{source} {manifest_path}")


def _compare_identity(
    expected: Mapping[str, Any],
    actual_runtime: Mapping[str, str],
    actual_packages: Mapping[str, str],
    actual_artifacts: Mapping[str, Mapping[str, Any]],
    source: str,
) -> None:
    expected_runtime = expected["runtime"]
    if expected_runtime != actual_runtime:
        raise ProvenanceError(
            f"reference runtime mismatch against {source}: expected "
            f"{expected_runtime!r}, actual {dict(actual_runtime)!r}"
        )

    expected_packages = expected["packages"]
    if dict(expected_packages) != dict(actual_packages):
        raise ProvenanceError(
            f"reference package identity mismatch against {source}: expected "
            f"{dict(expected_packages)!r}, actual {dict(actual_packages)!r}"
        )

    expected_artifacts = expected["artifacts"]
    for name in REQUIRED_ARTIFACTS:
        expected_record = expected_artifacts[name]
        actual_record = actual_artifacts[name]
        if expected_record["size"] != actual_record["size"]:
            raise ProvenanceError(
                f"artifact identity mismatch against {source} for {name!r}: "
                f"expected size {expected_record['size']}, "
                f"actual {actual_record['size']}"
            )
        if expected_record["sha256"] != actual_record["sha256"]:
            raise ProvenanceError(
                f"artifact identity mismatch against {source} for {name!r}: "
                f"expected SHA-256 {expected_record['sha256']}, "
                f"actual {actual_record['sha256']}"
            )


def _canonical_command() -> list[str]:
    return [
        "python3",
        "test/reference/tokenizer_reference.py",
        "manifest",
        "--model-dir",
        "<MODEL_DIR>",
        "--output",
        "<OUTPUT>",
    ]


def build_manifest(
    model_dir: os.PathLike[str] | str,
    output: os.PathLike[str] | str | None = None,
    expected_manifest: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Validate all inputs and construct a complete deterministic manifest."""

    expected: dict[str, Any] | None = None
    expected_source = "caller-requested expected manifest"
    if expected_manifest is not None:
        expected = load_manifest(expected_manifest, expected_source)

    runtime, packages = check_environment(
        expected_runtime=expected["runtime"] if expected is not None else None,
        expected_packages=expected["packages"] if expected is not None else PINNED_PACKAGES,
    )
    expected_artifacts = expected["artifacts"] if expected is not None else None
    artifacts = collect_artifacts(model_dir, expected_artifacts)

    destination = (
        Path(output)
        if output is not None
        else Path(__file__).resolve().with_name("tokenizer_reference_manifest.json")
    )
    model_directory = Path(model_dir)
    try:
        destination_resolved = destination.resolve(strict=False)
        for name in REQUIRED_ARTIFACTS:
            artifact_resolved = (model_directory / name).resolve(strict=False)
            if destination_resolved == artifact_resolved:
                raise ProvenanceError(
                    f"manifest output {destination} must not replace required "
                    f"artifact {name!r} in caller-supplied model directory"
                )
    except OSError as error:
        raise ProvenanceError(
            f"cannot resolve manifest output {destination}: {error}"
        ) from error

    command = _canonical_command()
    manifest = {
        "artifacts": artifacts,
        "generation": {"command": command},
        "packages": packages,
        "runtime": runtime,
    }

    destination_exists = destination.exists() or destination.is_symlink()
    if destination_exists:
        existing = load_manifest(destination, "existing manifest")
        _compare_identity(
            existing,
            runtime,
            packages,
            artifacts,
            f"existing manifest {destination}",
        )
        if existing["generation"] != manifest["generation"]:
            raise ProvenanceError(
                f"generation command mismatch against existing manifest {destination}: "
                f"expected {existing['generation']!r}, "
                f"actual {manifest['generation']!r}"
            )

    if expected is not None:
        if expected["generation"] != manifest["generation"]:
            raise ProvenanceError(
                f"generation command mismatch against {expected_source}: expected "
                f"{expected['generation']!r}, actual {manifest['generation']!r}"
            )

    return manifest


def serialize_manifest(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a validated manifest with the repository's stable JSON policy."""

    normalized = validate_manifest(manifest)
    try:
        return (
            json.dumps(
                normalized,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ProvenanceError(f"cannot serialize provenance manifest: {error}") from error


def publish_manifest(
    manifest: Mapping[str, Any],
    output: os.PathLike[str] | str,
) -> None:
    """Atomically publish a complete manifest to *output*."""

    output_path = Path(output)
    payload = serialize_manifest(manifest)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    except ProvenanceError:
        raise
    except OSError as error:
        raise ProvenanceError(
            f"cannot atomically publish provenance manifest {output_path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an offline TinyLlama tokenizer provenance manifest."
    )
    parser.add_argument(
        "operation",
        choices=("manifest",),
        help="offline provenance operation to run",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        metavar="PATH",
        help="caller-supplied local tokenizer distribution directory",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="manifest destination (defaults beside this script)",
    )
    parser.add_argument(
        "--expected-manifest",
        metavar="PATH",
        help="existing manifest whose runtime, package, and artifact identity must match",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = (
            Path(args.output)
            if args.output is not None
            else Path(__file__).resolve().with_name("tokenizer_reference_manifest.json")
        )
        manifest = build_manifest(
            args.model_dir,
            output=destination,
            expected_manifest=args.expected_manifest,
        )
        publish_manifest(manifest, destination)
    except ProvenanceError as error:
        print(f"tokenizer reference provenance: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
