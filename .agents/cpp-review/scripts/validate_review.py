#!/usr/bin/env python3
"""Validate cpp-inference-code-review report structure and optional spec-linked path."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REQUIRED_SECTIONS = [
    "## Review metadata",
    "## 1. Contract & correctness",
    "## 2. C++/GPU stability",
    "## 3. Backend architecture & simplicity",
    "## 4. Numerical correctness & tests",
    "## 5. Performance",
    "## 6. Synthesis / overall assessment",
]

REQUIRED_FIELDS = [
    "Severity",
    "Verification",
    "Confidence",
    "Scope relation",
    "Backend scope",
    "Location",
    "Invariant",
    "Failure mode",
    "Evidence",
    "Impact",
    "Recommended fix",
    "Verification method",
]

FINDING_RE = re.compile(r"^###\s+(CC|ST|AR|NT|PF)-\d{3}\s+—\s+.+$", re.M)


def validate_spec_path(review_file: Path, spec_dir: Path) -> list[str]:
    errors = []
    spec = spec_dir.resolve()
    review = review_file.resolve()
    parts = spec.parts
    try:
        idx = parts.index("docs")
    except ValueError:
        return ["spec-dir must be beneath docs/changes/"]
    if idx + 1 >= len(parts) or parts[idx + 1] != "changes":
        errors.append("spec-dir must be beneath docs/changes/")
    if review != spec / "review.md":
        errors.append(f"review-file must be exactly {spec / 'review.md'}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-file", required=True)
    ap.add_argument("--spec-dir")
    args = ap.parse_args()

    path = Path(args.review_file)
    if not path.is_file():
        print(f"review-error: file not found: {path}", file=sys.stderr)
        return 2
    text = path.read_text(encoding="utf-8")
    errors = []

    last = -1
    for heading in REQUIRED_SECTIONS:
        pos = text.find(heading)
        if pos < 0:
            errors.append(f"missing section: {heading}")
        elif pos < last:
            errors.append(f"section out of order: {heading}")
        else:
            last = pos

    if args.spec_dir:
        errors.extend(validate_spec_path(path, Path(args.spec_dir)))

    matches = list(FINDING_RE.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        finding_id = match.group(0).split()[1]
        for field in REQUIRED_FIELDS:
            if f"**{field}:**" not in block:
                errors.append(f"{finding_id}: missing field {field}")
        conf = re.search(r"\*\*Confidence:\*\*\s*(\d{1,3})", block)
        if conf and not (0 <= int(conf.group(1)) <= 100):
            errors.append(f"{finding_id}: confidence must be 0-100")

    if errors:
        for error in errors:
            print(f"review-error: {error}", file=sys.stderr)
        return 2

    print(f"review-ok: {path} ({len(matches)} finding(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
