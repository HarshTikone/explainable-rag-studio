"""Fail when known private artifacts remain reachable in Git history."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

MANIFEST = Path("security/history-rewrite-manifest.json")


def reachable_paths() -> set[str]:
    output = subprocess.run(["git", "log", "--all", "--name-only", "--pretty=format:"], check=True, capture_output=True, text=True).stdout
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def main() -> None:
    targets = set(json.loads(MANIFEST.read_text("utf-8"))["paths"])
    found = sorted(targets & reachable_paths())
    if found:
        raise SystemExit("Private artifacts remain in Git history: " + ", ".join(found))
    print("No manifest-listed private artifacts are reachable in Git history.")


if __name__ == "__main__":
    main()
