from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import SETTINGS
from backend.database import DatabaseRuntime
from backend.platform_migration import PlatformMigrator
from backend.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run, execute, or validate the PostgreSQL/pgvector cutover.")
    parser.add_argument("mode", choices=("plan", "execute", "validate"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--mapping")
    parser.add_argument("--plan", default="outputs/platform-migration.json")
    args = parser.parse_args()
    database = DatabaseRuntime(SETTINGS.database_url)
    migrator = PlatformMigrator(database, args.root)
    mapping = json.loads(Path(args.mapping).read_text("utf-8")) if args.mapping else {}
    if args.mode == "plan":
        result = migrator.plan(mapping)
    else:
        existing = json.loads(Path(args.plan).read_text("utf-8"))
        result = migrator.execute(existing) if args.mode == "execute" else migrator.validate(existing)
    write_json(args.plan, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
