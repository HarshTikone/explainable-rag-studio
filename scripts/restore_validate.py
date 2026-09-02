from __future__ import annotations

import argparse
import json

from backend.config import SETTINGS
from backend.platform_operations import restore_and_validate
from backend.platform_runtime import get_platform_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("object_key")
    parser.add_argument("--restore-url", required=True)
    args = parser.parse_args()
    runtime = get_platform_runtime(required=True)
    dump = runtime.object_store.get("org_platform", args.object_key)
    print(json.dumps(restore_and_validate(SETTINGS.database_url, args.restore_url, dump), indent=2))


if __name__ == "__main__":
    main()
