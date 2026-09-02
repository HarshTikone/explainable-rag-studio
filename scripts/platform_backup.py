from __future__ import annotations

import argparse
import json

from backend.config import SETTINGS
from backend.platform_operations import create_encrypted_backup
from backend.platform_runtime import get_platform_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    runtime = get_platform_runtime(required=True)
    print(json.dumps(create_encrypted_backup(SETTINGS.database_url, runtime.object_store), indent=2))


if __name__ == "__main__":
    main()
