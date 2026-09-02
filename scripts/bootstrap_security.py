"""Create the first protected organization and display its API key once."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.security import SecurityRegistry
from backend.platform_runtime import get_platform_runtime
from backend.postgres_security import PostgresSecurityRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organization", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Owner")
    args = parser.parse_args()
    platform = get_platform_runtime()
    registry = (PostgresSecurityRegistry(platform.database, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key)
                if platform else SecurityRegistry(SETTINGS.security_db_path, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key))
    result = registry.bootstrap(args.organization, args.email, args.name)
    result["warning"] = "Store the API key now. It cannot be displayed again."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
