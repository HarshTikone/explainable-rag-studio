"""Verify or export the redacted tamper-evident audit journal."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.security import SecurityRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify"])
    args = parser.parse_args()
    registry = SecurityRegistry(SETTINGS.security_db_path, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key)
    if args.action == "verify":
        print(json.dumps(registry.verify_audit_chain(), indent=2))


if __name__ == "__main__":
    main()
