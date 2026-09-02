"""Generate ignored local Docker secret files for the reference stack."""
from __future__ import annotations

import argparse
import base64
import os
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=".secrets")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = Path(args.directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    values = {
        "postgres_password": secrets.token_urlsafe(32),
        "rag_app_password": secrets.token_urlsafe(32),
        "keycloak_db_password": secrets.token_urlsafe(32),
        "keycloak_admin_password": secrets.token_urlsafe(32),
        "redis_password": secrets.token_urlsafe(32),
        "minio_access_key": "rag" + secrets.token_hex(8),
        "minio_secret_key": secrets.token_urlsafe(32),
        "api_key_pepper": secrets.token_urlsafe(48),
        "audit_hmac_key": secrets.token_urlsafe(48),
        "object_master_key": base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("="),
        "oidc_client_secret": secrets.token_urlsafe(32),
    }
    for name, value in values.items():
        path = target / name
        if path.exists() and not args.force:
            continue
        path.write_text(value + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    print(f"Development secrets are ready in {target}")


if __name__ == "__main__":
    main()
