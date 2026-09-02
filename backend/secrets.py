"""Secret providers that keep production credentials out of application config."""
from __future__ import annotations

import os
from pathlib import Path


class SecretUnavailableError(RuntimeError):
    pass


class FileSecretProvider:
    """Read Docker/Kubernetes mounted secrets with strict file checks."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = dict(mapping)

    def read(self, name: str) -> bytes:
        configured = self.mapping.get(name, "")
        if not configured:
            raise SecretUnavailableError(f"Secret file is not configured: {name}")
        path = Path(configured).resolve()
        if not path.is_file():
            raise SecretUnavailableError(f"Secret file is unavailable: {name}")
        value = path.read_bytes().strip()
        if not value:
            raise SecretUnavailableError(f"Secret file is empty: {name}")
        return value


class DevelopmentSecretProvider(FileSecretProvider):
    """Permit environment fallbacks only outside required production mode."""

    def __init__(self, mapping: dict[str, str], environment_mapping: dict[str, str]):
        super().__init__(mapping)
        self.environment_mapping = environment_mapping

    def read(self, name: str) -> bytes:
        try:
            return super().read(name)
        except SecretUnavailableError:
            value = os.getenv(self.environment_mapping.get(name, ""), "").encode()
            if not value:
                raise
            return value
