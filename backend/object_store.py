"""Private S3-compatible storage with per-object envelope encryption."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import secrets
import uuid
from typing import BinaryIO, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .platform_contracts import StoredObject
from .platform_contracts import SecretProvider

MAGIC = b"RAGOBJ1\x00"
ORG_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


class ObjectIntegrityError(RuntimeError):
    pass


def _master_key(value: bytes) -> bytes:
    candidate = value.strip()
    try:
        decoded = base64.urlsafe_b64decode(candidate + b"=" * (-len(candidate) % 4))
    except Exception:
        decoded = b""
    if len(decoded) == 32:
        return decoded
    if len(candidate) == 32:
        return candidate
    raise ValueError("The object master key must contain exactly 32 bytes or URL-safe base64 for 32 bytes.")


class EnvelopeCipher:
    """AES-256-GCM data encryption with a separately wrapped random data key."""

    def __init__(self, master_key: bytes):
        self._master = AESGCM(_master_key(master_key))

    def encrypt(self, organization_id: str, plaintext: bytes) -> bytes:
        data_key = AESGCM.generate_key(bit_length=256)
        wrap_nonce, data_nonce = os.urandom(12), os.urandom(12)
        aad = organization_id.encode("utf-8")
        wrapped_key = self._master.encrypt(wrap_nonce, data_key, aad)
        ciphertext = AESGCM(data_key).encrypt(data_nonce, plaintext, aad)
        return MAGIC + wrap_nonce + len(wrapped_key).to_bytes(2, "big") + wrapped_key + data_nonce + ciphertext

    def decrypt(self, organization_id: str, envelope: bytes) -> bytes:
        if not envelope.startswith(MAGIC) or len(envelope) < len(MAGIC) + 26:
            raise ObjectIntegrityError("Encrypted object envelope is malformed.")
        offset = len(MAGIC)
        wrap_nonce = envelope[offset:offset + 12]
        offset += 12
        wrapped_length = int.from_bytes(envelope[offset:offset + 2], "big")
        offset += 2
        wrapped_key = envelope[offset:offset + wrapped_length]
        offset += wrapped_length
        data_nonce = envelope[offset:offset + 12]
        ciphertext = envelope[offset + 12:]
        try:
            aad = organization_id.encode("utf-8")
            data_key = self._master.decrypt(wrap_nonce, wrapped_key, aad)
            return AESGCM(data_key).decrypt(data_nonce, ciphertext, aad)
        except Exception as exc:
            raise ObjectIntegrityError("Encrypted object authentication failed.") from exc


class S3EnvelopeObjectStore:
    def __init__(
        self,
        endpoint_url: str,
        region: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        secret_provider: SecretProvider,
        master_key_name: str = "object_master_key",
        secure: bool = True,
        client=None,
    ):
        if not bucket:
            raise ValueError("An object-store bucket is required.")
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                use_ssl=secure,
            )
        self.client = client
        self.bucket = bucket
        self.cipher = EnvelopeCipher(secret_provider.read(master_key_name))

    @staticmethod
    def _validate_org(organization_id: str) -> None:
        if not ORG_PATTERN.fullmatch(organization_id):
            raise ValueError("Invalid organization identifier.")

    def put(
        self,
        organization_id: str,
        source_name: str,
        content: bytes | BinaryIO,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject:
        self._validate_org(organization_id)
        plaintext = content if isinstance(content, bytes) else content.read()
        checksum = hashlib.sha256(plaintext).hexdigest()
        suffix = os.path.splitext(source_name)[1].casefold()[:16]
        object_key = f"organizations/{organization_id}/uploads/{uuid.uuid4().hex}{suffix}"
        encrypted = self.cipher.encrypt(organization_id, plaintext)
        safe_metadata = {
            "organization": organization_id,
            "sha256": checksum,
            "encryption": "AES-256-GCM-envelope-v1",
            **{str(key)[:40]: str(value)[:200] for key, value in (metadata or {}).items()},
        }
        response = self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=encrypted,
            ContentType="application/octet-stream",
            Metadata=safe_metadata,
            ServerSideEncryption="AES256",
        )
        return StoredObject(
            organization_id=organization_id,
            object_key=object_key,
            version_id=response.get("VersionId"),
            checksum_sha256=checksum,
            size_bytes=len(plaintext),
            encryption_algorithm="AES-256-GCM-envelope-v1",
        )

    def get(self, organization_id: str, object_key: str) -> bytes:
        self._validate_org(organization_id)
        expected_prefix = f"organizations/{organization_id}/"
        if not object_key.startswith(expected_prefix):
            raise PermissionError("Object does not belong to the authenticated organization.")
        response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        envelope = response["Body"].read()
        plaintext = self.cipher.decrypt(organization_id, envelope)
        expected = response.get("Metadata", {}).get("sha256")
        if expected and not secrets.compare_digest(hashlib.sha256(plaintext).hexdigest(), expected):
            raise ObjectIntegrityError("Object checksum verification failed.")
        return plaintext

    def soft_delete(self, organization_id: str, object_key: str) -> None:
        self._validate_org(organization_id)
        if not object_key.startswith(f"organizations/{organization_id}/"):
            raise PermissionError("Object does not belong to the authenticated organization.")
        # Versioning keeps earlier encrypted versions recoverable for the retention window.
        self.client.delete_object(Bucket=self.bucket, Key=object_key)


class MemoryEnvelopeObjectStore:
    """Deterministic contract test adapter; never selected in protected production."""

    def __init__(self, master_key: bytes = b"0" * 32):
        self.cipher = EnvelopeCipher(master_key)
        self.objects: dict[str, bytes] = {}

    def put(self, organization_id, source_name, content, content_type, metadata=None):
        plaintext = content if isinstance(content, bytes) else content.read()
        key = f"organizations/{organization_id}/uploads/{uuid.uuid4().hex}{os.path.splitext(source_name)[1]}"
        self.objects[key] = self.cipher.encrypt(organization_id, plaintext)
        return StoredObject(organization_id, key, "memory-v1", hashlib.sha256(plaintext).hexdigest(), len(plaintext), "AES-256-GCM-envelope-v1")

    def get(self, organization_id, object_key):
        if not object_key.startswith(f"organizations/{organization_id}/"):
            raise PermissionError("Object does not belong to the authenticated organization.")
        return self.cipher.decrypt(organization_id, self.objects[object_key])

    def soft_delete(self, organization_id, object_key):
        if not object_key.startswith(f"organizations/{organization_id}/"):
            raise PermissionError("Object does not belong to the authenticated organization.")
        self.objects.pop(object_key, None)
