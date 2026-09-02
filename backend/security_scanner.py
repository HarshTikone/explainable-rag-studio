"""Layered upload and indirect-prompt-injection screening.

This scanner is deliberately conservative and dependency-light. It is a gate,
not a malware sandbox; findings are retained for human quarantine review.
"""
from __future__ import annotations

import hashlib
import io
import mimetypes
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import SETTINGS
from .security_models import SecurityFinding

ALLOWED_MIME = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".html": {"text/html", "application/xhtml+xml", "text/plain"},
    ".htm": {"text/html", "application/xhtml+xml", "text/plain"},
}
INJECTION_PATTERNS = (
    ("PROMPT_OVERRIDE", re.compile(r"\b(ignore|disregard|override)\b.{0,50}\b(previous|system|developer|instructions?)\b", re.I | re.S)),
    ("SYSTEM_PROMPT_EXTRACTION", re.compile(r"\b(reveal|print|show|extract)\b.{0,50}\b(system|developer) prompt\b", re.I | re.S)),
    ("CREDENTIAL_EXFILTRATION", re.compile(r"\b(send|upload|exfiltrate|reveal)\b.{0,60}\b(api[ _-]?key|credential|password|secret|token)\b", re.I | re.S)),
    ("TOOL_INSTRUCTION", re.compile(r"\b(call|invoke|execute|run)\b.{0,40}\b(tool|shell|command|function)\b", re.I | re.S)),
)
BASE64_PAYLOAD = re.compile(r"(?:[A-Za-z0-9+/]{4}){40,}(?:==|=)?")
HIDDEN_HTML = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0)", re.I)


@dataclass(frozen=True)
class ScanReport:
    findings: tuple[SecurityFinding, ...]
    detected_mime: str

    @property
    def quarantined(self) -> bool:
        return any(finding.severity in {"high", "critical"} for finding in self.findings)

    def to_dict(self) -> dict:
        return {"detected_mime": self.detected_mime, "quarantined": self.quarantined,
                "findings": [finding.to_dict() for finding in self.findings]}


def _finding(code: str, severity: str, excerpt: bytes | str = b"", location: str = "file", confidence: float = 1.0, **details) -> SecurityFinding:
    raw = excerpt.encode("utf-8", "ignore") if isinstance(excerpt, str) else excerpt
    return SecurityFinding(code, severity, location, "1.0", confidence, hashlib.sha256(raw[:4096]).hexdigest(), details)


def _detect_mime(content: bytes, suffix: str) -> str:
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"PK\x03\x04"):
        return "application/zip"
    sample = content[:4096].lstrip().lower()
    if sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return "text/html"
    if b"\x00" not in sample:
        try:
            sample.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            pass
    return "application/octet-stream"


def _inspect_zip(content: bytes, findings: list[SecurityFinding], suffix: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > SETTINGS.max_archive_entries:
                findings.append(_finding("ARCHIVE_ENTRY_LIMIT", "high", details_count=len(entries)))
            total = sum(entry.file_size for entry in entries)
            if total > SETTINGS.max_decompressed_mb * 1024 * 1024:
                findings.append(_finding("ARCHIVE_SIZE_LIMIT", "high", decompressed_bytes=total))
            compressed = max(1, sum(entry.compress_size for entry in entries))
            if total / compressed > SETTINGS.max_compression_ratio:
                findings.append(_finding("ARCHIVE_COMPRESSION_BOMB", "critical", ratio=round(total / compressed, 2)))
            names = {entry.filename for entry in entries}
            for entry in entries:
                normalized = PurePosixPath(entry.filename.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    findings.append(_finding("ARCHIVE_PATH_TRAVERSAL", "critical", entry.filename, location=entry.filename))
            if suffix == ".docx" and not {"[Content_Types].xml", "word/document.xml"} <= names:
                findings.append(_finding("MALFORMED_DOCX_CONTAINER", "high"))
            xml = b""
            for name in ("word/document.xml", "word/styles.xml", "word/_rels/document.xml.rels"):
                if name in names and archive.getinfo(name).file_size <= 5 * 1024 * 1024:
                    xml += archive.read(name)
            if b"w:vanish" in xml or b"w:color w:val=\"FFFFFF\"" in xml:
                findings.append(_finding("HIDDEN_DOCX_TEXT", "high", xml, location="document.xml", confidence=.9))
            if b"vbaProject" in content or any(name.casefold().endswith((".bin", ".exe", ".js")) for name in names):
                findings.append(_finding("ACTIVE_OFFICE_CONTENT", "critical"))
            return xml
    except (zipfile.BadZipFile, RuntimeError, KeyError):
        findings.append(_finding("MALFORMED_CONTAINER", "high"))
        return b""


def scan_upload(filename: str, content: bytes, declared_mime: str | None = None) -> ScanReport:
    suffix = Path(filename).suffix.casefold()
    findings: list[SecurityFinding] = []
    detected = _detect_mime(content, suffix)
    if suffix not in ALLOWED_MIME:
        findings.append(_finding("EXTENSION_NOT_ALLOWED", "critical", suffix))
    allowed = ALLOWED_MIME.get(suffix, set())
    if declared_mime and declared_mime.casefold().split(";")[0].strip() not in allowed:
        findings.append(_finding("DECLARED_MIME_MISMATCH", "high", declared_mime))
    if suffix == ".pdf" and detected != "application/pdf":
        findings.append(_finding("MAGIC_SIGNATURE_MISMATCH", "critical", content[:16]))
    if suffix == ".docx" and detected != "application/zip":
        findings.append(_finding("MAGIC_SIGNATURE_MISMATCH", "critical", content[:16]))

    inspect_bytes = content[:2_000_000]
    if suffix == ".docx" and detected == "application/zip":
        inspect_bytes += _inspect_zip(content, findings, suffix)
    if suffix == ".pdf" and detected == "application/pdf":
        lowered = inspect_bytes.lower()
        if b"/encrypt" in lowered:
            findings.append(_finding("ENCRYPTED_PDF", "high"))
        if any(token in lowered for token in (b"/javascript", b"/js ", b"/launch", b"/embeddedfile")):
            findings.append(_finding("ACTIVE_PDF_CONTENT", "critical"))
        if b"3 tr" in lowered and b"tr 0.001" in lowered:
            findings.append(_finding("HIDDEN_PDF_TEXT", "high", confidence=.7))
        try:
            import fitz
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if pdf.page_count > SETTINGS.max_document_pages:
                    findings.append(_finding("PAGE_LIMIT", "high", page_count=pdf.page_count))
        except Exception:
            findings.append(_finding("MALFORMED_PDF", "high", confidence=.8))

    text = inspect_bytes.decode("utf-8", "ignore")
    if suffix in {".html", ".htm"} and (HIDDEN_HTML.search(text) or "hidden=" in text.casefold()):
        findings.append(_finding("HIDDEN_HTML_CONTENT", "high", text, confidence=.9))
    for code, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(_finding(code, "high", match.group(0), confidence=.9))
    if BASE64_PAYLOAD.search(text):
        findings.append(_finding("ENCODED_PAYLOAD", "medium", confidence=.7))
    return ScanReport(tuple(findings), detected)


def scanner_capabilities() -> dict[str, object]:
    return {"version": "1.0", "container_validation": True, "prompt_injection": True,
            "malware_sandbox": False, "archive_limits": True}
