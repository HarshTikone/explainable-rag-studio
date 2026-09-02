"""Normalized records shared by parsing, chunking, and lifecycle storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Document:
    document_id: str
    logical_source: str
    title: str
    mime_type: str
    organization_id: str = "org_public"
    created_by: str = ""
    uploaded_by: str = ""
    trust_state: str = "public"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentVersion:
    document_version_id: str
    document_id: str
    source_sha256: str
    source_version: str
    size_bytes: int
    organization_id: str = "org_public"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedBlock:
    block_id: str
    block_type: str
    text: str
    page: int | None
    heading_path: Tuple[str, ...] = ()
    ordinal: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return data


@dataclass(frozen=True)
class ParentSection:
    parent_id: str
    document_id: str
    document_version_id: str
    heading_path: Tuple[str, ...]
    text: str
    page_start: int | None
    page_end: int | None
    block_types: Tuple[str, ...]
    ordinal: int


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    document_id: str
    document_version_id: str
    parent_id: str
    source: str
    source_version: str
    title: str
    heading_path: Tuple[str, ...]
    text: str
    retrieval_text: str
    generation_text: str
    contextual_prefix: str
    context_provenance: str
    page: int | None
    page_start: int | None
    page_end: int | None
    block_types: Tuple[str, ...]
    token_count: int
    content_fingerprint: str
    ordinal: int
    legacy_chunk_id: str | None = None
    organization_id: str = "org_public"
    trust_state: str = "public"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        data["block_types"] = list(self.block_types)
        return data


@dataclass
class ParseResult:
    document: Document
    version: DocumentVersion
    blocks: List[ParsedBlock]
    warnings: List[Dict[str, str]] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)
