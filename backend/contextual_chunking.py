"""Heading-aware parent/child chunk construction and contextual prefixes."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from .chunking import chunk_text_token_based, count_tokens
from .document_parsers import normalize_text
from .ingestion_models import ChildChunk, Document, DocumentVersion, ParentSection, ParsedBlock

CHUNKER_VERSION = "2.0"
CONTEXT_PROMPT_VERSION = "1.0"


def _digest(*values: object, length: int = 20) -> str:
    payload = "\x1f".join(normalize_text(str(value)) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def deterministic_context(document: Document, version: DocumentVersion, heading_path: Sequence[str], page_start: int | None, page_end: int | None, block_types: Sequence[str]) -> str:
    section = " > ".join(heading_path) if heading_path else "Document body"
    pages = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
    return f"Document: {document.title}. Version: {version.source_version}. Section: {section}. Pages: {pages or 'n/a'}. Content: {', '.join(block_types)}."


def build_parent_sections(document: Document, version: DocumentVersion, blocks: Sequence[ParsedBlock], parent_tokens: int = 1200) -> List[ParentSection]:
    groups: Dict[Tuple[str, ...], List[ParsedBlock]] = defaultdict(list)
    order: List[Tuple[str, ...]] = []
    for block in blocks:
        key = block.heading_path
        if key not in groups:
            order.append(key)
        groups[key].append(block)
    parents: List[ParentSection] = []
    for heading_path in order:
        current: List[ParsedBlock] = []
        current_tokens = 0
        section_ordinal = 0
        for block in groups[heading_path]:
            block_tokens = count_tokens(block.text)
            if current and current_tokens + block_tokens > parent_tokens:
                section_ordinal += 1
                parents.append(_make_parent(document, version, heading_path, current, section_ordinal))
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            section_ordinal += 1
            parents.append(_make_parent(document, version, heading_path, current, section_ordinal))
    return parents


def _make_parent(document: Document, version: DocumentVersion, heading_path: Tuple[str, ...], blocks: Sequence[ParsedBlock], ordinal: int) -> ParentSection:
    text = "\n\n".join(block.text for block in blocks)
    pages = [block.page for block in blocks if block.page is not None]
    block_types = tuple(dict.fromkeys(block.block_type for block in blocks))
    parent_id = "par_" + _digest(document.document_id, heading_path, ordinal, text)
    return ParentSection(parent_id, document.document_id, version.document_version_id, heading_path, text, min(pages) if pages else None, max(pages) if pages else None, block_types, ordinal)


def build_contextual_chunks(
    document: Document,
    version: DocumentVersion,
    blocks: Sequence[ParsedBlock],
    *,
    child_tokens: int = 420,
    overlap_tokens: int = 80,
    parent_tokens: int = 1200,
    context_enhancer: Callable[[str, str], str | None] | None = None,
) -> List[ChildChunk]:
    chunks: List[ChildChunk] = []
    for parent in build_parent_sections(document, version, blocks, parent_tokens):
        prefix = deterministic_context(document, version, parent.heading_path, parent.page_start, parent.page_end, parent.block_types)
        for child_ordinal, text in enumerate(chunk_text_token_based(parent.text, child_tokens, overlap_tokens), 1):
            provenance = "deterministic"
            if context_enhancer:
                enhanced = normalize_text(context_enhancer(prefix, text) or "")
                if enhanced:
                    prefix, provenance = enhanced, "provider_cached"
            chunk_id = "chk_" + _digest(document.document_id, parent.heading_path, parent.ordinal, child_ordinal, text)
            retrieval_text = f"{prefix}\n\n{text}"
            parent_excerpt = parent.text[:2400]
            generation_text = text if parent_excerpt == text else f"{text}\n\nParent section context:\n{parent_excerpt}"
            cache_namespace = "" if document.organization_id == "org_public" else document.organization_id + "\x1f"
            fingerprint = hashlib.sha256(f"{cache_namespace}{normalize_text(text)}\x1f{normalize_text(prefix)}".encode("utf-8")).hexdigest()
            chunks.append(ChildChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                document_version_id=version.document_version_id,
                parent_id=parent.parent_id,
                source=document.logical_source,
                source_version=version.source_version,
                title=document.title,
                heading_path=parent.heading_path,
                text=text,
                retrieval_text=retrieval_text,
                generation_text=generation_text,
                contextual_prefix=prefix,
                context_provenance=provenance,
                page=parent.page_start,
                page_start=parent.page_start,
                page_end=parent.page_end,
                block_types=parent.block_types,
                token_count=count_tokens(text),
                content_fingerprint=fingerprint,
                ordinal=child_ordinal,
                organization_id=document.organization_id,
                trust_state=document.trust_state,
            ))
    return chunks
