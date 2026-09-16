# Threat model

Scope: the multi-tenant RAG platform as deployed by `docker-compose.yml` (`PLATFORM_MODE=postgres`).
Assets in scope: tenant document content, tenant query/answer history, API keys and OIDC tokens,
the audit log, and the LLM's context window (since ingested content becomes part of every future
prompt for that tenant). Out of scope: the underlying cloud/host infrastructure, and Gemini's own
model-side safety behavior — this document covers what this codebase controls.

Each threat below cites the control that mitigates it and the test that verifies it, so this stays
a checkable claim rather than a description of intent. Where there's a real, unclosed gap, it's
listed as one — see [Known gaps](#known-gaps).

## Trust boundaries

```
Untrusted                    │  Trusted (this platform)               │  Trusted (external)
──────────────────────────── │ ─────────────────────────────────────  │ ────────────────────
Uploaded documents           │  api / worker / streamlit containers   │  Gemini API
Query text (any tenant)      │  Postgres (RLS-enforced per-tenant)    │  Keycloak (OIDC)
API requests (any caller)    │  MinIO (envelope-encrypted per-object) │
```

The critical assumption the whole design leans on: **content that crosses from "uploaded
document" into "retrieved context" is still untrusted** — it was written by whoever uploaded it,
and it ends up concatenated into the LLM prompt. The platform treats it accordingly rather than
trusting it once it's indexed.

## Threats and mitigations

### 1. Cross-tenant data leakage (retrieval-time)

**Threat**: a query from organization A retrieves or cites a chunk that belongs to organization B.

**Mitigation**: tenant scoping is enforced at two independent layers, not one — application-level
filtering (`security_models.RetrievalScope`, checked inside `backend/retriever.py` before results
are returned, across all four retrieval strategies) *and* database-level `FORCE ROW LEVEL
SECURITY` on Postgres with a `NOBYPASSRLS` application role (`backend/database.py`), so a bug in
the application filter alone can't leak data — the database would still refuse the row. In legacy
(non-Postgres) mode, tenants get physically separate FAISS/BM25 index files
(`backend/tenant_store.py`), so there's no shared index to filter at all.

**Verified by**: `tests/test_tenant_isolation.py` (all 4 retrieval strategies).

### 2. Indirect prompt injection via uploaded documents

**Threat**: an uploaded document contains text instructing the LLM to ignore its system prompt,
reveal the system prompt, exfiltrate credentials, or invoke tools/commands — classic indirect
prompt injection, since the attacker controls content the model will read, not the chat turn.

**Mitigation**: `backend/security_scanner.py` pattern-matches ingested content against known
injection framings (`PROMPT_OVERRIDE`, `SYSTEM_PROMPT_EXTRACTION`, `CREDENTIAL_EXFILTRATION`,
`TOOL_INSTRUCTION`) plus hidden-content techniques (base64-blob payloads, CSS-hidden HTML via
`display:none`/`visibility:hidden`/zero-opacity/zero-font-size). A match at `high`/`critical`
severity quarantines the document (`ScanReport.quarantined`) before it ever reaches the index,
pending human review (`SecurityRegistry.resolve_quarantine`, requiring a reasoned owner/admin
decision, not just a click-through).

This is defense-in-depth, not a hard guarantee — pattern matching on natural language can't catch
every phrasing. The second, independent backstop is claim-level grounding itself
(`docs/CLAIM_LEVEL_GROUNDING.md`): even if injected text slips past the scanner, the answer
pipeline only displays claims that are cited *and* verified against retrieved evidence, so an
injected instruction has no path to becoming an unverified claim in a displayed answer — at worst
it becomes retrievable (but unsupported, and therefore filtered) text.

**Verified by**: `tests/test_security_scanner.py`, plus indirect-injection/poisoning/exfiltration
red-team cases in `tests/test_security.py` and `tests/test_api_security.py`.

### 3. Credential compromise / API key theft

**Threat**: an API key is leaked (logs, source control, a compromised client) and used by an
attacker to impersonate the legitimate caller.

**Mitigation**: keys are never stored in recoverable form — `SecurityRegistry._key_digest`
HMACs the key with a server-side pepper (`API_KEY_PEPPER`, itself a Docker secret, never in the
image or a committed file) and stores only the digest; `authenticate()` compares with
`hmac.compare_digest` (timing-safe) against a dummy digest even when the key ID doesn't exist, so
key-ID enumeration doesn't get a timing oracle. Keys expire (`expires_days`, default 90) and are
individually revocable. Every key's scopes are re-intersected against its current role at
*authentication* time, not just creation time (`authenticate()`: `scopes = ... & ROLE_SCOPES[role]`),
so a role downgrade after key issuance narrows the key immediately rather than waiting for
rotation.

**Residual risk**: this mitigates *use* of a stolen key within its scope and lifetime; it doesn't
prevent theft itself (that's a client-side/transport concern — HTTPS termination and secret
hygiene are the deployer's responsibility, same as any bearer-token API). It also doesn't throttle
*guessing* — see [Known gaps](#known-gaps).

### 4. Audit log tampering

**Threat**: an attacker with some level of database access rewrites or deletes audit history to
cover their tracks.

**Mitigation**: audit events are hash-chained — each event's hash covers the previous event's
hash (`backend/security.py::append_audit`/`verify_audit_chain`), so altering or removing an
entry breaks the chain from that point forward and is detected by `verify_audit_chain()`
(exercised live in CI via `/audit/verify` against the running stack, not just unit-tested).
Sensitive fields (API keys, secrets, tokens, prompts, raw answer text) are redacted before
storage (`_redact_details`), so the audit log itself isn't a secondary place secrets can leak
from.

**Residual risk**: this detects tampering after the fact; it doesn't prevent someone with direct,
unmediated database write access from tampering (no system-level audit log can, short of
write-once storage, which isn't in place here).

### 5. Object storage compromise

**Threat**: the object store (MinIO, or whatever S3-compatible backend replaces it in a real
deployment) is compromised or misconfigured to be publicly readable.

**Mitigation**: objects are client-side envelope-encrypted before they're written
(`backend/object_store.py`: AES-256-GCM, random per-object data key, wrapped by a master key,
organization ID bound in as AEAD associated data so a wrapped key from one tenant can't decrypt
another tenant's object even if somehow presented together) — the object store only ever holds
ciphertext. This is a deliberate choice over relying on the store's own server-side encryption
(MinIO's SSE requires a configured KMS this deployment doesn't run, and even where SSE is
available, it protects data at rest from the storage layer, not from a storage-layer
misconfiguration that makes objects readable).

### 6. Resource exhaustion by an authenticated caller

**Threat**: a valid API key or session is used to flood the API with requests — expensive query
traffic, bulk ingestion jobs, or high-frequency writes — degrading service for other tenants
sharing the same infrastructure.

**Mitigation**: every authenticated request is rate-limited per organization+identity+endpoint
class (`api.py::_rate_limit`, backed by `backend/rate_limit.py`'s `RedisRateLimiter`), with
separate budgets for queries (`RATE_LIMIT_QUERY_PER_MINUTE`, default 60/min), writes
(`RATE_LIMIT_WRITE_PER_MINUTE`, default 20/min), and ingestion jobs
(`RATE_LIMIT_INGEST_PER_HOUR`, default 10/hour) — ingestion gets the tightest budget since it's
the most resource-intensive path. Responses carry standard `X-RateLimit-*`/`Retry-After` headers,
and if Redis itself is unavailable the limiter fails closed (`RateLimitUnavailable` → HTTP 503),
not open.

### 7. Malicious or malformed file upload

**Threat**: a file with a spoofed extension, an oversized payload, or a malformed archive (zip
bomb via `.docx`) is uploaded to exhaust resources or execute unintended parsing behavior.

**Mitigation**: `scan_upload()` checks the *declared* extension against an explicit allow-list,
sniffs actual content type independently (`_detect_mime`) rather than trusting the declared MIME
type, and inspects `.docx` (a zip container) contents before full extraction (`_inspect_zip`).

## Known gaps

Being direct about what isn't closed, per this project's own rule against overclaiming results:

- **Grounding quality is not yet promoted.** The NLI verifier's held-out `macro_f1` and
  `contradiction_recall` are still below the promotion bar as of the latest CI run (see
  `docs/ADVANCED_BUILD_PLAN.md`, Milestone 6.5) — meaning claim verification, while
  architecturally sound, doesn't yet meet its own accuracy bar on hard contradiction cases. This
  is a quality gap, not a security bypass (unsupported claims are still filtered from display
  either way), but it's tracked here because grounding is this platform's actual security
  boundary against hallucinated/injected content reaching a user, not just a quality metric.
- **Authentication attempts themselves are not rate-limited.** `_rate_limit()` (§6) runs only
  *after* `authenticate()`/OIDC validation succeeds — a wrong API key or bearer token gets a 401
  with no per-caller throttling before that point, so brute-forcing the key-secret portion of an
  API key (or probing many key IDs) isn't slowed down at the application layer. Successful,
  authenticated traffic volume *is* bounded (§6); this gap is specifically about the pre-auth
  guessing surface.
- **Prompt-injection pattern matching (§2) is signature-based**, not a learned classifier — novel
  phrasings that don't match `INJECTION_PATTERNS` won't quarantine on ingestion, though the
  grounding backstop still applies at answer time.
