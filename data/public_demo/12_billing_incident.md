# Ledger Incident BI-2207

Incident ID FI-2026-031. A duplicated metering batch produced incorrect BI-2207 adjustments for 84 test accounts. Finance Operations reversed the batch before invoices were released. No customer was charged, and remediation added an idempotency key to metering imports.
