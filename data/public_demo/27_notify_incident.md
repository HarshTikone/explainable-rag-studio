# Pulse Incident NT-5502

Incident ID CM-2026-082. A rotated webhook secret was not propagated to one worker pool, producing NT-5502. Four retries failed before the secret was synchronized. Delivery recovered within eleven minutes and secret-version telemetry was added.
