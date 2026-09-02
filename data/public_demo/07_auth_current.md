# Meridian Authentication — Current Runbook

Status: current. Meridian issues access tokens with a 45-minute lifetime and rotates signing keys every 14 days. Authentication failures with code AU-4012 require checking the tenant clock skew before rotating credentials. The owning team is Identity Reliability. This runbook supersedes the legacy eight-hour token procedure.
