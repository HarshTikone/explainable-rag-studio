# Harbor Incident BK-1500

Incident ID OP-2026-052. A restore drill exposed an expired encryption grant on the us-east-2 backup copy. The primary backup remained valid. Renewing the grant restored access in 43 minutes, and weekly verification now validates both integrity and authorization.
