# Orbit Document Lifecycle — Current Policy

Status: current. Orbit identifies sources by SHA-256 checksum and version ID. Updating a document creates a new generation, while deletion removes all associated chunks before activation. Lifecycle code DL-2406 means stale chunks remain after an update.
