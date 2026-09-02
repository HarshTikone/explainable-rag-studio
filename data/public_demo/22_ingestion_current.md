# Atlas Ingestion — Current Parsing Policy

Status: current. Atlas accepts PDF, Markdown, and plain text. Files are checksummed with SHA-256 before parsing, and duplicate content is skipped. Ingestion code IN-3306 indicates unsupported or inconsistent MIME type. Parser failures enter a retry queue with three attempts.
