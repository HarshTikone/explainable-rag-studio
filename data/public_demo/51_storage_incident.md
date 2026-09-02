# Vault Incident ST-7305

Incident ID KG-2026-169. A terminated rebuild wrote vectors but not final metadata, causing ST-7305. Activation checks rejected the generation and kept the previous index online. The failed generation was safely removed.
