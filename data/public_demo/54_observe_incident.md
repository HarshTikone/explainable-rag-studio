# Signal Incident OB-6602

Incident ID OP-2026-178. A client update omitted reranking spans and triggered OB-6602. Requests remained successful, but cost attribution was incomplete for 46 minutes. The SDK now validates required span names at startup.
