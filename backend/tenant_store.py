"""Bounded, lazy cache of physically isolated organization vector stores."""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from .vectorstore import FaissStore

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


class TenantStoreManager:
    def __init__(self, root: str, max_cached: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_cached = max(1, max_cached)
        self._stores: OrderedDict[str, FaissStore] = OrderedDict()
        self._lock = RLock()

    def organization_dir(self, organization_id: str) -> Path:
        if not SAFE_ID.fullmatch(organization_id or ""):
            raise ValueError("Invalid organization identifier.")
        candidate = (self.root / organization_id).resolve()
        if self.root.resolve() not in candidate.parents:
            raise ValueError("Organization path escaped the tenant root.")
        return candidate

    def get(self, organization_id: str, reload: bool = False) -> FaissStore:
        with self._lock:
            if reload or organization_id not in self._stores:
                store = FaissStore(str(self.organization_dir(organization_id)))
                store.load()
                self._stores[organization_id] = store
            self._stores.move_to_end(organization_id)
            while len(self._stores) > self.max_cached:
                self._stores.popitem(last=False)
            return self._stores[organization_id]

    def invalidate(self, organization_id: str) -> None:
        with self._lock:
            self._stores.pop(organization_id, None)
