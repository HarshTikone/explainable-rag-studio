"""Atomic Redis rate limits with a deliberately narrow demo fallback."""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict

from .platform_contracts import RateLimitDecision

TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local values = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(values[1]) or capacity
local updated = tonumber(values[2]) or now_ms
tokens = math.min(capacity, tokens + math.max(0, now_ms - updated) * refill_per_ms)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HMSET', key, 'tokens', tokens, 'updated', now_ms)
redis.call('PEXPIRE', key, math.ceil((capacity / refill_per_ms) * 2))
return {allowed, math.floor(tokens), math.ceil((capacity - tokens) / refill_per_ms)}
"""


class RateLimitUnavailable(RuntimeError):
    pass


class RedisRateLimiter:
    def __init__(self, redis_client, namespace: str = "rag:rate"):
        self.redis = redis_client
        self.namespace = namespace

    def check(self, key: str, limit: int, window_seconds: int, cost: int = 1) -> RateLimitDecision:
        if limit <= 0 or window_seconds <= 0 or cost <= 0:
            raise ValueError("Rate-limit values must be positive.")
        now_ms = int(time.time() * 1000)
        try:
            allowed, remaining, reset_ms = self.redis.eval(
                TOKEN_BUCKET_LUA,
                1,
                f"{self.namespace}:{key}",
                limit,
                limit / (window_seconds * 1000),
                now_ms,
                cost,
            )
        except Exception as exc:
            raise RateLimitUnavailable("Rate-limit enforcement is unavailable.") from exc
        return RateLimitDecision(bool(allowed), limit, max(0, int(remaining)), max(1, math.ceil(int(reset_ms) / 1000)), "" if allowed else "RATE_LIMITED")

    def ping(self) -> bool:
        try:
            return bool(self.redis.ping())
        except Exception:
            return False


class MemoryDemoRateLimiter:
    """Single-process fixed window used only for anonymous public demo reads."""

    def __init__(self):
        self._windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int, cost: int = 1) -> RateLimitDecision:
        current = int(time.time())
        window = current - current % window_seconds
        with self._lock:
            stored_window, used = self._windows[key]
            if stored_window != window:
                used = 0
            allowed = used + cost <= limit
            if allowed:
                used += cost
            self._windows[key] = (window, used)
        return RateLimitDecision(allowed, limit, max(0, limit - used), max(1, window + window_seconds - current), "" if allowed else "RATE_LIMITED")
