from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Basic structured logger
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hackrx")


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class TTLCache:
    """Simple TTL cache for small objects. Not thread-safe but sufficient for this app.
    Used for embedding/method-level memoization to reduce cost.
    """

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 5000):
        self.ttl = ttl_seconds
        self.max = max_items
        self.store: Dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        rec = self.store.get(key)
        if not rec:
            return None
        ts, val = rec
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        if len(self.store) >= self.max:
            # drop arbitrary item (oldest not tracked to keep it simple)
            self.store.pop(next(iter(self.store)))
        self.store[key] = (time.time(), val)


@dataclass
class ChunkMeta:
    doc_url: str
    page: int
    chunk_id: str
    start: int
    end: int


def sanitize_url(url: str) -> str:
    # Basic sanitization: allow http(s) and file URLs, disallow query with credentials
    if not re.match(r"^(https?://|file://).+", url):
        raise ValueError("Unsupported URL scheme. Only http(s):// and file:// are allowed")
    return url


def read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    return v.strip() or default


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
