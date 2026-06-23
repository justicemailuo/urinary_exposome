from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEMO_API_FREE_CALLS, DEMO_API_QUOTA_FILE


_LOCK = threading.Lock()


def make_client_key(client_id: str | None, fallback: str | None = None) -> str:
    raw = (client_id or fallback or "anonymous").strip() or "anonymous"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load(path: Path = DEMO_API_QUOTA_FILE) -> dict[str, Any]:
    if not path.exists():
        return {"clients": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"clients": {}}


def _save(data: dict[str, Any], path: Path = DEMO_API_QUOTA_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_usage(client_key: str) -> dict[str, int]:
    with _LOCK:
        data = _load()
        used = int(data.get("clients", {}).get(client_key, {}).get("used", 0))
    return {
        "used": used,
        "remaining": max(DEMO_API_FREE_CALLS - used, 0),
        "limit": DEMO_API_FREE_CALLS,
    }


def has_quota(client_key: str) -> bool:
    return get_usage(client_key)["remaining"] > 0


def increment_usage(client_key: str) -> dict[str, int]:
    with _LOCK:
        data = _load()
        clients = data.setdefault("clients", {})
        row = clients.setdefault(client_key, {})
        row["used"] = int(row.get("used", 0)) + 1
        row["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _save(data)
        used = int(row["used"])
    return {
        "used": used,
        "remaining": max(DEMO_API_FREE_CALLS - used, 0),
        "limit": DEMO_API_FREE_CALLS,
    }
