"""The Diary: an append-only, hash-chained record of what happened during training.

Each line is one JSON event. It carries the SHA-256 of the previous line, so editing,
deleting or reordering any line breaks every hash after it, and `verify` finds where.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    payload = prev + json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class Diary:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._prev = GENESIS
        self._seq = 0

    def write(self, event: str, **fields) -> dict:
        body = {"seq": self._seq, "time": round(time.time(), 3), "event": event, **fields}
        line = {**body, "prev": self._prev, "hash": _digest(self._prev, body)}
        self._fh.write(json.dumps(line, default=str) + "\n")
        self._fh.flush()
        self._prev, self._seq = line["hash"], self._seq + 1
        return line

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read(path: str | Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def verify(path: str | Path) -> tuple[bool, int, str]:
    """Return (intact, number of events, message)."""
    prev = GENESIS
    events = read(path)
    for i, line in enumerate(events):
        body = {k: v for k, v in line.items() if k not in ("prev", "hash")}
        if line.get("prev") != prev or line.get("seq") != i or _digest(prev, body) != line.get("hash"):
            return False, len(events), f"record broken at event {i} (seq {line.get('seq')})"
        prev = line["hash"]
    return True, len(events), f"all {len(events)} events intact"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
