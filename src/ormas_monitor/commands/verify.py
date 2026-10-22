"""`ormas-monitor verify`: check that a run's record has not been edited."""
from __future__ import annotations

from pathlib import Path


def register(sub) -> None:
    v = sub.add_parser("verify", help="check that a run's record has not been edited")
    v.add_argument("run", help="run directory or diary.jsonl")
    v.set_defaults(func=run)


def run(a) -> int:
    from ..diary import verify
    path = Path(a.run)
    diary = path / "diary.jsonl" if path.is_dir() else path
    ok, n, msg = verify(diary)
    print(("intact: " if ok else "BROKEN: ") + msg)
    return 0 if ok else 1
