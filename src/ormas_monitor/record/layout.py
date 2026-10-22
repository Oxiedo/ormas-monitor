"""Where the parts of a record live inside a run directory, and how a path given by a user maps
onto them."""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

FORMAT = "ormas-monitor-record"


@dataclass(frozen=True)
class RecordLayout:
    """File names of a run directory. `diary` and `metrics` are the evidence the seal covers;
    reports are derived from them and can be regenerated, so they are not sealed."""

    diary: str = "diary.jsonl"
    metrics: str = "metrics.jsonl"
    seal: str = "record.json"
    signature: str = "record.sig"

    @property
    def sealed_files(self) -> tuple[str, ...]:
        return (self.diary, self.metrics)


LAYOUT = RecordLayout()


@dataclass(frozen=True)
class Location:
    """A resolved record. `run_dir` is None when the user gave a lone diary file (for example
    `tests/fixtures/v0_1_cifar10_with_repair.diary.jsonl`): then there are no metrics, seal or signature
    to look for, and verification says so instead of failing."""

    diary: Path
    run_dir: Path | None

    def file(self, name: str) -> Path | None:
        return None if self.run_dir is None else self.run_dir / name


def locate(path: str | Path, layout: RecordLayout = LAYOUT) -> Location:
    """Resolve a run directory or a diary file. A file named like the layout's diary counts as
    part of its run directory; any other file is treated as a stand-alone diary."""
    p = Path(path)
    if p.is_dir():
        return Location(diary=p / layout.diary, run_dir=p)
    if p.name == layout.diary:
        return Location(diary=p, run_dir=p.parent)
    return Location(diary=p, run_dir=None)


def file_problem(path: Path) -> str | None:
    """Why `path` is not a plain file of the run directory: None for a regular file, "missing",
    or a short reason ("a symbolic link ...", "not a regular file", ...). Never follows a link.

    Record files must be the directory's own regular files. A link would let a run directory
    handed over by someone else make `sign` write, or `verify` read, a file elsewhere on the
    reviewer's machine; a FIFO or device (say /dev/zero) would make hashing it hang.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    except ValueError:              # an embedded NUL byte
        return "not a valid file name"
    except OSError as e:
        return f"unreadable ({e.strerror or e})"
    if stat.S_ISLNK(st.st_mode):
        return "a symbolic link (not followed: a record must hold its own files)"
    if not stat.S_ISREG(st.st_mode):
        return "not a regular file"
    return None
