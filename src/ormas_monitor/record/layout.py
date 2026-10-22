"""Where the parts of a record live inside a run directory, and how a path given by a user maps
onto them."""
from __future__ import annotations

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
    `sample_reports/cifar10_with_repair.diary.jsonl`): then there are no metrics, seal or signature
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
