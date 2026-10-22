"""Self-contained reports, written on this machine with no network access.

    write_all(run_dir, gpu_hour_cost=None, gpus=None)  -> {"report", "summary", "summary_md"}
    write_run_report(run_dir)                          -> report.html   (engineers)
    write_summary(run_dir, ...)                        -> summary.html + summary.md (decision-makers)
    write_check_report(check_result, path)             -> checkpoint health report
    write_diff_report(diff_result, path)               -> comparison of two runs

Every page is a single HTML file with inline CSS (light, dark and print) and inline SVG. No
fonts, scripts or images are fetched, so a report opens offline years later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import check as _check
from . import diff as _diff
from . import run as _run
from . import summary as _summary
from .facts import CostInputs, ReportConfig, RunFacts, analyse

__all__ = ["CostInputs", "ReportConfig", "RunFacts", "analyse", "write_all", "write_check_report",
           "write_diff_report", "write_run_report", "write_summary"]


def write_all(run_dir: str | Path, gpu_hour_cost: Optional[float] = None, gpus: Optional[int] = None,
              config: Optional[ReportConfig] = None, public_key: str | Path | None = None) -> dict[str, Path]:
    """Write every report for a monitored run; `Monitor.close()` and `ormas-monitor report` call this.

    `gpu_hour_cost` (price per GPU-hour) and `gpus` are optional. Only when a price is given does the
    executive summary print a dollar figure, labelled ESTIMATE with its formula. `public_key` (PEM)
    lets the record check verify a signature instead of reporting it as present but unchecked.
    Returns {"report": engineer HTML, "summary": executive HTML, "summary_md": executive Markdown}.
    """
    facts = analyse(run_dir, config, CostInputs(gpu_hour_cost=gpu_hour_cost, gpus=gpus), public_key)
    report = _run.write(facts)
    summary, summary_md = _summary.write(facts)
    return {"report": report, "summary": summary, "summary_md": summary_md}


def write_run_report(run_dir: str | Path, config: Optional[ReportConfig] = None) -> Path:
    """Write the engineer report `report.html` into the run directory and return its path."""
    return _run.write(analyse(run_dir, config))


def write_summary(run_dir: str | Path, gpu_hour_cost: Optional[float] = None,
                  gpus: Optional[int] = None) -> tuple[Path, Path]:
    """Write the executive summary (`summary.html`, `summary.md`) into the run directory."""
    return _summary.write(analyse(run_dir, None, CostInputs(gpu_hour_cost=gpu_hour_cost, gpus=gpus)))


def write_check_report(check_result: dict, path: str | Path) -> Path:
    """Render a checkpoint check result (`ormas_monitor.checkpoint.run`) as HTML at `path`."""
    return _check.write(check_result, path)


def write_diff_report(diff_result: dict, path: str | Path) -> Path:
    """Render a run diff (`ormas_monitor.record.diff.diff_runs`) as HTML at `path`."""
    return _diff.write(diff_result, path)
