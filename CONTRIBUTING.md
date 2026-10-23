# Contributing to ormas-monitor

Thank you for helping. Bug reports, "a finding looks wrong" reports, docs fixes and code are all welcome.

## Development setup

```bash
git clone https://github.com/Oxiedo/ormas-monitor
cd ormas-monitor
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pip install -e ".[dev,all]"                            # optional: every integration, for the full test suite
```

On Linux without a GPU, install the CPU build of PyTorch first to save a large download:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Checks to run before a pull request

```bash
ruff check .                 # lint
pytest -q                    # tests: the whole suite should finish in about a minute on CPU
mkdocs build --strict        # docs, if you touched docs/ or mkdocs.yml
python -m build && twine check dist/*   # packaging, if you touched pyproject.toml
```

Tests for optional integrations use `pytest.importorskip`, so they are skipped when the extra is not installed.

## House style

- **Module shape.** A module docstring saying what the file is; `from __future__ import annotations`; imports grouped stdlib / third-party / first-party (relative `from .x import y`); configuration in a dataclass, never scattered constants.
- **Python 3.9+.** No `match`. No `X | Y` at runtime (in annotations it is fine thanks to the future import). No `list[int]` inside `isinstance`.
- **Types.** Fully type-annotate public functions. *Absent* (`None`) and *zero* are different states; never conflate them.
- **Errors.** Raise on programmer error; return a sentinel on expected absence. Never swallow an exception to keep training alive. The one sanctioned exception: a user sink that raises is disabled and recorded as a `sink_error` event.
- **Records that cross a boundary** are frozen dataclasses or documented dicts with a schema. Every diary event type is listed in `ormas_monitor/schema.py`.
- **Assertions beat comments.** Where a property must hold, assert it.
- **Performance.** Never read a scalar from a device tensor inside a per-layer loop; stack on the device and read once (`torch.stack(...).tolist()`). Hooks do nothing on unsampled steps.
- **No network in the core package.** Network appears only in opt-in sinks and the demo's dataset download.
- **Delete scaffolding.** No debug prints, no commented-out alternatives, no dead flags. Comments are few and useful.
- **Tests.** pytest, fast, every new module gets tests.

## The honesty rule for claims

This project's value depends on every claim being true. In code, docstrings, reports, docs and pull-request descriptions:

- **Every number must be reproducible** from a script in the repository, and state its conditions (hardware, seed, dataset). A number you cannot reproduce is labelled `DESIGNED`, `OPEN` or `ESTIMATE`, or it is not written.
- **Never claim** that the tool makes anyone compliant with any regulation; that the certificate applies to transformers or LLMs; that the tool reads or validates training data; that it is the first to detect dead units; or that it has been tested on hardware or at a scale where it has not.
- **A proof is a proof only under its assumptions.** If you change when a certificate is issued, update `docs/concepts/dead-channels.md` and add a test for the case where it must be withheld.
- **No hype words.** "Revolutionary", "first-ever", "AI-powered" and their relatives do not appear.

The list of things the project does not claim is in [`docs/limitations.md`](docs/limitations.md). If your change would contradict it, the change or the page must be updated in the same pull request.

## Reporting a finding that looks wrong

If `check` or the monitor reports something you believe is incorrect, open an issue with the "A finding looks wrong" template. Include the finding's JSON (`ormas-monitor check FILE --json`, or the event line from `diary.jsonl`) and, if you can, how to build the model. False certificates are treated as the most serious kind of bug.

## Releasing (maintainers)

1. Bump `__version__` in `src/ormas_monitor/__init__.py` (the only place the version lives; `CITATION.cff` carries a copy), move the "Unreleased" entries in `CHANGELOG.md` under the new version, and merge to `main`.
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`. The release workflow checks that the tag matches `__version__`, builds, runs `twine check`, publishes to PyPI and creates a GitHub release.

One-time setup, before the first release:

- **PyPI Trusted Publishing.** On pypi.org, under *Your account → Publishing → Add a new pending publisher*, enter: PyPI project name `ormas-monitor`, owner `Oxiedo`, repository `ormas-monitor`, workflow `release.yml`, environment `pypi`. No API token is created or stored.
- **GitHub environment.** In the repository, *Settings → Environments → New environment* named `pypi`. Optionally require a reviewer, so every publish needs a click.
- **GitHub Pages.** *Settings → Pages → Source: GitHub Actions*, so `docs.yml` can deploy the site.

## Security

Do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contributions are licensed under the Apache License 2.0, as the rest of the project.
