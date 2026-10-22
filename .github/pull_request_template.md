## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## How it was tested

<!-- Commands run and their result, e.g. `pytest -q`, `ruff check .`, `mkdocs build --strict`. -->

## Checklist

- [ ] Tests added or updated; `pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] Docs updated (`docs/`, `README.md`, `CHANGELOG.md` under "Unreleased") if behaviour, flags or output changed.
- [ ] Every new number states its conditions and can be reproduced from a script in the repo, or is labelled DESIGNED / OPEN / ESTIMATE.
- [ ] No new claim contradicts `docs/limitations.md` (compliance, transformers, data, "first", untested hardware).
- [ ] If this changes when a certificate is issued: `docs/concepts/dead-channels.md` is updated and a test covers the case where it must be withheld.
- [ ] No network calls added to the core package.
